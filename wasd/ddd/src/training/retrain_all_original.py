"""
Retrain all 4 models on the ORIGINAL (clean) dataset.

Goals:
  1. Fix data leakage — train on original splits (0 overlap)
  2. Generalize to real-world internet X-rays
  3. Produce accurate Grad-CAM heatmaps that pinpoint fracture location

Key design decisions for heatmap accuracy:
  - Use 224x224 (standard) — higher res not needed for these architectures
  - Medical-appropriate augmentation (no excessive spatial distortion)
  - Keep rotation moderate (15 deg) — bone orientation matters
  - No random erasing on fracture classes (would hide fracture region)
  - Progressive unfreezing: freeze backbone first, then fine-tune all
  - This forces the head to learn meaningful class separation first,
    so gradients later flow through feature maps that already encode
    discriminative spatial patterns, yielding sharper Grad-CAM heatmaps

Training strategy:
  Phase 1 (10 epochs): Freeze backbone, train only classifier head
  Phase 2 (40 epochs): Unfreeze all, fine-tune with lower LR

Run:
    cd /home/praneeth4265/wasd/ddd
    /home/praneeth4265/wasd/ddd/ml_env_linux/bin/python src/training/retrain_all_original.py
"""

import os
import sys
import json
import time
import copy
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
import timm
import numpy as np

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

CLASSES = ["comminuted_fracture", "no_fracture", "simple_fracture"]

MODELS = [
    {
        "name": "MaxViT-Tiny",
        "timm_name": "maxvit_tiny_tf_224.in1k",
        "save_name": "maxvit_3class_original_best.pth",
        "batch_size": 16,
        "freeze_prefix": ["stages"],  # freeze backbone stages
    },
    {
        "name": "Swin Transformer",
        "timm_name": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
        "save_name": "swin_3class_original_best.pth",
        "batch_size": 24,
        "freeze_prefix": ["layers", "patch_embed"],
    },
    {
        "name": "EfficientNetV2-S",
        "timm_name": "tf_efficientnetv2_s.in21k_ft_in1k",
        "save_name": "efficientnetv2_3class_original_best.pth",
        "batch_size": 24,
        "freeze_prefix": ["blocks", "conv_stem", "bn1"],
    },
    {
        "name": "ConvNeXt V2",
        "timm_name": "convnextv2_base.fcmae_ft_in22k_in1k",
        "save_name": "convnextv2_3class_original_best.pth",
        "batch_size": 12,
        "freeze_prefix": ["stages", "stem"],
    },
]

CONFIG = {
    "img_size": 224,
    # Phase 1: head-only training
    "phase1_epochs": 10,
    "phase1_lr": 5e-4,
    # Phase 2: full fine-tuning
    "phase2_epochs": 40,
    "phase2_lr": 2e-5,
    # Shared
    "weight_decay": 1e-2,
    "label_smoothing": 0.1,
    "early_stopping_patience": 12,
    "num_workers": 4,
    "grad_clip_norm": 1.0,
    "warmup_epochs": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

DATA_DIR = PROJECT_ROOT / "datasets" / "original"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
CKPT_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "models" / "results"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BoneFractureDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        data_dir = Path(data_dir)
        for ci, cn in enumerate(CLASSES):
            cd = data_dir / cn
            if not cd.exists():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"):
                for p in cd.glob(ext):
                    self.images.append(str(p))
                    self.labels.append(ci)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # CRITICAL: Convert to grayscale FIRST, then back to 3-channel.
        # This eliminates the dataset bias where no_fracture = pure grayscale PNG
        # while fracture images = JPEGs with color channel differences.
        # Without this, models learn to classify by color artifacts, not anatomy.
        img = Image.open(self.images[idx]).convert("L").convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    def class_counts(self):
        c = Counter(self.labels)
        return {CLASSES[k]: v for k, v in sorted(c.items())}


# ---------------------------------------------------------------------------
# Transforms — medical-appropriate augmentation
# ---------------------------------------------------------------------------
# IMPORTANT: All images are converted to grayscale in the Dataset.__getitem__()
# to eliminate the color-channel bias in this dataset (no_fracture = pure gray PNG,
# fracture = JPG with color artifacts). After grayscale conversion, we replicate
# to 3-channel for the pretrained ImageNet models.
#
# Augmentation is moderate to preserve anatomical structure
# so Grad-CAM learns to focus on the actual fracture region.
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # no saturation/hue — images are grayscale
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def freeze_backbone(model, prefixes):
    """Freeze all parameters whose name starts with any of the given prefixes."""
    frozen = 0
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in prefixes):
            param.requires_grad = False
            frozen += 1
    total = sum(1 for _ in model.parameters())
    print(f"    Frozen {frozen}/{total} parameter tensors")


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"    Unfrozen all — {trainable} trainable parameter tensors")


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_clip):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, 100.0 * correct / total, np.array(all_preds), np.array(all_labels)


def cosine_lr(optimizer, epoch, total_epochs, base_lr, warmup_epochs):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / (warmup_epochs + 1)
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ---------------------------------------------------------------------------
# Train one model (2-phase)
# ---------------------------------------------------------------------------
def train_model(model_cfg, config):
    print("\n" + "=" * 80)
    print(f"  TRAINING: {model_cfg['name']}")
    print(f"  Architecture: {model_cfg['timm_name']}")
    print("=" * 80)

    device = config["device"]
    bs = model_cfg["batch_size"]

    # Datasets
    train_ds = BoneFractureDataset(TRAIN_DIR, train_transform)
    val_ds = BoneFractureDataset(VAL_DIR, val_transform)
    test_ds = BoneFractureDataset(TEST_DIR, val_transform)
    print(f"\n  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=config["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=config["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=config["num_workers"], pin_memory=True)

    # Model
    model = timm.create_model(model_cfg["timm_name"], pretrained=True,
                              num_classes=len(CLASSES)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params / 1e6:.1f}M")

    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    scaler = GradScaler()

    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    all_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [], "phase": []}

    t0 = time.time()
    global_epoch = 0

    # ── PHASE 1: Head-only ──────────────────────────────────────────
    print(f"\n  --- Phase 1: Head-only ({config['phase1_epochs']} epochs, lr={config['phase1_lr']}) ---")
    freeze_backbone(model, model_cfg["freeze_prefix"])
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=config["phase1_lr"], weight_decay=config["weight_decay"])

    for ep in range(1, config["phase1_epochs"] + 1):
        global_epoch += 1
        ep_t0 = time.time()
        lr = cosine_lr(optimizer, ep - 1, config["phase1_epochs"], config["phase1_lr"], config["warmup_epochs"])
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, config["grad_clip_norm"])
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        all_history["train_loss"].append(train_loss)
        all_history["train_acc"].append(train_acc)
        all_history["val_loss"].append(val_loss)
        all_history["val_acc"].append(val_acc)
        all_history["lr"].append(lr)
        all_history["phase"].append(1)

        tag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = global_epoch
            best_state = copy.deepcopy(model.state_dict())
            tag = " *BEST*"
        print(f"    P1 Ep {ep:2d}  train {train_loss:.4f}/{train_acc:.1f}%  val {val_loss:.4f}/{val_acc:.1f}%  lr={lr:.2e}  ({time.time()-ep_t0:.0f}s){tag}")

    # ── PHASE 2: Full fine-tuning ───────────────────────────────────
    print(f"\n  --- Phase 2: Full fine-tune ({config['phase2_epochs']} epochs, lr={config['phase2_lr']}) ---")
    unfreeze_all(model)
    optimizer = optim.AdamW(model.parameters(), lr=config["phase2_lr"], weight_decay=config["weight_decay"])
    patience_counter = 0

    for ep in range(1, config["phase2_epochs"] + 1):
        global_epoch += 1
        ep_t0 = time.time()
        lr = cosine_lr(optimizer, ep - 1, config["phase2_epochs"], config["phase2_lr"], config["warmup_epochs"])
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, config["grad_clip_norm"])
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        all_history["train_loss"].append(train_loss)
        all_history["train_acc"].append(train_acc)
        all_history["val_loss"].append(val_loss)
        all_history["val_acc"].append(val_acc)
        all_history["lr"].append(lr)
        all_history["phase"].append(2)

        tag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = global_epoch
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            tag = " *BEST*"
        else:
            patience_counter += 1

        print(f"    P2 Ep {ep:2d}  train {train_loss:.4f}/{train_acc:.1f}%  val {val_loss:.4f}/{val_acc:.1f}%  lr={lr:.2e}  ({time.time()-ep_t0:.0f}s){tag}")

        if patience_counter >= config["early_stopping_patience"]:
            print(f"\n    Early stopping (no improvement for {patience_counter} epochs)")
            break

    train_time = time.time() - t0
    print(f"\n  Training time: {train_time / 60:.1f} min | Best val: {best_val_acc:.2f}% @ epoch {best_epoch}")

    # Restore best & save
    model.load_state_dict(best_state)
    save_path = CKPT_DIR / model_cfg["save_name"]
    torch.save({"epoch": best_epoch, "model_state_dict": best_state, "val_acc": best_val_acc,
                "config": {**config, **model_cfg}}, save_path)
    print(f"  Saved: {save_path}")

    # Test
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"\n  TEST: loss={test_loss:.4f}  acc={test_acc:.2f}%")
    for ci, cn in enumerate(CLASSES):
        mask = test_labels == ci
        if mask.sum() > 0:
            cls_acc = 100.0 * (test_preds[mask] == ci).sum() / mask.sum()
            print(f"    {cn}: {cls_acc:.1f}% ({mask.sum()} samples)")

    # Save results
    results = {
        "model": model_cfg["timm_name"], "model_name": model_cfg["name"],
        "dataset": "original (clean, no leakage)",
        "classes": CLASSES, "img_size": config["img_size"],
        "best_epoch": best_epoch, "best_val_acc": best_val_acc,
        "test_acc": test_acc, "test_loss": test_loss,
        "training_time_minutes": round(train_time / 60, 2),
        "total_parameters": total_params,
        "training_strategy": "2-phase (head-only then full fine-tune)",
        "augmentation": "moderate medical (rotation 15, flip, crop, color jitter, blur)",
        "history": all_history,
        "train_size": len(train_ds), "val_size": len(val_ds), "test_size": len(test_ds),
    }
    rname = model_cfg["save_name"].replace("_best.pth", "_results.json")
    with open(RESULTS_DIR / rname, "w") as f:
        json.dump(results, f, indent=2)

    del model, optimizer, scaler, best_state
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("  RETRAINING ON CLEAN ORIGINAL DATASET (2-phase for heatmap accuracy)")
    print("=" * 80)
    print(f"  Device: {CONFIG['device']}")
    if CONFIG["device"] == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Phase 1: {CONFIG['phase1_epochs']} epochs (head-only, lr={CONFIG['phase1_lr']})")
    print(f"  Phase 2: {CONFIG['phase2_epochs']} epochs (full fine-tune, lr={CONFIG['phase2_lr']})")
    print()

    all_results = []
    total_t0 = time.time()

    for mcfg in MODELS:
        try:
            all_results.append(train_model(mcfg, CONFIG))
        except Exception as e:
            print(f"\n  ERROR training {mcfg['name']}: {e}")
            import traceback; traceback.print_exc()

    total_time = time.time() - total_t0
    print("\n" + "=" * 80)
    print("  RETRAINING COMPLETE")
    print("=" * 80)
    print(f"  Total: {total_time / 60:.1f} min\n")
    print(f"  {'Model':<25} {'Val':>8} {'Test':>8} {'Epochs':>8} {'Time':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in all_results:
        print(f"  {r['model_name']:<25} {r['best_val_acc']:>7.2f}% {r['test_acc']:>7.2f}% "
              f"{r['best_epoch']:>8} {r['training_time_minutes']:>6.1f}m")

    summary = {"retrained_at": datetime.now().isoformat(), "dataset": "original",
               "total_minutes": round(total_time / 60, 2), "models": all_results}
    with open(RESULTS_DIR / "retrain_original_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("=" * 80)


if __name__ == "__main__":
    main()
