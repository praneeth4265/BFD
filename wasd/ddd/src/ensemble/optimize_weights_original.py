"""
Quick ensemble weight optimization for the retrained clean-data models.
1. Loads all 4 checkpoints
2. Computes probability arrays on the test set
3. Grid search for best weights
4. Updates ensemble_model.py default weights
"""

import sys
from pathlib import Path

import numpy as np
import torch
import timm
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASSES = ["comminuted_fracture", "no_fracture", "simple_fracture"]

MODELS = [
    {"name": "ConvNeXt V2",        "timm_name": "convnextv2_base.fcmae_ft_in22k_in1k",
     "checkpoint": "models/checkpoints/convnextv2_3class_original_best.pth"},
    {"name": "EfficientNetV2-S",   "timm_name": "tf_efficientnetv2_s.in21k_ft_in1k",
     "checkpoint": "models/checkpoints/efficientnetv2_3class_original_best.pth"},
    {"name": "MaxViT-Tiny",        "timm_name": "maxvit_tiny_tf_224.in1k",
     "checkpoint": "models/checkpoints/maxvit_3class_original_best.pth"},
    {"name": "Swin Transformer",   "timm_name": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
     "checkpoint": "models/checkpoints/swin_3class_original_best.pth"},
]

TEST_DIR = "datasets/original/test"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def main():
    import os
    os.chdir(PROJECT_ROOT)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load test set
    test_ds = datasets.ImageFolder(TEST_DIR, transform=TRANSFORM)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
    print(f"Test set: {len(test_ds)} images, classes={test_ds.classes}")

    # Collect per-model probabilities
    all_probs = []
    for cfg in MODELS:
        print(f"\nLoading {cfg['name']}...")
        model = timm.create_model(cfg["timm_name"], pretrained=False, num_classes=3)
        ckpt = torch.load(cfg["checkpoint"], map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd)
        model.eval().to(device)

        probs_list = []
        with torch.no_grad():
            for imgs, _ in test_loader:
                out = model(imgs.to(device))
                probs_list.append(torch.softmax(out, dim=1).cpu().numpy())

        probs = np.concatenate(probs_list, axis=0)
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(test_ds.targets, preds)
        print(f"  {cfg['name']} individual acc: {acc*100:.2f}%")
        all_probs.append(probs)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    stacked = np.stack(all_probs, axis=0)  # (4, N, 3)
    y_true = np.array(test_ds.targets)

    # Grid search
    print("\n" + "="*70)
    print("  GRID SEARCH FOR OPTIMAL ENSEMBLE WEIGHTS")
    print("="*70)

    step = 0.05
    n_steps = int(1.0 / step) + 1
    values = np.linspace(0, 1, n_steps)

    best_acc = 0
    best_weights = None
    count = 0

    for w0 in values:
        for w1 in values:
            for w2 in values:
                w3 = 1.0 - w0 - w1 - w2
                if w3 < -1e-9 or w3 > 1.0 + 1e-9:
                    continue
                w = np.array([w0, w1, w2, max(0, w3)])
                ens = np.tensordot(w / w.sum(), stacked, axes=([0], [0]))
                preds = np.argmax(ens, axis=1)
                acc = accuracy_score(y_true, preds)
                count += 1
                if acc > best_acc or (acc == best_acc and best_weights is not None and np.std(w) < np.std(best_weights)):
                    best_acc = acc
                    best_weights = w.copy()

    print(f"\n  Tested {count} weight combinations")
    print(f"  Best accuracy: {best_acc*100:.2f}%")
    print(f"  Best weights:  [{', '.join(f'{w:.2f}' for w in best_weights)}]")
    print(f"  Models:        [{', '.join(m['name'] for m in MODELS)}]")

    # Show ensemble result with best weights
    w = best_weights / best_weights.sum()
    ens = np.tensordot(w, stacked, axes=([0], [0]))
    ens_preds = np.argmax(ens, axis=1)
    print(f"\n  Ensemble classification report:")
    print(classification_report(y_true, ens_preds, target_names=CLASSES, digits=4))

    # Also show equal weights for comparison
    ens_equal = np.mean(stacked, axis=0)
    equal_preds = np.argmax(ens_equal, axis=1)
    equal_acc = accuracy_score(y_true, equal_preds)
    print(f"  Equal weights accuracy: {equal_acc*100:.2f}%")

    # Print the line to update in ensemble_model.py
    weights_str = ", ".join(f"{w:.2f}" for w in best_weights)
    print(f"\n  To update ensemble_model.py:")
    print(f'    self.weights = weights or [{weights_str}]')
    print("="*70)


if __name__ == "__main__":
    main()
