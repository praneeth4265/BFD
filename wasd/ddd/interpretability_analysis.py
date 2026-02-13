"""
Interpretability – multi-scale occlusion sensitivity with LOGIT scoring (v3).

Key improvements (v3):
  1. BLUR baseline instead of gray — blurring removes high-frequency features
     (fracture lines, edges) while preserving overall bone structure, so the
     logit drop specifically highlights what makes fractures visible.
  2. Raw LOGIT drop (not softmax) — avoids saturation at 99.9%+ confidence.
  3. 4 scales (4:2, 8:4, 16:8, 32:16) — finest scale captures thin fracture
     lines, coarsest gives broad region context.
  4. Weighted fusion — fine scales get MORE weight because fracture lines
     are thin high-frequency features.
  5. Noise-floor thresholding — zeroes bottom 35% of heatmap to suppress
     diffuse background noise, making hot spots much more focal.
  6. GPU-batched for speed.

Outputs:
  project_reports/ensemble_eval/interpretability/
    - gradcam_samples.pdf
    - gradcam_samples.csv
    - per-image PNGs (2×3 grid: Original + 4 models + Ensemble)
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import timm

CLASSES = ["comminuted_fracture", "no_fracture", "simple_fracture"]
DISPLAY_CLASSES = ["Comminuted Fracture", "No Fracture", "Simple Fracture"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224

MODEL_CONFIGS = [
    dict(key="convnextv2", display="ConvNeXt V2 Base",
         timm_name="convnextv2_base.fcmae_ft_in22k_in1k",
         checkpoint="bone_fracture_detection/models/convnextv2_3class_augmented_best.pth"),
    dict(key="efficientnetv2", display="EfficientNetV2-S",
         timm_name="tf_efficientnetv2_s.in21k_ft_in1k",
         checkpoint="bone_fracture_detection/models/efficientnetv2_3class_augmented_best.pth"),
    dict(key="maxvit", display="MaxViT-Tiny",
         timm_name="maxvit_tiny_tf_224.in1k",
         checkpoint="bone_fracture_detection/models/maxvit_3class_augmented_best.pth"),
    dict(key="swin", display="Swin Transformer (Tiny)",
         timm_name="swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
         checkpoint="bone_fracture_detection/models/swin_3class_augmented_best.pth"),
]


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def load_model(cfg: Dict, device: torch.device) -> torch.nn.Module:
    m = timm.create_model(cfg["timm_name"], pretrained=False, num_classes=len(CLASSES))
    ckpt = torch.load(cfg["checkpoint"], map_location=device)
    m.load_state_dict(ckpt.get("model_state_dict", ckpt))
    m.to(device).eval()
    return m


def collect_samples(data_dir: Path, n: int, seed: int) -> List[Tuple[Path, int]]:
    set_seed(seed)
    out: List[Tuple[Path, int]] = []
    for ci, cn in enumerate(CLASSES):
        d = data_dir / cn
        imgs = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
        random.shuffle(imgs)
        for p in imgs[:n]:
            out.append((p, ci))
    return out


# ─────────── GPU-batched occlusion with LOGIT scoring ─────
def _make_blur_baseline(
    inp: torch.Tensor,    # (1, C, H, W) normalised input
    blur_radius: int = 11,
) -> torch.Tensor:
    """
    Blur baseline: Gaussian-blur the input image.
    This removes high-frequency features (fracture lines, edges, texture)
    while preserving overall bone shape/position/brightness.
    Logit drop = what the model loses when fine details are erased
    → specifically highlights fracture lines and sharp edges.
    """
    # Apply Gaussian blur in normalised space using avg_pool trick
    # Use a proper Gaussian kernel
    C, H, W = inp.shape[1], inp.shape[2], inp.shape[3]
    sigma = blur_radius / 3.0
    ks = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
    # Create 1D Gaussian kernel
    x = torch.arange(ks, dtype=torch.float32, device=inp.device) - ks // 2
    gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    # Make 2D separable kernel
    gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]  # (ks, ks)
    gauss_2d = gauss_2d.view(1, 1, ks, ks).repeat(C, 1, 1, 1)  # (C, 1, ks, ks)
    # Apply depthwise convolution
    pad = ks // 2
    blurred = F.conv2d(inp, gauss_2d, padding=pad, groups=C)
    return blurred


def occlusion_logit_gpu(
    model: torch.nn.Module,
    inp: torch.Tensor,          # (1, C, H, W)
    target: int,
    patch_size: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    baseline: torch.Tensor | None = None,
) -> np.ndarray:
    """
    Occlusion sensitivity using raw logit drop (not softmax prob).
    Uses blur baseline (patches replaced with blurred version).
    Returns (H, W) heatmap of logit drops.
    """
    _, C, H, W = inp.shape

    if baseline is None:
        baseline = _make_blur_baseline(inp)

    with torch.no_grad():
        base_logit = model(inp)[0, target].item()

    # Build position list
    positions = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            positions.append((y, x))

    heatmap = np.zeros((H, W), dtype=np.float64)
    counts  = np.zeros((H, W), dtype=np.float64)

    for b0 in range(0, len(positions), batch_size):
        batch_pos = positions[b0 : b0 + batch_size]
        B = len(batch_pos)
        batch = inp.expand(B, -1, -1, -1).clone()
        for i, (y, x) in enumerate(batch_pos):
            batch[i, :, y:y+patch_size, x:x+patch_size] = \
                baseline[0, :, y:y+patch_size, x:x+patch_size]

        with torch.no_grad():
            logits = model(batch)[:, target].cpu().numpy()

        for i, (y, x) in enumerate(batch_pos):
            drop = max(0.0, base_logit - logits[i])
            heatmap[y:y+patch_size, x:x+patch_size] += drop
            counts[y:y+patch_size, x:x+patch_size] += 1.0

    counts[counts == 0] = 1.0
    return (heatmap / counts).astype(np.float32)


def multiscale_occlusion(
    model: torch.nn.Module,
    inp: torch.Tensor,
    target: int,
    scales: List[Tuple[int, int]],   # [(patch, stride), ...]
    batch_size: int,
    device: torch.device,
    noise_floor: float = 0.35,
) -> np.ndarray:
    """
    Run occlusion at multiple scales and fuse with WEIGHTED combination.

    Key improvements:
      - Blur baseline (computed once, reused across scales)
      - Fine scales get higher weight (fracture lines are thin)
      - Noise-floor thresholding zeroes diffuse low-confidence regions
      - Combination: weighted sum (not geometric mean) for better signal
    """
    baseline = _make_blur_baseline(inp)

    maps = []
    for ps, st in scales:
        hm = occlusion_logit_gpu(model, inp, target, ps, st, batch_size,
                                  device, baseline=baseline)
        # Normalise each scale independently to [0, 1]
        if hm.max() > 0:
            hm = hm / hm.max()
        maps.append(hm)

    # Weight: finer scales (smaller patch) → higher weight
    # Rationale: fracture lines are thin, fine scales pick them up best
    # Sort scales by patch size to assign weights
    patch_sizes = [ps for ps, st in scales]
    min_ps = min(patch_sizes)
    # Weight inversely proportional to patch size
    weights = [min_ps / ps for ps in patch_sizes]
    w_total = sum(weights)
    weights = [w / w_total for w in weights]

    # Weighted sum
    fused = np.zeros_like(maps[0])
    for m, w in zip(maps, weights):
        fused += w * m

    # Apply noise-floor thresholding: zero out bottom fraction
    if noise_floor > 0:
        thresh = np.percentile(fused[fused > 0], noise_floor * 100) \
            if np.any(fused > 0) else 0.0
        fused[fused < thresh] = 0.0

    if fused.max() > 0:
        fused = fused / fused.max()
    return fused.astype(np.float32)


# ──────────────────── Normalise + Overlay ──────────────────
def normalise(hm: np.ndarray, pct: float = 97) -> np.ndarray:
    hm = np.maximum(hm, 0).astype(np.float64)
    if hm.max() == 0:
        return hm.astype(np.float32)
    clip = np.percentile(hm, pct)
    if clip > 0:
        hm = np.clip(hm, 0, clip) / clip
    else:
        hm = hm / hm.max()
    return hm.astype(np.float32)


def smooth(hm: np.ndarray, r: float = 2.0) -> np.ndarray:
    pil = Image.fromarray((hm * 255).astype(np.uint8))
    pil = pil.filter(ImageFilter.GaussianBlur(radius=r))
    return np.array(pil).astype(np.float32) / 255.0


def overlay(image: Image.Image, hm: np.ndarray, alpha: float, cmap: str) -> np.ndarray:
    img = np.array(image.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if hm.shape != (IMG_SIZE, IMG_SIZE):
        hm = np.array(Image.fromarray((hm*255).astype(np.uint8)).resize(
            (IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)).astype(np.float32) / 255.0
    cm = plt.get_cmap(cmap)
    hm_rgb = cm(hm)[:, :, :3].astype(np.float32)
    # Per-pixel alpha: stronger where heatmap is hot, zero where cold
    pa = np.power(hm, 0.6)[:, :, np.newaxis] * alpha
    blended = (1.0 - pa) * img + pa * hm_rgb
    return np.clip(blended * 255, 0, 255).astype(np.uint8)


def make_figure(
    orig: Image.Image,
    overlays: Dict[str, np.ndarray],
    preds: Dict[str, Dict],
    true_label: str,
    title: str,
) -> Figure:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    af = axes.flatten()
    af[0].imshow(orig.resize((IMG_SIZE, IMG_SIZE)))
    af[0].set_title(f"Original\nTrue: {true_label}", fontsize=10)
    af[0].axis("off")
    for i, k in enumerate(["convnextv2","efficientnetv2","maxvit","swin","ensemble"], 1):
        af[i].imshow(overlays[k])
        p = preds[k]
        af[i].set_title(f"{p['display']}\nPred: {p['label']} ({p['conf']:.2f})", fontsize=9)
        af[i].axis("off")
    fig.suptitle(f"{title}  [Multi-scale Blur-Occlusion (logit, v3)]", fontsize=13)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="datasets/augmented/test")
    ap.add_argument("--samples-per-class", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="project_reports/ensemble_eval/interpretability")
    ap.add_argument("--cam-target", choices=["pred","true"], default="true")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--percentile", type=float, default=97)
    ap.add_argument("--smooth", type=float, default=2.0)
    ap.add_argument("--scales", default="4:2,8:4,16:8,32:16",
                    help="patch:stride pairs, comma-separated")
    ap.add_argument("--noise-floor", type=float, default=0.35,
                    help="Zero out bottom fraction of heatmap (0-1)")
    args = ap.parse_args()

    scales = []
    for s in args.scales.split(","):
        p, st = s.split(":")
        scales.append((int(p), int(st)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Scales: {scales}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    samples = collect_samples(Path(args.data_dir), args.samples_per_class, args.seed)
    print(f"Samples: {len(samples)} ({args.samples_per_class}/class)")

    models: Dict[str, torch.nn.Module] = {}
    for cfg in MODEL_CONFIGS:
        print(f"  Loading {cfg['display']}...")
        models[cfg["key"]] = load_model(cfg, device)

    rows: List[Dict] = []
    pdf_path = out_dir / "gradcam_samples.pdf"
    csv_path = out_dir / "gradcam_samples.csv"

    with PdfPages(pdf_path) as pdf:
        for idx, (img_path, true_idx) in enumerate(samples, start=1):
            print(f"\n== Sample {idx}/{len(samples)}: {img_path.name}")
            orig = Image.open(img_path).convert("RGB")
            inp = cast(torch.Tensor, tfm(orig)).unsqueeze(0).to(device)

            ovs: Dict[str, np.ndarray] = {}
            pds: Dict[str, Dict] = {}
            hms: List[np.ndarray] = []
            mps: List[np.ndarray] = []

            for cfg in MODEL_CONFIGS:
                k = cfg["key"]
                model = models[k]
                with torch.no_grad():
                    probs = F.softmax(model(inp), dim=1)[0].cpu().numpy()
                pi = int(np.argmax(probs))
                tgt = true_idx if args.cam_target == "true" else pi

                print(f"  {cfg['display']:25s}  pred={DISPLAY_CLASSES[pi]:25s}  "
                      f"conf={probs[pi]:.3f}  target={DISPLAY_CLASSES[tgt]}")

                hm = multiscale_occlusion(model, inp, tgt, scales,
                                          args.batch_size, device,
                                          noise_floor=args.noise_floor)
                hm = normalise(hm, args.percentile)
                if args.smooth > 0:
                    hm = smooth(hm, args.smooth)
                    hm = normalise(hm, 100)

                hms.append(hm)
                mps.append(probs)
                ovs[k] = overlay(orig, hm, args.alpha, args.cmap)
                pds[k] = dict(display=cfg["display"],
                              label=DISPLAY_CLASSES[pi],
                              conf=float(probs[pi]))

            # Ensemble heatmap — weighted average of individual model heatmaps
            ehm = np.mean(hms, axis=0)
            # Apply noise floor to ensemble too
            if args.noise_floor > 0 and np.any(ehm > 0):
                ethresh = np.percentile(ehm[ehm > 0], args.noise_floor * 100)
                ehm[ehm < ethresh] = 0.0
            ehm = normalise(ehm, args.percentile)
            if args.smooth > 0:
                ehm = smooth(ehm, args.smooth)
                ehm = normalise(ehm, 100)
            ovs["ensemble"] = overlay(orig, ehm, args.alpha, args.cmap)

            ep = np.mean(mps, axis=0)
            epi = int(np.argmax(ep))
            pds["ensemble"] = dict(display="Ensemble (Avg)",
                                   label=DISPLAY_CLASSES[epi],
                                   conf=float(ep[epi]))

            title = f"Sample {idx}: {img_path.name}"
            fig = make_figure(orig, ovs, pds, DISPLAY_CLASSES[true_idx], title)
            png = out_dir / f"gradcam_{idx:02d}_{img_path.stem}.png"
            fig.savefig(png, dpi=180)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  -> {png.name}")

            row = dict(sample_index=idx, image_path=str(img_path),
                       true_label=DISPLAY_CLASSES[true_idx], output_path=str(png))
            for k in ["convnextv2","efficientnetv2","maxvit","swin","ensemble"]:
                row[f"{k}_pred"] = pds[k]["label"]
                row[f"{k}_conf"] = f"{pds[k]['conf']:.4f}"
            rows.append(row)

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"\nDone -- {out_dir}")


if __name__ == "__main__":
    main()
