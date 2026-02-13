"""
Interpretability analysis using Grad-CAM for the 4-model ensemble.

Outputs under project_reports/ensemble_eval/interpretability/:
- gradcam_samples.pdf
- gradcam_samples.csv
- per-image PNGs (one per sample)
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import timm

try:
    from pytorch_grad_cam import GradCAMPlusPlus, ScoreCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_PGCAM = True
except ImportError:  # pragma: no cover
    HAS_PGCAM = False

CLASSES = ["comminuted_fracture", "no_fracture", "simple_fracture"]
DISPLAY_CLASSES = ["Comminuted Fracture", "No Fracture", "Simple Fracture"]
RESAMPLE_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")

MODEL_CONFIGS = [
    {
        "key": "convnextv2",
        "display": "ConvNeXt V2 Base",
        "timm_name": "convnextv2_base.fcmae_ft_in22k_in1k",
        "checkpoint": "bone_fracture_detection/models/convnextv2_3class_augmented_best.pth",
    },
    {
        "key": "efficientnetv2",
        "display": "EfficientNetV2-S",
        "timm_name": "tf_efficientnetv2_s.in21k_ft_in1k",
        "checkpoint": "bone_fracture_detection/models/efficientnetv2_3class_augmented_best.pth",
    },
    {
        "key": "maxvit",
        "display": "MaxViT-Tiny",
        "timm_name": "maxvit_tiny_tf_224.in1k",
        "checkpoint": "bone_fracture_detection/models/maxvit_3class_augmented_best.pth",
    },
    {
        "key": "swin",
        "display": "Swin Transformer (Tiny)",
        "timm_name": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
        "checkpoint": "bone_fracture_detection/models/swin_3class_augmented_best.pth",
    },
]


def reshape_transform_swin(tensor: torch.Tensor) -> torch.Tensor:
    # Swin target layer outputs BHWC; Grad-CAM expects BCHW
    if tensor.dim() == 4:
        return tensor.permute(0, 3, 1, 2)
    return tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def _compute_cam(self, method: str) -> torch.Tensor:
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture gradients/activations")
        gradients = self.gradients[0]
        activations = self.activations[0]

        if method == "gradcampp":
            grads_power_2 = gradients ** 2
            grads_power_3 = gradients ** 3
            sum_acts = (activations * grads_power_3).sum(dim=(1, 2), keepdim=True)
            eps = 1e-8
            alpha = grads_power_2 / (2 * grads_power_2 + sum_acts + eps)
            positive_grads = F.relu(gradients)
            weights = (alpha * positive_grads).sum(dim=(1, 2), keepdim=True)
        else:
            weights = gradients.mean(dim=(1, 2), keepdim=True)

        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def generate(
        self,
        image_tensor: torch.Tensor,
        class_idx: int | None = None,
        method: str = "gradcampp",
        smooth: int = 1,
        noise_sigma: float = 0.15,
    ) -> Tuple[np.ndarray, int, torch.Tensor]:
        self.model.eval()
        cams = []
        output = None
        resolved_class = class_idx

        for _ in range(max(1, smooth)):
            if smooth > 1:
                noise = torch.randn_like(image_tensor) * noise_sigma
                input_tensor = image_tensor + noise
            else:
                input_tensor = image_tensor

            output = self.model(input_tensor)
            if resolved_class is None:
                resolved_class = int(output.argmax(dim=1).item())
            self.model.zero_grad()
            class_score = output[0, resolved_class]
            class_score.backward(retain_graph=True)

            cam_tensor = self._compute_cam(method)
            cams.append(cam_tensor.cpu().numpy())

        if output is None or resolved_class is None:
            raise RuntimeError("Failed to compute Grad-CAM output")
        cam = np.mean(cams, axis=0)
        return cam, int(resolved_class), output

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()


def get_target_layer(model: torch.nn.Module, model_key: str) -> torch.nn.Module:
    if model_key == "convnextv2":
        return cast(torch.nn.Module, cast(Any, model).stages[-1])
    if model_key == "efficientnetv2":
        return cast(torch.nn.Module, cast(Any, model).conv_head)
    if model_key == "maxvit":
        return cast(torch.nn.Module, cast(Any, model).stages[-1])
    if model_key == "swin":
        return cast(torch.nn.Module, cast(Any, model).layers[-1])
    raise KeyError(f"Unknown model key: {model_key}")


def load_model(config: Dict, device: torch.device) -> torch.nn.Module:
    model = timm.create_model(config["timm_name"], pretrained=False, num_classes=len(CLASSES))
    checkpoint = torch.load(config["checkpoint"], map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def collect_samples(data_dir: Path, samples_per_class: int, seed: int) -> List[Tuple[Path, int]]:
    set_seed(seed)
    samples: List[Tuple[Path, int]] = []
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = data_dir / class_name
        images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        if not images:
            continue
        random.shuffle(images)
        for img_path in images[:samples_per_class]:
            samples.append((img_path, class_idx))
    return samples


def normalize_heatmap(
    heatmap: np.ndarray,
    percentile: float,
    blur_radius: float,
    threshold: float,
) -> np.ndarray:
    heatmap = np.maximum(heatmap, 0)
    if percentile > 0:
        clip_val = np.percentile(heatmap, percentile)
        if clip_val > 0:
            heatmap = np.clip(heatmap, 0, clip_val)
    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    if blur_radius > 0:
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img = heatmap_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        heatmap = np.array(heatmap_img).astype(np.float32) / 255.0
    if threshold > 0:
        heatmap = np.where(heatmap >= threshold, heatmap, 0)
    return heatmap


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    cmap: str = "jet",
) -> np.ndarray:
    if heatmap.shape[:2] != (image.size[1], image.size[0]):
        heatmap_resized = np.array(Image.fromarray(heatmap).resize(image.size, RESAMPLE_BILINEAR))
    else:
        heatmap_resized = heatmap
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap="gray")
    alpha_map = np.clip(heatmap_resized * alpha, 0, 1)
    ax.imshow(heatmap_resized, cmap=cmap, alpha=alpha_map)
    ax.axis("off")
    canvas = cast(FigureCanvasAgg, fig.canvas)
    canvas.draw()
    buffer = np.asarray(canvas.buffer_rgba())
    overlay = buffer[:, :, :3].copy()
    plt.close(fig)
    return overlay


def occlusion_sensitivity(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_idx: int,
    patch_size: int,
    stride: int,
    baseline: str = "mean",
) -> np.ndarray:
    model.eval()
    _, _, h, w = image_tensor.shape
    if baseline == "zero":
        fill = torch.zeros_like(image_tensor)
    else:
        fill_value = float(image_tensor.mean().item())
        fill = torch.full_like(image_tensor, fill_value)

    with torch.no_grad():
        base_probs = F.softmax(model(image_tensor), dim=1)[0]
        base_score = base_probs[target_idx].item()

    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(y + patch_size, h)
            x2 = min(x + patch_size, w)
            occluded = image_tensor.clone()
            occluded[:, :, y:y2, x:x2] = fill[:, :, y:y2, x:x2]
            with torch.no_grad():
                probs = F.softmax(model(occluded), dim=1)[0]
                score = probs[target_idx].item()
            drop = max(0.0, base_score - score)
            heatmap[y:y2, x:x2] += drop
            counts[y:y2, x:x2] += 1

    counts[counts == 0] = 1
    heatmap = heatmap / counts
    return heatmap


def create_sample_figure(
    original: Image.Image,
    overlays: Dict[str, np.ndarray],
    predictions: Dict[str, Dict[str, float]],
    true_label: str,
    sample_title: str,
) -> Figure:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title(f"Original\nTrue: {true_label}")
    axes[0].axis("off")

    order = [
        "convnextv2",
        "efficientnetv2",
        "maxvit",
        "swin",
        "ensemble",
    ]

    for idx, key in enumerate(order, start=1):
        axes[idx].imshow(overlays[key])
        pred = predictions[key]
        axes[idx].set_title(
            f"{pred['display']}\nPred: {pred['label']} ({pred['conf']:.2f})",
            fontsize=10,
        )
        axes[idx].axis("off")

    fig.suptitle(sample_title, fontsize=14)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM interpretability artifacts")
    parser.add_argument("--data-dir", default="datasets/augmented/test")
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="project_reports/ensemble_eval/interpretability")
    parser.add_argument("--cam-target", choices=["pred", "true"], default="true")
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--blur", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--cmap", default="jet")
    parser.add_argument("--method", choices=["gradcam", "gradcampp"], default="gradcampp")
    parser.add_argument("--smooth", type=int, default=4)
    parser.add_argument("--noise-sigma", type=float, default=0.15)
    parser.add_argument("--engine", choices=["pgcam", "scorecam", "custom", "occlusion"], default="scorecam")
    parser.add_argument("--aug-smooth", action="store_true")
    parser.add_argument("--eigen-smooth", action="store_true")
    parser.add_argument("--occ-patch", type=int, default=32)
    parser.add_argument("--occ-stride", type=int, default=16)
    parser.add_argument("--occ-baseline", choices=["mean", "zero"], default="mean")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_dir = Path(args.data_dir)
    samples = collect_samples(data_dir, args.samples_per_class, args.seed)
    if not samples:
        raise RuntimeError("No samples found. Check --data-dir.")

    models = {}
    cams = {}
    pgcams = {}
    scorecams = {}
    target_layers = {}
    for config in MODEL_CONFIGS:
        model = load_model(config, device)
        target_layer = get_target_layer(model, config["key"])
        models[config["key"]] = model
        target_layers[config["key"]] = target_layer
        cams[config["key"]] = GradCAM(model, target_layer)
        if args.engine in {"pgcam", "scorecam"}:
            if not HAS_PGCAM:
                raise RuntimeError("pytorch-grad-cam is not installed. Install it to use --engine pgcam.")
            reshape_transform = reshape_transform_swin if config["key"] == "swin" else None
            pgcams[config["key"]] = GradCAMPlusPlus(
                model=model,
                target_layers=[target_layer],
                reshape_transform=reshape_transform,
            )
            scorecams[config["key"]] = ScoreCAM(
                model=model,
                target_layers=[target_layer],
                reshape_transform=reshape_transform,
            )

    pdf_path = output_dir / "gradcam_samples.pdf"
    csv_path = output_dir / "gradcam_samples.csv"

    rows = []

    with PdfPages(pdf_path) as pdf:
        for idx, (img_path, true_idx) in enumerate(samples, start=1):
            original = Image.open(img_path).convert("RGB")
            image_tensor = cast(torch.Tensor, transform(original))
            image_tensor = image_tensor.unsqueeze(0).to(device)

            overlays = {}
            predictions = {}
            heatmaps = []
            model_probs = []

            for config in MODEL_CONFIGS:
                key = config["key"]
                model = models[key]
                output = model(image_tensor)
                probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()
                pred_idx = int(np.argmax(probs))
                target_idx = true_idx if args.cam_target == "true" else pred_idx

                if args.engine == "pgcam":
                    cam_engine = pgcams[key]
                    cam_map = cam_engine(
                        input_tensor=image_tensor,
                        targets=[ClassifierOutputTarget(target_idx)],
                        aug_smooth=args.aug_smooth,
                        eigen_smooth=args.eigen_smooth,
                    )
                    heatmap = cam_map[0]
                elif args.engine == "scorecam":
                    cam_engine = scorecams[key]
                    cam_map = cam_engine(
                        input_tensor=image_tensor,
                        targets=[ClassifierOutputTarget(target_idx)],
                    )
                    heatmap = cam_map[0]
                elif args.engine == "occlusion":
                    heatmap = occlusion_sensitivity(
                        model,
                        image_tensor,
                        target_idx,
                        patch_size=args.occ_patch,
                        stride=args.occ_stride,
                        baseline=args.occ_baseline,
                    )
                else:
                    cam = cams[key]
                    heatmap, pred_idx, output = cam.generate(
                        image_tensor,
                        class_idx=target_idx,
                        method=args.method,
                        smooth=args.smooth,
                        noise_sigma=args.noise_sigma,
                    )
                conf = float(probs[pred_idx])

                heatmap_resized = np.array(Image.fromarray(heatmap).resize(original.size, RESAMPLE_BILINEAR))
                heatmap_resized = normalize_heatmap(
                    heatmap_resized,
                    args.percentile,
                    args.blur,
                    args.threshold,
                )
                overlays[key] = overlay_heatmap(
                    original,
                    heatmap_resized,
                    alpha=args.alpha,
                    cmap=args.cmap,
                )
                predictions[key] = {
                    "display": config["display"],
                    "label": DISPLAY_CLASSES[pred_idx],
                    "conf": conf,
                }
                heatmaps.append(heatmap_resized)
                model_probs.append(probs)

            # Ensemble heatmap: mean of normalized heatmaps
            ensemble_heatmap = np.mean(heatmaps, axis=0)
            ensemble_heatmap = normalize_heatmap(
                ensemble_heatmap,
                args.percentile,
                args.blur,
                args.threshold,
            )

            ensemble_overlay = overlay_heatmap(
                original,
                ensemble_heatmap,
                alpha=args.alpha,
                cmap=args.cmap,
            )
            overlays["ensemble"] = ensemble_overlay

            ensemble_probs = np.mean(model_probs, axis=0)
            ensemble_pred_idx = int(np.argmax(ensemble_probs))
            ensemble_conf = float(ensemble_probs[ensemble_pred_idx])
            predictions["ensemble"] = {
                "display": "Ensemble (Avg CAM)",
                "label": DISPLAY_CLASSES[ensemble_pred_idx],
                "conf": ensemble_conf,
            }

            sample_title = f"Sample {idx}: {img_path.name}"
            fig = create_sample_figure(
                original=original,
                overlays=overlays,
                predictions=predictions,
                true_label=DISPLAY_CLASSES[true_idx],
                sample_title=sample_title,
            )

            png_path = output_dir / f"gradcam_{idx:02d}_{img_path.stem}.png"
            fig.savefig(png_path, dpi=180)
            pdf.savefig(fig)
            plt.close(fig)

            row = {
                "sample_index": idx,
                "image_path": str(img_path),
                "true_label": DISPLAY_CLASSES[true_idx],
                "output_path": str(png_path),
            }
            for key in ["convnextv2", "efficientnetv2", "maxvit", "swin", "ensemble"]:
                row[f"{key}_pred"] = predictions[key]["label"]
                row[f"{key}_conf"] = f"{predictions[key]['conf']:.4f}"
            rows.append(row)

    for cam in cams.values():
        cam.close()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "image_path",
                "true_label",
                "output_path",
                "convnextv2_pred",
                "convnextv2_conf",
                "efficientnetv2_pred",
                "efficientnetv2_conf",
                "maxvit_pred",
                "maxvit_conf",
                "swin_pred",
                "swin_conf",
                "ensemble_pred",
                "ensemble_conf",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Grad-CAM outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
