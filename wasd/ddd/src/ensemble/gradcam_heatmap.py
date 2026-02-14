"""
Grad-CAM heatmap generation for the 4-model ensemble.

Generates per-model and ensemble-averaged heatmaps that highlight
the region the models are looking at for their prediction.

Architecture-specific target layers:
  - MaxViT-Tiny:       stages.3   (last multi-axis attention stage)
  - Swin Transformer:  layers.3   (last shifted-window stage)
  - EfficientNetV2-S:  blocks.5   (last MBConv block group) + conv_head
  - ConvNeXt V2:       stages.3   (last ConvNeXt block stage)

Usage:
    from gradcam_heatmap import GradCAMGenerator
    gen = GradCAMGenerator(ensemble_model)
    heatmap_data = gen.generate(image_tensor)
"""

import io
import base64
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ---------------------------------------------------------------------------
# Target layer mapping per timm model name
# ---------------------------------------------------------------------------
TARGET_LAYER_MAP = {
    "maxvit_tiny_tf_224.in1k": "stages.3",
    "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k": "layers.3",
    "tf_efficientnetv2_s.in21k_ft_in1k": "conv_head",
    "convnextv2_base.fcmae_ft_in22k_in1k": "stages.3",
}

# Transformer architectures need reshape_transform because their
# feature maps are (B, N_tokens, C) not (B, C, H, W)
NEEDS_RESHAPE = {
    "maxvit_tiny_tf_224.in1k": True,
    "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k": True,
    "tf_efficientnetv2_s.in21k_ft_in1k": False,
    "convnextv2_base.fcmae_ft_in22k_in1k": False,
}


def _get_layer_by_name(model, layer_name: str):
    """Resolve a dot-separated layer name to the actual module."""
    parts = layer_name.split(".")
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def _swin_reshape_transform(tensor, height=7, width=7):
    """Reshape (B, H*W, C) -> (B, C, H, W) for Swin. Handle 4D pass-through."""
    if tensor.dim() == 4:
        return tensor
    B, N, C = tensor.size()
    h = w = int(N ** 0.5)
    return tensor.permute(0, 2, 1).reshape(B, C, h, w)


def _maxvit_reshape_transform(tensor, height=7, width=7):
    """Reshape for MaxViT. The output of stages.3 is (B, C, H, W) already
    but the internal attention blocks produce (B, N, C). We handle both."""
    if tensor.dim() == 4:
        return tensor
    B, N, C = tensor.size()
    h = w = int(N ** 0.5)
    return tensor.permute(0, 2, 1).reshape(B, C, h, w)


def get_reshape_transform(timm_name: str):
    """Return the appropriate reshape transform or None."""
    if not NEEDS_RESHAPE.get(timm_name, False):
        return None
    if "swin" in timm_name:
        return _swin_reshape_transform
    if "maxvit" in timm_name:
        return _maxvit_reshape_transform
    return None


class GradCAMGenerator:
    """Generate Grad-CAM heatmaps for the ensemble models."""

    def __init__(self, ensemble_model, method="gradcam"):
        """
        Args:
            ensemble_model: EnsembleModel instance with .models, .model_names, .weights
            method: 'gradcam' or 'gradcam++' 
        """
        self.ensemble = ensemble_model
        self.method = method
        self.cam_objects = []

        # Model config lookup (need timm_name for each model)
        from ensemble_model import DEFAULT_MODELS
        timm_names = {cfg["name"]: cfg["timm_name"] for cfg in DEFAULT_MODELS}

        for i, (model, name) in enumerate(zip(ensemble_model.models, ensemble_model.model_names)):
            timm_name = timm_names[name]
            layer_name = TARGET_LAYER_MAP.get(timm_name)
            if layer_name is None:
                self.cam_objects.append(None)
                continue

            target_layer = _get_layer_by_name(model, layer_name)
            reshape_fn = get_reshape_transform(timm_name)

            CAMClass = GradCAMPlusPlus if method == "gradcam++" else GradCAM
            cam = CAMClass(
                model=model,
                target_layers=[target_layer],
                reshape_transform=reshape_fn,
            )
            self.cam_objects.append(cam)

    def generate(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
    ) -> Dict:
        """
        Generate heatmaps for a single image.

        Args:
            input_tensor: (1, 3, 224, 224) normalized tensor on device
            original_image: (H, W, 3) uint8 numpy array (original image before normalization)
            target_class: class index to explain (None = predicted class)

        Returns:
            dict with keys:
                "per_model": [{name, heatmap_b64, active, weight}, ...]
                "ensemble": str (base64 PNG of ensemble heatmap)
                "predicted_class": int
                "predicted_name": str
        """
        # Get ensemble prediction first
        results = self.ensemble.predict(input_tensor)
        pred = results[0]
        from ensemble_model import CLASSES
        pred_idx = CLASSES.index(pred.prediction) if target_class is None else target_class

        targets = [ClassifierOutputTarget(pred_idx)]

        # Prepare display image (0-1 float, 224x224)
        display_img = cv2.resize(original_image, (224, 224))
        display_float = display_img.astype(np.float32) / 255.0

        per_model = []
        weighted_cams = []

        for i, (cam_obj, name) in enumerate(zip(self.cam_objects, self.ensemble.model_names)):
            weight = self.ensemble.weights[i]
            active = weight > 0

            if cam_obj is None:
                per_model.append({
                    "name": name, "heatmap_b64": None,
                    "active": active, "weight": weight,
                })
                continue

            # Generate Grad-CAM
            grayscale_cam = cam_obj(input_tensor=input_tensor, targets=targets)
            cam_image = grayscale_cam[0]  # (224, 224) float 0-1

            if active:
                weighted_cams.append((cam_image, weight))

            # Overlay on original
            overlay = show_cam_on_image(display_float, cam_image, use_rgb=True)
            overlay_b64 = _numpy_to_b64png(overlay)

            per_model.append({
                "name": name,
                "heatmap_b64": overlay_b64,
                "active": active,
                "weight": weight,
            })

        # Ensemble heatmap (weighted average of active models)
        if weighted_cams:
            total_w = sum(w for _, w in weighted_cams)
            ensemble_cam = sum(c * w for c, w in weighted_cams) / total_w
            ensemble_cam = np.clip(ensemble_cam, 0, 1)
        else:
            ensemble_cam = np.zeros((224, 224), dtype=np.float32)

        ensemble_overlay = show_cam_on_image(display_float, ensemble_cam, use_rgb=True)
        ensemble_b64 = _numpy_to_b64png(ensemble_overlay)

        return {
            "per_model": per_model,
            "ensemble_heatmap_b64": ensemble_b64,
            "predicted_class": pred_idx,
            "predicted_name": pred.prediction,
            "confidence": pred.confidence,
        }


def _numpy_to_b64png(img_array: np.ndarray) -> str:
    """Convert (H,W,3) uint8 numpy array to base64 PNG string."""
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")
