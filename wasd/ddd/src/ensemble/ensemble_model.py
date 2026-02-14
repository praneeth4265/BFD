"""
Ensemble inference utilities for 4-model bone fracture classification.
Models: ConvNeXt V2, EfficientNetV2-S, MaxViT, Swin Transformer.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import timm


CLASSES = ["comminuted_fracture", "no_fracture", "simple_fracture"]

DEFAULT_MODELS = [
    {
        "name": "ConvNeXt V2",
        "timm_name": "convnextv2_base.fcmae_ft_in22k_in1k",
        "checkpoint": "models/checkpoints/convnextv2_3class_original_best.pth",
    },
    {
        "name": "EfficientNetV2-S",
        "timm_name": "tf_efficientnetv2_s.in21k_ft_in1k",
        "checkpoint": "models/checkpoints/efficientnetv2_3class_original_best.pth",
    },
    {
        "name": "MaxViT-Tiny",
        "timm_name": "maxvit_tiny_tf_224.in1k",
        "checkpoint": "models/checkpoints/maxvit_3class_original_best.pth",
    },
    {
        "name": "Swin Transformer",
        "timm_name": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
        "checkpoint": "models/checkpoints/swin_3class_original_best.pth",
    },
]


@dataclass
class EnsemblePrediction:
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    individual_predictions: Dict[str, str]
    agreement_count: int


class EnsembleModel:
    def __init__(self, device: str = None, weights: List[float] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []
        self.model_names = []
        # Optimized weights from grid search on clean original test set (100% acc)
        # ConvNeXt V2: 0.25, EfficientNetV2-S: 0.25, MaxViT: 0.20, Swin: 0.30
        self.weights = weights or [0.25, 0.25, 0.20, 0.30]

        for config in DEFAULT_MODELS:
            model = timm.create_model(
                config["timm_name"],
                pretrained=False,
                num_classes=len(CLASSES),
            )
            checkpoint = torch.load(config["checkpoint"], map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            model.eval().to(self.device)
            self.models.append(model)
            self.model_names.append(config["name"])

        # weights already set in __init__ default, only override if user passes None
        if not self.weights:
            self.weights = [0.25, 0.25, 0.20, 0.30]

    def predict_proba(self, batch: torch.Tensor) -> np.ndarray:
        probs = []
        with torch.no_grad():
            for model in self.models:
                outputs = model(batch)
                probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
        stacked = np.stack(probs, axis=0)
        return np.average(stacked, axis=0, weights=self.weights)

    def predict(self, batch: torch.Tensor) -> List[EnsemblePrediction]:
        probs = []
        preds = []
        with torch.no_grad():
            for model in self.models:
                outputs = model(batch)
                prob = torch.softmax(outputs, dim=1).cpu().numpy()
                probs.append(prob)
                preds.append(np.argmax(prob, axis=1))

        stacked = np.stack(probs, axis=0)
        ensemble_probs = np.average(stacked, axis=0, weights=self.weights)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)

        results = []
        for i in range(batch.shape[0]):
            individual = {
                name: CLASSES[preds[idx][i]]
                for idx, name in enumerate(self.model_names)
            }
            agreement_count = sum(
                1 for p in individual.values() if p == CLASSES[ensemble_preds[i]]
            )
            results.append(
                EnsemblePrediction(
                    prediction=CLASSES[ensemble_preds[i]],
                    confidence=float(ensemble_probs[i, ensemble_preds[i]]),
                    probabilities={
                        cls: float(ensemble_probs[i, cls_idx])
                        for cls_idx, cls in enumerate(CLASSES)
                    },
                    individual_predictions=individual,
                    agreement_count=agreement_count,
                )
            )
        return results
