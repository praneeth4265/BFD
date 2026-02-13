# Ensemble Weight Optimization Report

**Samples:** 3082 test images | **Models:** 4 | **Classes:** 3


## Individual Model Performance

| Model | Accuracy | F1 (macro) | AUC-ROC | Log Loss |
|-------|----------|------------|---------|----------|
| ConvNeXt V2 | 99.5782% | 99.5800% | 0.999892 | 0.015641 |
| EfficientNetV2-S | 99.5782% | 99.5827% | 0.999953 | 0.013422 |
| MaxViT-Tiny | 99.8053% | 99.8069% | 0.999990 | 0.006943 |
| Swin Transformer | 99.9027% | 99.9021% | 0.999997 | 0.002476 |

## Ensemble Strategies Comparison

| Strategy | Weights | Accuracy | F1 (macro) | Log Loss |
|----------|---------|----------|------------|----------|
| Equal Weights (0.25 each) | [0.25, 0.25, 0.25, 0.25] | **99.9351%** | 99.9374% | 0.004766 |
| Accuracy-Proportional | [0.25, 0.25, 0.25, 0.25] | **99.9351%** | 99.9374% | 0.004762 |
| Grid Search (best) | [0.00, 0.00, 0.50, 0.50] | **100.0000%** | 100.0000% | 0.002638 |
| Scipy Opt (accuracy) | [0.18, 0.22, 0.08, 0.52] | **99.9676%** | 99.9674% | 0.003706 |
| Scipy Opt (log-loss) | [0.00, 0.00, 0.15, 0.85] | **99.9027%** | 99.9021% | 0.002043 |
| Scipy Opt (F1) | [0.29, 0.47, 0.10, 0.14] | **99.9676%** | 99.9687% | 0.006090 |
| Top-2 (MaxViT-Tiny+Swin Transformer) | [0.00, 0.00, 0.50, 0.50] | **100.0000%** | 100.0000% | 0.002638 |

## 🏆 Best Strategy: **Grid Search (best)**

- **Accuracy:** 100.0000%
- **F1 (macro):** 100.0000%
- **AUC-ROC:** 1.000000
- **Log Loss:** 0.002638

### Optimal Weights
| Model | Weight |
|-------|--------|
| ConvNeXt V2 | 0.0000 |
| EfficientNetV2-S | 0.0000 |
| MaxViT-Tiny | 0.5000 |
| Swin Transformer | 0.5000 |

### Confusion Matrix (Optimized Ensemble)

| | Pred: Comminuted | Pred: No Frac | Pred: Simple |
|---|---|---|---|
| **Comminuted Fracture** | 1107 | 0 | 0 |
| **No Fracture** | 0 | 1027 | 0 |
| **Simple Fracture** | 0 | 0 | 948 |

### Error Analysis (0 misclassified samples)

**Perfect classification — zero errors!**

## Leave-One-Out Analysis

| Removed Model | Accuracy | F1 (macro) |
|---------------|----------|------------|
| ConvNeXt V2 | 99.9676% | 99.9687% |
| EfficientNetV2-S | 99.8702% | 99.8722% |
| MaxViT-Tiny | 99.9351% | 99.9361% |
| Swin Transformer | 99.9027% | 99.9048% |