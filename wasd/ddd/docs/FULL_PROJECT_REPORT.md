# Full Project Report — Bone Fracture Classification (3-Class)
**Date:** February 7, 2026  
**Repository:** BFD (praneeth4265/BFD)  
**Workspace:** `/home/praneeth4265/wasd/ddd`

---

## 1) Executive Summary
This project built a **3‑class bone fracture classifier** for X‑ray images using a **diverse 4‑model ensemble**. The pipeline covers dataset restructuring, GPU‑based augmentation, model training (ConvNeXt V2, EfficientNetV2‑S, MaxViT, Swin Transformer), evaluation on a 3,082‑image test set, and ensemble evaluation with ROC/Confusion matrices.

**Final Outcome:**
- **Ensemble accuracy:** **0.999351** (99.935%)
- **Best single model:** **Swin Transformer** (Accuracy 0.999027)
- **All results, plots, and logs stored in** `project_reports/`

---

## 2) Dataset Details
### 2.1 Original Dataset
- **Total images:** 3,584
- **Classes:** comminuted_fracture, no_fracture, simple_fracture
- **Split:** 70% train / 15% val / 15% test

**Original Distribution:**
- **Train:** 1,668 images
- **Val:** 356 images
- **Test:** 360 images

### 2.2 Augmented Dataset (Final)
- **Total images:** 20,530
- **Train:** 14,370
- **Val:** 3,078
- **Test:** 3,082

**Augmented Class Distribution:**
| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| comminuted_fracture | 5,163 | 1,106 | 1,107 | 7,376 |
| no_fracture | 4,790 | 1,026 | 1,027 | 6,843 |
| simple_fracture | 4,417 | 946 | 948 | 6,311 |

### 2.3 Folder Structure
```
datasets/
├── original/
│   ├── train/
│   ├── val/
│   └── test/
└── augmented/
    ├── train/
    ├── val/
    └── test/
```

---

## 3) Data Augmentation (GPU)
**Goal:** Balance no_fracture and expand dataset for robust training.

**Transforms Used:**
- Rotation (±25°)
- Horizontal flip
- Brightness/contrast
- Affine transform
- Color jitter
- Blur (light)

**Outcome:**
- Generated **6,843** augmented no_fracture images in ~19.5 minutes.

---

## 4) Training Environment
- **GPU:** NVIDIA RTX 4060 Laptop (8GB)
- **OS:** Linux
- **Python:** 3.10
- **PyTorch:** 2.8.0+cu128
- **timm:** 1.0.20

**Core Libraries:**
- torch, torchvision, timm, scikit‑learn, numpy, matplotlib

---

## 5) Models Trained (4‑Model Ensemble)

### 5.1 ConvNeXt V2 Base
- **Model:** `convnextv2_base.fcmae_ft_in22k_in1k`
- **Params:** 87.7M
- **Batch size:** 16
- **Best Val Acc:** 99.87% (Epoch 5)
- **Test Acc:** 99.578%
- **Checkpoint:** `convnextv2_3class_augmented_best.pth`

### 5.2 EfficientNetV2‑S
- **Model:** `tf_efficientnetv2_s.in21k_ft_in1k`
- **Params:** 20.2M
- **Batch size:** 32
- **Best Val Acc:** 99.74% (Epoch 7)
- **Test Acc:** 99.578%
- **Checkpoint:** `efficientnetv2_3class_augmented_best.pth`

### 5.3 MaxViT‑Tiny
- **Model:** `maxvit_tiny_tf_224.in1k`
- **Params:** 30.4M
- **Batch size:** 24
- **Best Val Acc:** 99.94% (Epoch 6)
- **Test Acc:** 99.805%
- **Checkpoint:** `maxvit_3class_augmented_best.pth`

### 5.4 Swin Transformer (Tiny)
- **Model:** `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k`
- **Params:** 27.5M
- **Batch size:** 32
- **Best Val Acc:** 99.97% (Epoch 10)
- **Test Acc:** 99.903%
- **Checkpoint:** `swin_3class_augmented_best.pth`

---

## 6) Evaluation Results (Individual Models)
**Test set:** 3,082 images

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | AUC‑ROC (Macro) |
|---|---:|---:|---:|---:|---:|
| ConvNeXt V2 | 0.995782 | 0.995789 | 0.995813 | 0.995800 | 0.999892 |
| EfficientNetV2‑S | 0.995782 | 0.995920 | 0.995739 | 0.995827 | 0.999953 |
| MaxViT | 0.998053 | 0.998147 | 0.997995 | 0.998069 | 0.999990 |
| Swin Transformer | 0.999027 | 0.999046 | 0.998996 | 0.999021 | 0.999997 |

---

## 7) Ensemble Evaluation (Soft Voting)
**Ensemble:** Average probability across all 4 models

| Metric | Value |
|---|---:|
| Accuracy | **0.999351** |
| Precision (Macro) | 0.999399 |
| Recall (Macro) | 0.999351 |
| F1 (Macro) | 0.999374 |
| AUC‑ROC (Macro) | **0.9999998** |

**Artifacts:**
- ROC curves (full + zoom): `project_reports/ensemble_eval/roc_*`
- Confusion matrices: `project_reports/ensemble_eval/confusion_matrix_*`
- Summary PDF: `project_reports/ensemble_eval/ensemble_summary.pdf`

---

## 8) Training Process Summary
**Common settings:**
- Optimizer: Adam
- LR: 1e‑4
- Scheduler: ReduceLROnPlateau
- Early stopping: patience=5
- Image size: 224×224
- Normalization: ImageNet mean/std

**Monitoring:**
- Training logs saved for each model
- Progress tracked with epoch summaries

---

## 9) ROC & Confusion Matrix Artifacts
All saved in:
```
project_reports/ensemble_eval/
```
Includes:
- ROC curves for each model (full + zoom)
- Confusion matrices for each model + ensemble
- JSON metrics
- PDF summary

---

## 10) Final Artifacts (Where Everything Lives)
```
project_reports/
├── FULL_PROJECT_REPORT.md  ✅ (this report)
├── ensemble_eval/          ✅ (all metrics + plots + PDF)
├── *_training.log          ✅ (all logs)
├── FINAL_MODEL_COMPARISON.md
├── PROJECT_COMPLETE_REPORT.txt
├── ENSEMBLE_TRAINING_STATUS.md
├── DATASET_RESTRUCTURE_SUMMARY.md
└── ...
```

---

## 11) Conclusion
This project delivered a **state‑of‑the‑art 3‑class fracture classifier** with near‑perfect accuracy. The **4‑model ensemble** provides both **accuracy** and **robustness**, and all artifacts (code, results, plots, logs, and reports) are fully stored and reproducible.

If you want a single combined PDF or a submission‑ready package, I can generate that too.
