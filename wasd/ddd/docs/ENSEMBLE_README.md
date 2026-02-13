# 4-Model Ensemble Evaluation

This directory contains scripts to evaluate the 4-model ensemble and generate
metrics, confusion matrices, and ROC curves.

## ✅ Models Included
- ConvNeXt V2 Base
- EfficientNetV2-S
- MaxViT-Tiny
- Swin Transformer (Tiny)

## 📦 Outputs
Running the evaluation script produces:
- `results/ensemble_eval/ensemble_results.json`
- Confusion matrix PNGs for each model + ensemble
- ROC curve PNGs for each model + ensemble

## ▶️ Run Evaluation

```bash
ml_env_linux/bin/python3 ensemble_evaluate.py
```

### Optional Arguments
```bash
ml_env_linux/bin/python3 ensemble_evaluate.py \
  --batch-size 64 \
  --num-workers 4 \
  --results-dir results/ensemble_eval \
  --max-samples 200
```

## 🧪 Smoke Test (Quick Verification)
Use a small subset to validate everything runs:

```bash
ml_env_linux/bin/python3 ensemble_evaluate.py --max-samples 30
```

## 📝 Notes
- The script uses ImageNet normalization and 224×224 resizing.
- Ensure the checkpoints exist in `bone_fracture_detection/models/`.
- GPU is used automatically if available.
