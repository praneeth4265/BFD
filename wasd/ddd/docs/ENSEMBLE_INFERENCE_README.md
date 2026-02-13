# Ensemble Inference (4 Models)

This module runs inference using the 4‑model ensemble:
- ConvNeXt V2
- EfficientNetV2‑S
- MaxViT
- Swin Transformer

## Files
- `ensemble_model.py` — ensemble model loader + prediction logic
- `predict_ensemble.py` — CLI to run inference and export CSV

## Run Inference

```bash
ml_env_linux/bin/python3 predict_ensemble.py \
  --input-dir datasets/augmented/test \
  --output-csv project_reports/ensemble_eval/ensemble_predictions.csv
```

### Quick Smoke Test
```bash
ml_env_linux/bin/python3 predict_ensemble.py \
  --input-dir datasets/augmented/test \
  --output-csv project_reports/ensemble_eval/ensemble_predictions_sample.csv \
  --max-samples 10
```

## Output CSV Columns
- image_path
- prediction
- confidence
- agreement_count
- prob_comminuted_fracture
- prob_no_fracture
- prob_simple_fracture
