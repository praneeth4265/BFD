# Ensemble Evaluation Summary

**Samples:** 3082
**Classes:** comminuted_fracture, no_fracture, simple_fracture

## Leaderboard (sorted by accuracy)

| Rank | Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | AUC-ROC (Macro) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Ensemble (Soft Voting) | 0.99935 | 0.99940 | 0.99935 | 0.99937 | 1.00000 |
| 2 | Swin Transformer (Tiny) | 0.99903 | 0.99905 | 0.99900 | 0.99902 | 1.00000 |
| 3 | MaxViT-Tiny | 0.99805 | 0.99815 | 0.99799 | 0.99807 | 0.99999 |
| 4 | ConvNeXt V2 Base | 0.99578 | 0.99579 | 0.99581 | 0.99580 | 0.99989 |
| 5 | EfficientNetV2-S | 0.99578 | 0.99592 | 0.99574 | 0.99583 | 0.99995 |

## Best Model Summary
**Top performer:** Ensemble (Soft Voting) with accuracy 0.99935 and macro F1 0.99937.

## Per-class metrics (Ensemble)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| comminuted_fracture | 0.99820 | 1.00000 | 0.99910 | 1107 |
| no_fracture | 1.00000 | 0.99805 | 0.99903 | 1027 |
| simple_fracture | 1.00000 | 1.00000 | 1.00000 | 948 |

## Artifacts

- Confusion matrices: `results/ensemble_eval/confusion_matrix_*.png`
- ROC curves: `results/ensemble_eval/roc_*.png`
- Leaderboard CSV: `results/ensemble_eval/model_leaderboard.csv`
- Comparison plot: `results/ensemble_eval/model_comparison.png`
- PDF summary: `results/ensemble_eval/ensemble_summary.pdf`
