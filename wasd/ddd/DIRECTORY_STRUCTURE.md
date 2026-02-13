# 📁 BFD Project Directory Structure

> Bone Fracture Detection — 4-Model Ensemble (ConvNeXt V2, EfficientNetV2, MaxViT, Swin Transformer)

```
BFD/
│
├── README.md                          # Project overview & quick start
├── .gitignore                         # Git ignore rules
├── .gitattributes                     # Git LFS tracking (*.pth)
│
├── configs/                           # ⚙️  Configuration files
│   ├── config.yaml                    #   Model & training configuration
│   ├── preprocessing_config.yaml      #   Data preprocessing settings
│   └── requirements.txt               #   Python package dependencies
│
├── datasets/                          # 📊 Datasets (gitignored — download separately)
│   ├── augmented/                     #   20,530 augmented images (train/val/test)
│   │   ├── train/                     #   Training set
│   │   ├── val/                       #   Validation set
│   │   └── test/                      #   Test set (3,082 images)
│   ├── original/                      #   Raw dataset before augmentation
│   └── info/                          #   Dataset metadata
│
├── models/                            # 🧠 Trained model artifacts
│   ├── checkpoints/                   #   Model weights (.pth, tracked via Git LFS)
│   │   ├── convnextv2_3class_augmented_best.pth
│   │   ├── efficientnetv2_3class_augmented_best.pth
│   │   ├── maxvit_3class_augmented_best.pth
│   │   ├── swin_3class_augmented_best.pth
│   │   ├── convnext_v2_best.pth       #   Legacy v1 (2-class) checkpoint
│   │   ├── convnext_v2_improved_best.pth
│   │   └── efficientnetv2_s_improved_best.pth
│   └── results/                       #   Per-model training results (JSON)
│       ├── convnextv2_3class_augmented_results.json
│       ├── efficientnetv2_3class_augmented_results.json
│       ├── maxvit_3class_augmented_results.json
│       └── swin_3class_augmented_results.json
│
├── src/                               # 💻 Source code
│   ├── training/                      #   Model training scripts
│   │   ├── train_convnext_3class_augmented.py
│   │   ├── train_convnext_pytorch_3class_augmented.py
│   │   ├── train_efficientnetv2_3class_augmented.py
│   │   ├── train_efficientnetv2_pytorch_3class_augmented.py
│   │   ├── train_maxvit_pytorch_3class_augmented.py
│   │   └── train_swin_pytorch_3class_augmented.py
│   │
│   ├── evaluation/                    #   Model evaluation & metrics
│   │   ├── ensemble_evaluate.py       #   Full evaluation pipeline (all models + ensemble)
│   │   └── evaluate_convnext_3class.py
│   │
│   ├── ensemble/                      #   Ensemble model system
│   │   ├── ensemble_model.py          #   EnsembleModel class (soft voting, optimized weights)
│   │   ├── predict_ensemble.py        #   CLI batch prediction → CSV
│   │   └── optimize_ensemble_weights.py  # Weight optimization (grid search + scipy)
│   │
│   ├── visualization/                 #   Plot & report generation
│   │   ├── generate_ensemble_detailed_report.py  # 6-page ensemble PDF report
│   │   ├── generate_ensemble_report.py           # Summary ensemble report
│   │   ├── generate_model_reports.py             # Per-model detailed PDFs
│   │   ├── generate_roc_plots.py                 # ROC curve generation
│   │   ├── generate_pr_calibration_plots.py      # PR + calibration plots
│   │   ├── generate_training_curves.py           # Training history comparison
│   │   ├── generate_error_analysis.py            # Misclassification analysis
│   │   └── interpretability_analysis.py          # Blur-occlusion heatmaps
│   │
│   ├── data_processing/               #   Data augmentation & preprocessing
│   │   ├── generate_augmented_optimized.py       # GPU-accelerated augmentation
│   │   └── generate_augmented_nofracture_gpu.py  # No-fracture class generation
│   │
│   ├── app/                           #   Web application
│   │   ├── main.py                    #   FastAPI inference endpoint
│   │   ├── streamlit_app.py           #   Streamlit demo UI
│   │   ├── templates/
│   │   │   └── index.html
│   │   └── static/
│   │
│   └── legacy/                        #   🗄️  Archived v1 code (2-class system)
│       ├── interpretability_analysis_old.py
│       ├── v1_models/                 #   Old model definitions & utilities
│       ├── v1_training/               #   Old training scripts
│       └── v1_testing/                #   Old test & utility scripts
│
├── reports/                           # 📈 Generated reports & outputs
│   ├── ensemble_eval/
│   │   ├── plots/                     #   All PNG charts (ROC, CM, PR, calibration, etc.)
│   │   ├── pdfs/                      #   PDF reports (per-model + ensemble detailed)
│   │   ├── data/                      #   Raw data (JSON, CSV, NPZ, MD summaries)
│   │   ├── error_analysis/            #   Misclassification analysis outputs
│   │   └── interpretability/          #   Grad-CAM heatmaps & samples
│   ├── training_logs/                 #   Training log files & TensorBoard runs
│   └── project_reports_bundle.zip     #   Archived bundle
│
├── scripts/                           # 🔧 Shell scripts & utilities
│   ├── git_setup_instructions.sh
│   ├── monitor_maxvit_training.sh
│   ├── monitor_pytorch_training.sh
│   ├── monitor_swin_training.sh
│   ├── monitor.sh
│   ├── quick_status.sh
│   ├── start_efficientnetv2_training.sh
│   ├── watch_maxvit_live.sh
│   ├── watch_swin_live.sh
│   └── watch_training.sh
│
├── docs/                              # 📖 Documentation
│   ├── ADVANCED_ENSEMBLE_TODO.md      #   Phase-by-phase development plan
│   ├── FULL_PROJECT_REPORT.md         #   Comprehensive project report
│   ├── FINAL_MODEL_COMPARISON.md      #   Model accuracy comparison
│   ├── ENSEMBLE_README.md             #   Ensemble system documentation
│   ├── ENSEMBLE_INFERENCE_README.md   #   Inference pipeline guide
│   ├── ENSEMBLE_TRAINING_STATUS.md    #   Training progress tracker
│   ├── PYTORCH_TRAINING_STATUS.md     #   PyTorch migration status
│   ├── TRAINING_STATUS.md             #   Overall training status
│   ├── PROJECT_SUMMARY.md             #   High-level project summary
│   ├── PROJECT_LOG.md                 #   Development changelog
│   ├── EXPLAINABILITY_GUIDE.md        #   Grad-CAM usage guide
│   ├── EXPLAINABILITY_SUMMARY.md      #   Explainability results
│   ├── DATASET_CLEANUP_STATUS.md      #   Dataset cleaning notes
│   ├── DATASET_RESTRUCTURE_SUMMARY.md #   Dataset restructuring log
│   ├── FILES_SUMMARY.md               #   Legacy file inventory
│   ├── FINAL_RESULTS.md               #   Final evaluation results
│   ├── INTEGRATED_TESTING_GUIDE.md    #   Testing infrastructure guide
│   ├── LFS_UPLOAD_INSTRUCTIONS.md     #   Git LFS setup guide
│   ├── PROJECT_COMPLETE_REPORT.txt    #   Completion summary
│   └── README.md                      #   Legacy v1 README
│
├── tests/                             # 🧪 Test scripts (future)
│
└── data/                              # Dataset utilities (future)
```

## Quick Navigation

| What you need | Where to find it |
|---|---|
| Train a model | `src/training/` |
| Run ensemble prediction | `src/ensemble/predict_ensemble.py` |
| Ensemble model class | `src/ensemble/ensemble_model.py` |
| Generate PDF reports | `src/visualization/generate_ensemble_detailed_report.py` |
| Model weights (.pth) | `models/checkpoints/` |
| Evaluation metrics (JSON) | `reports/ensemble_eval/data/` |
| ROC / PR / CM plots | `reports/ensemble_eval/plots/` |
| PDF reports | `reports/ensemble_eval/pdfs/` |
| Web demo | `src/app/streamlit_app.py` |
| Project documentation | `docs/` |
| Shell utilities | `scripts/` |
| Old v1 code (archived) | `src/legacy/` |
