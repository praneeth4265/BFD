# Bone Fracture Detection - Usage Guide

## 🎯 Quick Start

### **🔬 Test with Explainability (BEST!) ⭐**
Get prediction AND visual explanation in one step:
```bash
python3 test_with_explainability.py
```
This tests with BOTH models and shows Grad-CAM heatmaps!

### **Compare Both Models**
Compare ConvNeXt V2 vs EfficientNetV2-S on 5 random images:
```bash
python3 compare_models.py
```

### **Quick Test**
Test the model on 3 random images immediately:
```bash
python3 quick_test.py
```

### **Test Single Image Without Arguments**
Will automatically pick a random test image:
```bash
python3 test_single_image.py
```

### **Test Your Own X-ray Image**
```bash
python3 test_single_image.py --image path/to/your/xray.jpg
```

### **Test Multiple Random Images**
Test on 10 random images:
```bash
python3 test_random_images.py
```

---

## 📋 Available Scripts

### 1. **test_with_explainability.py** � RECOMMENDED! ⭐
- **Purpose**: Complete analysis with prediction + explainability in one step
- **What it does**: 
  - Predicts fracture type with both models
  - Generates Grad-CAM heatmaps showing what models "see"
  - Creates comprehensive 6-panel visualization
  - Compares both models side-by-side
  - Shows agreement/disagreement status
  - Provides clinical recommendations
- **Usage**: `python3 test_with_explainability.py` (random) or `python3 test_with_explainability.py --image xray.jpg`
- **Best for**: Production use, clinical validation, comprehensive analysis
- **Output**: High-quality visualization with predictions, heatmaps, and detailed analysis
- **Guide**: See `INTEGRATED_TESTING_GUIDE.md` for complete documentation

### 2. **explainability.py** 🔍
- **Purpose**: Visualize what models "see" with Grad-CAM heatmaps (standalone)
- **What it does**: Generates attention visualizations showing which parts of X-ray influenced decision
- **Usage**: `python3 explainability.py` (random image) or `python3 explainability.py --image path/to/xray.jpg`
- **Best for**: Detailed explainability analysis, research, understanding model reasoning
- **Output**: Heatmap overlays saved to `explainability_outputs/`
- **Guide**: See `EXPLAINABILITY_GUIDE.md` for detailed documentation

### 3. **compare_models.py** ⚡
- **Purpose**: Compare ConvNeXt V2 vs EfficientNetV2-S
- **What it does**: Tests 5 random images on both models, shows speed & accuracy comparison
- **Usage**: `python3 compare_models.py`
- **Best for**: Understanding model trade-offs (accuracy vs speed)

### 3. **quick_test.py** ⚡
- **Purpose**: Fastest way to test the model
- **What it does**: Tests 3 random images with full output
- **Usage**: `python3 quick_test.py`
- **Best for**: Quick demonstration

### 4. **test_single_image.py** 🔬
- **Purpose**: Test any single X-ray image
- **What it does**: Predicts fracture type with confidence scores
- **Usage**: 
  - With image: `python3 test_single_image.py --image path/to/xray.jpg`
  - Without image (random): `python3 test_single_image.py`
- **Best for**: Testing specific images

### 5. **test_random_images.py** 🎲
- **Purpose**: Comprehensive testing
- **What it does**: Tests 10 random images and shows statistics
- **Usage**: `python3 test_random_images.py`
- **Best for**: Model evaluation

### 6. **train_convnext_improved.py** 🏋️
- **Purpose**: Train ConvNeXt V2 from scratch
- **What it does**: Trains ConvNeXt V2 with anti-overfitting techniques
- **Usage**: `python3 train_convnext_improved.py`
- **Best for**: Retraining or improving the model

### 7. **train_efficientnetv2_improved.py** 🏋️
- **Purpose**: Train EfficientNetV2-S from scratch
- **What it does**: Trains EfficientNetV2-S with anti-overfitting techniques
- **Usage**: `python3 train_efficientnetv2_improved.py`
- **Best for**: Training a faster, smaller model

---

## 🖼️ Sample Output

### Comminuted Fracture Detection
```
🔴 COMMINUTED FRACTURE
   Confidence: 98.47%

📈 Probability Distribution:
   Comminuted Fracture: 98.47%
   Simple Fracture: 1.53%

💡 Interpretation:
   Very high confidence - Strong prediction
```

### Simple Fracture Detection
```
🔵 SIMPLE FRACTURE
   Confidence: 96.25%

📈 Probability Distribution:
   Comminuted Fracture: 3.75%
   Simple Fracture: 96.25%

💡 Interpretation:
   Very high confidence - Strong prediction
```

---

## 📊 Model Performance

- **Test Accuracy**: 98.88%
- **Validation Accuracy**: 99.16%
- **Average Confidence**: 96.39%
- **Min Confidence**: 94.17%

---

## 🔧 Requirements

### System Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB minimum
- **Storage**: 2GB for model and dataset

### Software Requirements
```bash
# Python 3.10
# PyTorch 2.8.0 with CUDA 12.8
# timm 1.0.21
# PIL, numpy, scikit-learn
```

All dependencies are already installed in the virtual environment at:
`/home/praneeth4265/wasd/ddd/ml_env_linux/`

---

## 📁 Project Structure

```
bone_fracture_detection/
├── quick_test.py                      # ⚡ Fastest way to test
├── test_single_image.py               # 🔬 Test any single image
├── test_random_images.py              # 🎲 Test multiple random images
├── train_convnext_improved.py         # 🏋️ Train the model
│
├── models/
│   ├── convnext_v2_improved_best.pth  # Trained model (1GB)
│   └── convnext_v2_improved_results.json  # Training metrics
│
├── data_original/
│   ├── train/                         # 1,668 training images
│   ├── val/                           # 358 validation images
│   └── test/                          # 358 test images
│
└── PROJECT_SUMMARY.md                 # Complete documentation
```

---

## 🚀 Common Use Cases

### **Case 1: Quick Demo**
```bash
python3 quick_test.py
```
Shows 3 predictions in ~10 seconds.

### **Case 2: Test Your X-ray**
```bash
python3 test_single_image.py --image /path/to/your/xray.jpg
```

### **Case 3: Evaluate Model**
```bash
python3 test_random_images.py
```
Tests 10 random images and shows accuracy statistics.

### **Case 4: Batch Testing**
```bash
# Test all images in a directory
for img in /path/to/images/*.jpg; do
    python3 test_single_image.py --image "$img"
done
```

---

## 💡 Tips

1. **Image Format**: Supports `.jpg`, `.jpeg`, `.png`
2. **Image Size**: Any size (automatically resized to 224x224)
3. **Color**: Grayscale X-rays are converted to RGB automatically
4. **Confidence Levels**:
   - ≥95%: Very high confidence
   - 85-95%: High confidence
   - 75-85%: Moderate confidence
   - <75%: Manual review recommended

---

## ❓ Troubleshooting

### Script not running?
```bash
# Make sure you're in the correct directory
cd /home/praneeth4265/wasd/ddd/bone_fracture_detection

# Use the virtual environment Python
/home/praneeth4265/wasd/ddd/ml_env_linux/bin/python3 quick_test.py
```

### No GPU detected?
The scripts will automatically use CPU if no GPU is available. It will be slower but still work.

### File not found errors?
Make sure the model file exists:
```bash
ls -lh models/convnext_v2_improved_best.pth
```

---

## 🎓 Understanding the Output

### Prediction Components
1. **Classification**: Comminuted or Simple Fracture
2. **Confidence**: Model's certainty (0-100%)
3. **Probability Distribution**: Breakdown for each class
4. **Interpretation**: Guidance on prediction reliability

### Example Analysis
```
🔴 COMMINUTED FRACTURE
   Confidence: 98.47%
```
This means the model is 98.47% certain the X-ray shows a comminuted fracture.

---

## 📞 Support

For issues or questions:
1. Check `PROJECT_SUMMARY.md` for detailed information
2. Review training metrics in `models/convnext_v2_improved_results.json`
3. Test with known images from `data_original/test/`

---

## ✅ Verification

Test the installation:
```bash
# Quick test (3 images)
python3 quick_test.py

# Should show:
# - Model loaded successfully
# - 3 predictions with high confidence (>94%)
# - All predictions should be correct
```

---

**Last Updated**: November 3, 2025
**Model Version**: ConvNeXt V2 (Improved)
**Status**: Production Ready ✅
