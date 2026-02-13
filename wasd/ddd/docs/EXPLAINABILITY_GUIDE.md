# 🔍 Model Explainability Guide

## Overview

This project uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize what the AI models "see" when making predictions. This is crucial for:

- ✅ **Trust & Transparency** - Understand model decisions
- ✅ **Clinical Validation** - Verify models focus on relevant anatomical features
- ✅ **Error Analysis** - Debug misclassifications
- ✅ **Regulatory Compliance** - Medical AI requires explainability

---

## 🎯 What is Grad-CAM?

**Grad-CAM** generates heatmaps showing which parts of an X-ray image influenced the model's decision:

- 🔴 **Red/Yellow areas** = High attention (important for decision)
- 🔵 **Blue/Purple areas** = Low attention (less relevant)

### How it Works:
1. Forward pass through the model
2. Backward pass to compute gradients
3. Weight activation maps by gradient importance
4. Generate normalized heatmap
5. Overlay on original image

---

## 🚀 Quick Start

### 1. Compare Both Models (Recommended)
```bash
python3 explainability.py
```
This will:
- Pick a random test image
- Generate Grad-CAM for ConvNeXt V2
- Generate Grad-CAM for EfficientNetV2-S
- Create side-by-side comparison
- Save 3 visualizations

### 2. Analyze Specific Image
```bash
python3 explainability.py --image /path/to/xray.jpg
```

### 3. Single Model Analysis
```bash
# ConvNeXt V2 only
python3 explainability.py --image /path/to/xray.jpg --model convnext

# EfficientNetV2-S only
python3 explainability.py --image /path/to/xray.jpg --model efficientnet
```

---

## 📊 Output Files

All visualizations are saved to `explainability_outputs/`:

### Individual Model Outputs:
- **`gradcam_convnext_[image_name].png`** - ConvNeXt V2 analysis
  - 4-panel visualization:
    1. Original X-ray
    2. Grad-CAM heatmap
    3. Overlay (heatmap on X-ray)
    4. Prediction details with interpretation guide

- **`gradcam_efficientnet_[image_name].png`** - EfficientNetV2-S analysis
  - Same 4-panel layout

### Comparison Output:
- **`comparison_[image_name].png`** - Side-by-side comparison
  - 2 rows × 3 columns:
    - Row 1: ConvNeXt V2 (original, heatmap, overlay)
    - Row 2: EfficientNetV2-S (original, heatmap, overlay)
  - Shows if models agree/disagree
  - Displays confidence scores

---

## 🔬 Interpreting Results

### What to Look For:

#### ✅ **Good Model Behavior:**
- Focus on fracture lines
- Attention on bone structures
- Consistent between models
- High confidence on relevant areas

#### ⚠️ **Potential Issues:**
- Focus on image artifacts
- Attention on irrelevant areas
- Models disagree significantly
- Low confidence overall

### Example Interpretations:

#### **Comminuted Fracture:**
- Model should focus on:
  - Multiple fracture fragments
  - Bone displacement areas
  - Areas with bone separation

#### **Simple Fracture:**
- Model should focus on:
  - Single fracture line
  - Bone continuity break
  - Linear crack pattern

---

## 📈 Usage Examples

### Example 1: Random Test Image
```bash
cd /home/praneeth4265/wasd/ddd/bone_fracture_detection
python3 explainability.py
```

**Output:**
```
🎲 No image specified. Using random test image:
   data_original/test/comminuted_fracture/comminuted_fracture_test_0075.jpg

🔍 Comparing explainability for: comminuted_fracture_test_0075.jpg
================================================================================
📦 Loading CONVNEXT model...
✅ Model loaded: convnextv2_base.fcmae_ft_in22k_in1k
   Test accuracy: 99.16%

📦 Loading EFFICIENTNET model...
✅ Model loaded: tf_efficientnetv2_s.in21k_ft_in1k
   Test accuracy: 98.32%

🔵 Generating ConvNeXt V2 Grad-CAM...
✅ Visualization saved: explainability_outputs/gradcam_convnext_*.png

🟢 Generating EfficientNetV2-S Grad-CAM...
✅ Visualization saved: explainability_outputs/gradcam_efficientnet_*.png

📊 Creating comparison visualization...
✅ Comparison saved: explainability_outputs/comparison_*.png

================================================================================
📊 COMPARISON SUMMARY
================================================================================
ConvNeXt V2:
  Prediction: Comminuted Fracture
  Confidence: 96.94%

EfficientNetV2-S:
  Prediction: Comminuted Fracture
  Confidence: 98.93%

✅ Both models agree: Comminuted Fracture
```

### Example 2: Specific Image
```bash
python3 explainability.py --image data_original/test/simple_fracture/simple_fracture_test_0054.jpg
```

### Example 3: ConvNeXt Only
```bash
python3 explainability.py --image my_xray.jpg --model convnext
```

---

## 🎨 Visualization Components

### 4-Panel Individual Model Visualization:

```
┌─────────────────┬─────────────────┐
│  Original X-ray │  Grad-CAM       │
│                 │  Heatmap        │
├─────────────────┼─────────────────┤
│  Overlay        │  Prediction     │
│  (Model Focus)  │  Details        │
└─────────────────┴─────────────────┘
```

### 6-Panel Comparison Visualization:

```
ConvNeXt V2:
┌─────────────┬─────────────┬─────────────┐
│  Original   │  Heatmap    │  Overlay    │
└─────────────┴─────────────┴─────────────┘

EfficientNetV2-S:
┌─────────────┬─────────────┬─────────────┐
│  Original   │  Heatmap    │  Overlay    │
└─────────────┴─────────────┴─────────────┘
```

---

## 🔧 Technical Details

### Grad-CAM Implementation:
- **Target Layer:**
  - ConvNeXt V2: Last stage (`model.stages[-1]`)
  - EfficientNetV2-S: Last conv layer (`model.conv_head`)

- **Process:**
  1. Register forward/backward hooks
  2. Compute gradients w.r.t. target class
  3. Global average pooling of gradients
  4. Weighted sum of activation maps
  5. ReLU + normalization

- **Colormap:** Jet (red = high, blue = low)
- **Resolution:** 224×224 (resized to original)

### Requirements:
- PyTorch 2.8.0+
- matplotlib
- numpy
- PIL
- timm

---

## 📋 Command Reference

### Basic Commands:
```bash
# Compare both models (random image)
python3 explainability.py

# Specific image, both models
python3 explainability.py --image path/to/xray.jpg

# Single model
python3 explainability.py --image path/to/xray.jpg --model convnext
python3 explainability.py --image path/to/xray.jpg --model efficientnet

# Help
python3 explainability.py --help
```

### Arguments:
- `--image PATH` - Path to X-ray image (optional, uses random if not specified)
- `--model {convnext,efficientnet,both}` - Which model to analyze (default: both)
- `--compare` - Force comparison mode

---

## 🎯 Use Cases

### 1. **Clinical Validation**
Verify model focuses on medically relevant features:
```bash
python3 explainability.py --image challenging_case.jpg
```
Check if heatmap highlights actual fracture areas.

### 2. **Error Analysis**
Investigate misclassifications:
```bash
# Find incorrect predictions
# Generate Grad-CAM to see what model focused on
python3 explainability.py --image misclassified_image.jpg
```

### 3. **Model Comparison**
Compare attention patterns between models:
```bash
python3 explainability.py --image sample.jpg --compare
```
See if models agree on important features.

### 4. **Research & Publications**
Generate publication-quality visualizations:
- High-resolution outputs (150 DPI)
- Professional layouts
- Clear labels and legends

---

## 📊 Expected Results

### Both Models Should:
✅ Focus on bone structures  
✅ Highlight fracture lines  
✅ Ignore background/artifacts  
✅ Show consistent attention patterns  
✅ Achieve >95% confidence on clear cases  

### Model Differences:
- **ConvNeXt V2**: May show broader attention (larger receptive field)
- **EfficientNetV2-S**: May show more focused attention (efficient architecture)

---

## ⚠️ Limitations

### Grad-CAM Limitations:
1. **Coarse Resolution** - Heatmaps are lower resolution than input
2. **Class-Specific** - Shows attention for predicted class only
3. **Post-hoc** - Visualization created after training
4. **Not Causal** - Correlation ≠ causation

### Best Practices:
- ✅ Use alongside clinical expertise
- ✅ Validate on multiple images
- ✅ Consider model confidence
- ✅ Cross-reference with medical knowledge
- ❌ Don't use as sole diagnostic tool

---

## 🔬 Advanced Usage

### Batch Processing:
Create explainability for all test images:
```bash
# Create a script
for img in data_original/test/*/*.jpg; do
    python3 explainability.py --image "$img"
done
```

### Integration with Testing:
Combine with testing scripts:
```bash
# First test
python3 test_random_images.py

# Then explain incorrect predictions
python3 explainability.py --image incorrect_prediction.jpg
```

---

## 📁 File Organization

```
explainability_outputs/
├── gradcam_convnext_image1.png
├── gradcam_efficientnet_image1.png
├── comparison_image1.png
├── gradcam_convnext_image2.png
├── gradcam_efficientnet_image2.png
└── comparison_image2.png
```

---

## 🎓 Learning Resources

### What Makes a Good Heatmap:
1. **Focused Attention** - Concentrated on fracture areas
2. **Consistent Patterns** - Similar across similar cases
3. **Clinically Meaningful** - Aligns with radiological features
4. **High Confidence** - Model is certain about decision

### Red Flags:
- 🚩 Attention on image borders
- 🚩 Focus on text/labels/artifacts
- 🚩 No clear focal point
- 🚩 Models strongly disagree

---

## 🤝 Contributing to Explainability

### Future Enhancements:
- [ ] Grad-CAM++ (improved localization)
- [ ] Integrated Gradients
- [ ] SHAP values
- [ ] Attention visualization (if using Transformers)
- [ ] Layer-wise relevance propagation (LRP)
- [ ] Occlusion sensitivity maps

---

## 📖 References

- **Grad-CAM Paper:** Selvaraju et al. (2017) - "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- **Medical AI Explainability:** Review papers on interpretable medical imaging AI
- **Regulatory Guidelines:** FDA guidance on AI/ML in medical devices

---

## ✅ Verification Checklist

Before trusting model predictions:
- [ ] Generated Grad-CAM visualization
- [ ] Verified attention on relevant anatomical areas
- [ ] Checked both models agree
- [ ] Confidence >95% for clear cases
- [ ] No attention on artifacts
- [ ] Clinical features match prediction
- [ ] Compared with similar cases

---

## 🎉 Summary

**Explainability is essential for medical AI!**

Use `explainability.py` to:
✅ Understand model decisions  
✅ Build trust in predictions  
✅ Validate clinical relevance  
✅ Debug errors  
✅ Meet regulatory requirements  

**Start with:**
```bash
python3 explainability.py
```

Then explore your own X-rays and see what the models see! 🔍

---

*For more information, see the main README.md or contact the development team.*
