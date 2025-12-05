"""
Bone Fracture Detection with Integrated Explainability
Predicts fracture type AND shows Grad-CAM visualization in one step
"""

import os
import sys
import random
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import timm

print("="*80)
print("🔬 BONE FRACTURE DETECTION + EXPLAINABILITY")
print("="*80)
print()

# Configuration
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Comminuted Fracture', 'Simple Fracture']

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'explainability_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class GradCAM:
    """Grad-CAM for visualization"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, image_tensor, class_idx=None):
        self.model.eval()
        output = self.model(image_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy(), class_idx, output

def load_model(model_type='convnext', dropout=0.3):
    """Load trained model"""
    if model_type == 'convnext':
        model_name = 'convnextv2_base.fcmae_ft_in22k_in1k'
        model_path = MODELS_DIR / 'convnext_v2_improved_best.pth'
        print(f"📦 Loading ConvNeXt V2...")
    else:
        model_name = 'tf_efficientnetv2_s.in21k_ft_in1k'
        model_path = MODELS_DIR / 'efficientnetv2_s_improved_best.pth'
        print(f"📦 Loading EfficientNetV2-S...")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    model = timm.create_model(model_name, pretrained=False, num_classes=2,
                             drop_rate=dropout, drop_path_rate=0.2)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✅ Model loaded (Test Acc: {checkpoint.get('val_acc', 0):.2f}%)")
    return model

def get_target_layer(model, model_type='convnext'):
    """Get target layer for Grad-CAM"""
    if model_type == 'convnext':
        return model.stages[-1]
    else:
        return model.conv_head

def test_with_explainability(image_path, model_type='convnext', save_output=True):
    """
    Test image and generate explainability visualization
    """
    print(f"📷 Testing: {Path(image_path).name}")
    print("="*80)
    
    # Load model
    model = load_model(model_type)
    if not model:
        return None
    
    # Load image
    original_image = Image.open(image_path).convert('RGB')
    image_tensor = transform(original_image).unsqueeze(0).to(DEVICE)
    
    print()
    print("🔍 Generating prediction and explainability...")
    
    # Get target layer and create Grad-CAM
    target_layer = get_target_layer(model, model_type)
    gradcam = GradCAM(model, target_layer)
    
    # Generate CAM and prediction
    cam, predicted_class, output = gradcam.generate_cam(image_tensor)
    
    # Get probabilities
    probabilities = F.softmax(output, dim=1)[0]
    confidence = probabilities[predicted_class].item() * 100
    
    # Resize heatmap
    cam_resized = np.array(Image.fromarray(cam).resize(original_image.size, Image.BILINEAR))
    
    print()
    print("="*80)
    print("🎯 PREDICTION RESULTS")
    print("="*80)
    print(f"\n{'Predicted Class:':<20} {CLASS_NAMES[predicted_class]}")
    print(f"{'Confidence:':<20} {confidence:.2f}%")
    print(f"\n{'Probability Distribution:':}")
    for i, class_name in enumerate(CLASS_NAMES):
        prob = probabilities[i].item() * 100
        bar = '█' * int(prob / 2)
        print(f"  {class_name:<25} {prob:>6.2f}% {bar}")
    
    # Interpretation
    print(f"\n{'Interpretation:':}")
    if confidence >= 95:
        print("  ✅ Very high confidence - Strong prediction")
    elif confidence >= 85:
        print("  ✓ High confidence - Reliable prediction")
    elif confidence >= 70:
        print("  ⚠️  Moderate confidence - Consider review")
    else:
        print("  ⚠️  Low confidence - Manual review recommended")
    
    print()
    
    # Create visualization
    if save_output:
        print("🎨 Creating explainability visualization...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Row 1: Original and Heatmap
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.imshow(original_image, cmap='gray')
        ax1.set_title('Original X-ray Image', fontsize=14, fontweight='bold', pad=10)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 2])
        im = ax2.imshow(cam_resized, cmap='jet', alpha=1.0)
        ax2.set_title('Grad-CAM\nHeatmap', fontsize=12, fontweight='bold', pad=10)
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        # Row 2: Large Overlay
        ax3 = fig.add_subplot(gs[1, :])
        ax3.imshow(original_image, cmap='gray')
        ax3.imshow(cam_resized, cmap='jet', alpha=0.5)
        ax3.set_title('Explainability Overlay: Model Focus Areas', 
                     fontsize=14, fontweight='bold', pad=10)
        ax3.axis('off')
        
        # Add attention annotation
        ax3.text(0.02, 0.98, '🔴 Red = High Attention\n🔵 Blue = Low Attention',
                transform=ax3.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Row 3: Prediction Details
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Create detailed text
        details_text = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          PREDICTION ANALYSIS                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 PREDICTION:  {CLASS_NAMES[predicted_class]}
🎲 CONFIDENCE:  {confidence:.2f}%

📊 PROBABILITY DISTRIBUTION:
   • {CLASS_NAMES[0]:<30} {probabilities[0].item()*100:>6.2f}%  {'█' * int(probabilities[0].item()*50)}
   • {CLASS_NAMES[1]:<30} {probabilities[1].item()*100:>6.2f}%  {'█' * int(probabilities[1].item()*50)}

🤖 MODEL:  {model_type.upper()}
   • Architecture: {'ConvNeXt V2 Base' if model_type == 'convnext' else 'EfficientNetV2-S'}
   • Parameters: {'87.7M' if model_type == 'convnext' else '20.2M'}
   • Test Accuracy: {'98.88%' if model_type == 'convnext' else '96.65%'}

🔍 INTERPRETATION:
   {'✅ Very high confidence - Strong prediction' if confidence >= 95 else 
    '✓ High confidence - Reliable prediction' if confidence >= 85 else
    '⚠️  Moderate confidence - Consider review' if confidence >= 70 else
    '⚠️  Low confidence - Manual review recommended'}

💡 EXPLAINABILITY GUIDE:
   • Red/Yellow areas: High attention (model focuses here most)
   • Blue/Purple areas: Low attention (less important for decision)
   • The heatmap shows which parts of the X-ray influenced the prediction

📝 CLINICAL USE:
   • Verify heatmap focuses on fracture-relevant areas
   • Cross-reference with radiological findings
   • Use as decision support, not sole diagnostic tool
        """
        
        ax4.text(0.05, 0.95, details_text,
                transform=ax4.transAxes,
                fontsize=10,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Overall title
        fig.suptitle(f'Bone Fracture Detection with Explainability: {Path(image_path).name}',
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save
        save_path = OUTPUT_DIR / f'test_explain_{model_type}_{Path(image_path).stem}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved: {save_path}")
        plt.close()
    
    print()
    print("="*80)
    print("🎉 Analysis Complete!")
    print("="*80)
    
    return {
        'predicted_class': CLASS_NAMES[predicted_class],
        'confidence': confidence,
        'probabilities': probabilities.detach().cpu().numpy(),
        'heatmap': cam_resized
    }

def compare_both_models(image_path):
    """Test with both models and compare"""
    print(f"🔬 COMPREHENSIVE ANALYSIS WITH BOTH MODELS")
    print(f"📷 Image: {Path(image_path).name}")
    print("="*80)
    print()
    
    # Test with ConvNeXt V2
    print("🔵 CONVNEXT V2 ANALYSIS")
    print("-"*80)
    convnext_result = test_with_explainability(image_path, 'convnext', save_output=False)
    
    print()
    print()
    
    # Test with EfficientNetV2-S
    print("🟢 EFFICIENTNETV2-S ANALYSIS")
    print("-"*80)
    efficientnet_result = test_with_explainability(image_path, 'efficientnet', save_output=False)
    
    if not convnext_result or not efficientnet_result:
        return
    
    # Create comparison visualization
    print()
    print("🎨 Creating comprehensive comparison...")
    
    original_image = Image.open(image_path).convert('RGB')
    
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    # Title
    fig.suptitle(f'Comprehensive Analysis: {Path(image_path).name}',
                fontsize=18, fontweight='bold', y=0.98)
    
    # Row 1: ConvNeXt V2
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_image, cmap='gray')
    ax1.set_title('Original X-ray', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(convnext_result['heatmap'], cmap='jet')
    ax2.set_title('ConvNeXt V2 Heatmap', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(original_image, cmap='gray')
    ax3.imshow(convnext_result['heatmap'], cmap='jet', alpha=0.5)
    ax3.set_title(f"ConvNeXt V2 Overlay\n{convnext_result['predicted_class']} ({convnext_result['confidence']:.1f}%)",
                 fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Row 2: EfficientNetV2-S
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(original_image, cmap='gray')
    ax4.set_title('Original X-ray', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(efficientnet_result['heatmap'], cmap='jet')
    ax5.set_title('EfficientNetV2-S Heatmap', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(original_image, cmap='gray')
    ax6.imshow(efficientnet_result['heatmap'], cmap='jet', alpha=0.5)
    ax6.set_title(f"EfficientNetV2-S Overlay\n{efficientnet_result['predicted_class']} ({efficientnet_result['confidence']:.1f}%)",
                 fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    # Row 3 & 4: Comparison Details
    ax7 = fig.add_subplot(gs[2:, :])
    ax7.axis('off')
    
    # Agreement status
    agree = convnext_result['predicted_class'] == efficientnet_result['predicted_class']
    agreement_emoji = "✅" if agree else "⚠️"
    agreement_text = "MODELS AGREE" if agree else "MODELS DISAGREE"
    
    comparison_text = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        COMPREHENSIVE COMPARISON                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

{agreement_emoji} {agreement_text}

┌─────────────────────────────────────────────────────────────────────────────┐
│ 🔵 CONVNEXT V2 RESULTS                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Prediction:  {convnext_result['predicted_class']:<30}                    
│ Confidence:  {convnext_result['confidence']:>6.2f}%                                           
│ Comminuted:  {convnext_result['probabilities'][0]*100:>6.2f}%  {'█' * int(convnext_result['probabilities'][0]*40)}
│ Simple:      {convnext_result['probabilities'][1]*100:>6.2f}%  {'█' * int(convnext_result['probabilities'][1]*40)}
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 🟢 EFFICIENTNETV2-S RESULTS                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Prediction:  {efficientnet_result['predicted_class']:<30}                    
│ Confidence:  {efficientnet_result['confidence']:>6.2f}%                                           
│ Comminuted:  {efficientnet_result['probabilities'][0]*100:>6.2f}%  {'█' * int(efficientnet_result['probabilities'][0]*40)}
│ Simple:      {efficientnet_result['probabilities'][1]*100:>6.2f}%  {'█' * int(efficientnet_result['probabilities'][1]*40)}
└─────────────────────────────────────────────────────────────────────────────┘

📊 ANALYSIS:
   • Confidence difference: {abs(convnext_result['confidence'] - efficientnet_result['confidence']):.2f}%
   • {'Both models show high confidence' if min(convnext_result['confidence'], efficientnet_result['confidence']) > 90 else 
      'At least one model shows lower confidence'}
   • {'Attention patterns appear similar' if agree else 'Different attention patterns - review recommended'}

💡 RECOMMENDATION:
   {
    '✅ High confidence from both models - Strong consensus' if agree and min(convnext_result['confidence'], efficientnet_result['confidence']) > 90 else
    '✓ Models agree but review confidence levels' if agree else
    '⚠️  Models disagree - Manual review strongly recommended'
   }

🔍 EXPLAINABILITY NOTES:
   • Red/Yellow: High model attention (important for decision)
   • Blue/Purple: Low model attention (less relevant)
   • Compare heatmaps to verify both models focus on similar anatomical features
   • Verify attention is on fracture-relevant areas, not artifacts
    """
    
    ax7.text(0.05, 0.95, comparison_text,
            transform=ax7.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Save
    save_path = OUTPUT_DIR / f'comprehensive_analysis_{Path(image_path).stem}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Comprehensive analysis saved: {save_path}")
    plt.close()
    
    # Print summary
    print()
    print("="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"\n{agreement_emoji} {agreement_text}")
    print(f"\nConvNeXt V2:       {convnext_result['predicted_class']:<25} ({convnext_result['confidence']:.2f}%)")
    print(f"EfficientNetV2-S:  {efficientnet_result['predicted_class']:<25} ({efficientnet_result['confidence']:.2f}%)")
    print()
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description='Test bone fracture detection with integrated explainability'
    )
    parser.add_argument('--image', type=str, help='Path to X-ray image')
    parser.add_argument('--model', type=str, choices=['convnext', 'efficientnet', 'both'],
                       default='both', help='Which model to use (default: both)')
    
    args = parser.parse_args()
    
    # Get image path
    if args.image is None:
        # Use random test image
        test_dir = BASE_DIR / 'data_original' / 'test'
        test_images = []
        for class_dir in ['comminuted_fracture', 'simple_fracture']:
            class_path = test_dir / class_dir
            if class_path.exists():
                test_images.extend(list(class_path.glob('*.png')) + list(class_path.glob('*.jpg')))
        
        if not test_images:
            print("❌ No test images found!")
            return
        
        image_path = random.choice(test_images)
        print(f"🎲 No image specified. Using random test image:")
        print(f"   {image_path}")
        print()
    else:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return
    
    # Run analysis
    if args.model == 'both':
        compare_both_models(image_path)
    else:
        test_with_explainability(image_path, args.model)
    
    print()
    print(f"📁 All outputs saved to: {OUTPUT_DIR}")
    print("🎉 Complete!")

if __name__ == '__main__':
    main()
