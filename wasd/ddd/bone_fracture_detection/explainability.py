"""
Explainability for Bone Fracture Detection Models
Uses Grad-CAM to visualize which parts of X-ray images the model focuses on
"""

import os
import sys
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import timm

print("="*80)
print("🔍 MODEL EXPLAINABILITY - Grad-CAM Visualization")
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
    """Grad-CAM implementation for visualizing model attention"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, image_tensor, class_idx=None):
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        
        # Forward pass
        output = self.model(image_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=0)  # [H, W]
        
        # ReLU to only keep positive influences
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy(), class_idx, output

def load_model(model_type='convnext', dropout=0.3):
    """Load trained model"""
    print(f"📦 Loading {model_type.upper()} model...")
    
    if model_type == 'convnext':
        model_name = 'convnextv2_base.fcmae_ft_in22k_in1k'
        model_path = MODELS_DIR / 'convnext_v2_improved_best.pth'
    else:  # efficientnet
        model_name = 'tf_efficientnetv2_s.in21k_ft_in1k'
        model_path = MODELS_DIR / 'efficientnetv2_s_improved_best.pth'
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    # Create model
    model = timm.create_model(model_name, pretrained=False, num_classes=2,
                             drop_rate=dropout, drop_path_rate=0.2)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✅ Model loaded: {model_name}")
    print(f"   Test accuracy: {checkpoint.get('val_acc', 'N/A')}")
    return model

def get_target_layer(model, model_type='convnext'):
    """Get the target layer for Grad-CAM"""
    if model_type == 'convnext':
        # ConvNeXt: Use last stage
        return model.stages[-1]
    else:  # efficientnet
        # EfficientNet: Use last conv layer
        return model.conv_head

def visualize_gradcam(image_path, model, model_type='convnext', save_path=None):
    """
    Generate and visualize Grad-CAM heatmap
    
    Args:
        image_path: Path to input X-ray image
        model: Trained model
        model_type: 'convnext' or 'efficientnet'
        save_path: Optional path to save visualization
    """
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    image_tensor = transform(original_image).unsqueeze(0).to(DEVICE)
    
    # Get target layer
    target_layer = get_target_layer(model, model_type)
    
    # Create Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Generate heatmap
    cam, predicted_class, output = gradcam.generate_cam(image_tensor)
    
    # Get prediction probabilities
    probabilities = F.softmax(output, dim=1)[0]
    confidence = probabilities[predicted_class].item() * 100
    
    # Resize heatmap to match original image
    cam_resized = np.array(Image.fromarray(cam).resize(original_image.size, Image.BILINEAR))
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # 1. Original X-ray
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('Original X-ray Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Grad-CAM heatmap
    im = axes[0, 1].imshow(cam_resized, cmap='jet', alpha=1.0)
    axes[0, 1].set_title('Grad-CAM Heatmap\n(Model Attention)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 3. Overlay heatmap on original
    axes[1, 0].imshow(original_image, cmap='gray')
    axes[1, 0].imshow(cam_resized, cmap='jet', alpha=0.5)
    axes[1, 0].set_title('Overlay: Model Focus Areas', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 4. Prediction details
    axes[1, 1].axis('off')
    
    # Create prediction text
    prediction_text = f"""
MODEL PREDICTION
{'='*40}

Predicted Class: {CLASS_NAMES[predicted_class]}
Confidence: {confidence:.2f}%

Probability Distribution:
  • {CLASS_NAMES[0]}: {probabilities[0].item()*100:.2f}%
  • {CLASS_NAMES[1]}: {probabilities[1].item()*100:.2f}%

Model: {model_type.upper()}

{'='*40}

INTERPRETATION GUIDE:
• Red/Yellow areas: High attention
  (Model focuses here most)
• Blue/Purple areas: Low attention
  (Less important for decision)

The heatmap shows which parts of
the X-ray influenced the model's
decision most strongly.
    """
    
    axes[1, 1].text(0.05, 0.95, prediction_text, 
                    transform=axes[1, 1].transAxes,
                    fontsize=11,
                    verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Overall title
    image_name = Path(image_path).name
    fig.suptitle(f'Explainability Analysis: {image_name}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved: {save_path}")
    else:
        save_path = OUTPUT_DIR / f'gradcam_{model_type}_{Path(image_path).stem}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved: {save_path}")
    
    plt.close()
    
    return {
        'predicted_class': CLASS_NAMES[predicted_class],
        'confidence': confidence,
        'probabilities': probabilities.detach().cpu().numpy(),
        'heatmap': cam_resized,
        'save_path': str(save_path)
    }

def compare_models_explainability(image_path):
    """
    Compare Grad-CAM visualizations from both models
    """
    print(f"🔍 Comparing explainability for: {Path(image_path).name}")
    print("="*80)
    
    # Load both models
    convnext_model = load_model('convnext')
    efficientnet_model = load_model('efficientnet')
    
    if not convnext_model or not efficientnet_model:
        print("❌ Failed to load models")
        return
    
    print()
    print("🔵 Generating ConvNeXt V2 Grad-CAM...")
    convnext_result = visualize_gradcam(image_path, convnext_model, 'convnext')
    
    print()
    print("🟢 Generating EfficientNetV2-S Grad-CAM...")
    efficientnet_result = visualize_gradcam(image_path, efficientnet_model, 'efficientnet')
    
    # Create side-by-side comparison
    print()
    print("📊 Creating comparison visualization...")
    
    original_image = Image.open(image_path).convert('RGB')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: ConvNeXt V2
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('Original X-ray', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(convnext_result['heatmap'], cmap='jet', alpha=1.0)
    axes[0, 1].set_title('ConvNeXt V2 Heatmap', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(original_image, cmap='gray')
    axes[0, 2].imshow(convnext_result['heatmap'], cmap='jet', alpha=0.5)
    axes[0, 2].set_title(f"ConvNeXt V2 Overlay\n{convnext_result['predicted_class']} ({convnext_result['confidence']:.1f}%)", 
                        fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: EfficientNetV2-S
    axes[1, 0].imshow(original_image, cmap='gray')
    axes[1, 0].set_title('Original X-ray', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(efficientnet_result['heatmap'], cmap='jet', alpha=1.0)
    axes[1, 1].set_title('EfficientNetV2-S Heatmap', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(original_image, cmap='gray')
    axes[1, 2].imshow(efficientnet_result['heatmap'], cmap='jet', alpha=0.5)
    axes[1, 2].set_title(f"EfficientNetV2-S Overlay\n{efficientnet_result['predicted_class']} ({efficientnet_result['confidence']:.1f}%)", 
                        fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    fig.suptitle(f'Model Comparison: {Path(image_path).name}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    save_path = OUTPUT_DIR / f'comparison_{Path(image_path).stem}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Comparison saved: {save_path}")
    plt.close()
    
    # Print summary
    print()
    print("="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    print(f"\nConvNeXt V2:")
    print(f"  Prediction: {convnext_result['predicted_class']}")
    print(f"  Confidence: {convnext_result['confidence']:.2f}%")
    print(f"\nEfficientNetV2-S:")
    print(f"  Prediction: {efficientnet_result['predicted_class']}")
    print(f"  Confidence: {efficientnet_result['confidence']:.2f}%")
    
    if convnext_result['predicted_class'] == efficientnet_result['predicted_class']:
        print(f"\n✅ Both models agree: {convnext_result['predicted_class']}")
    else:
        print(f"\n⚠️  Models disagree!")
    
    print(f"\n📁 All outputs saved to: {OUTPUT_DIR}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations for bone fracture detection')
    parser.add_argument('--image', type=str, help='Path to X-ray image')
    parser.add_argument('--model', type=str, choices=['convnext', 'efficientnet', 'both'], 
                       default='both', help='Which model to use')
    parser.add_argument('--compare', action='store_true', 
                       help='Compare both models side-by-side')
    
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
        
        import random
        image_path = random.choice(test_images)
        print(f"🎲 No image specified. Using random test image:")
        print(f"   {image_path}")
        print()
    else:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return
    
    # Generate visualizations
    if args.model == 'both' or args.compare:
        compare_models_explainability(image_path)
    else:
        model = load_model(args.model)
        if model:
            print()
            visualize_gradcam(image_path, model, args.model)
    
    print()
    print("🎉 Explainability analysis complete!")
    print(f"📁 Check outputs in: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
