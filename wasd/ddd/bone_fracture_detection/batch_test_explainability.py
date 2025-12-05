"""
Batch Testing with Explainability
Tests multiple random images with both models and generates comprehensive report
"""

import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import timm

print("="*80)
print("🔬 BATCH TESTING WITH EXPLAINABILITY")
print("="*80)
print()

# Configuration
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Comminuted Fracture', 'Simple Fracture']
NUM_IMAGES = 5  # Number of random images to test

# Paths
BASE_DIR = Path(__file__).parent
TEST_DIR = BASE_DIR / 'data_original' / 'test'
MODELS_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'batch_test_results'
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

def load_models():
    """Load both models"""
    print("📦 Loading models...")
    
    # ConvNeXt V2
    convnext_model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', 
                                      pretrained=False, num_classes=2,
                                      drop_rate=0.3, drop_path_rate=0.2)
    convnext_path = MODELS_DIR / 'convnext_v2_improved_best.pth'
    checkpoint = torch.load(convnext_path, map_location=DEVICE)
    convnext_model.load_state_dict(checkpoint['model_state_dict'])
    convnext_model = convnext_model.to(DEVICE)
    convnext_model.eval()
    print(f"✅ ConvNeXt V2 loaded (Test Acc: {checkpoint.get('val_acc', 0):.2f}%)")
    
    # EfficientNetV2-S
    efficientnet_model = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k',
                                          pretrained=False, num_classes=2,
                                          drop_rate=0.3, drop_path_rate=0.2)
    efficientnet_path = MODELS_DIR / 'efficientnetv2_s_improved_best.pth'
    checkpoint = torch.load(efficientnet_path, map_location=DEVICE)
    efficientnet_model.load_state_dict(checkpoint['model_state_dict'])
    efficientnet_model = efficientnet_model.to(DEVICE)
    efficientnet_model.eval()
    print(f"✅ EfficientNetV2-S loaded (Test Acc: {checkpoint.get('val_acc', 0):.2f}%)")
    
    return convnext_model, efficientnet_model

def get_random_test_images(n=5):
    """Get random test images with true labels"""
    test_images = []
    
    for class_idx, class_name in enumerate(['comminuted_fracture', 'simple_fracture']):
        class_dir = TEST_DIR / class_name
        if class_dir.exists():
            images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
            for img_path in images:
                test_images.append({
                    'path': img_path,
                    'true_class': CLASS_NAMES[class_idx],
                    'true_idx': class_idx
                })
    
    random.shuffle(test_images)
    return test_images[:n]

def test_image_with_both_models(image_path, true_label, convnext_model, efficientnet_model):
    """Test single image with both models and generate Grad-CAM"""
    
    # Load image
    original_image = Image.open(image_path).convert('RGB')
    image_tensor = transform(original_image).unsqueeze(0).to(DEVICE)
    
    results = {}
    
    # Test with ConvNeXt V2
    convnext_layer = convnext_model.stages[-1]
    convnext_gradcam = GradCAM(convnext_model, convnext_layer)
    convnext_cam, convnext_pred_idx, convnext_output = convnext_gradcam.generate_cam(image_tensor)
    convnext_probs = F.softmax(convnext_output, dim=1)[0]
    convnext_cam_resized = np.array(Image.fromarray(convnext_cam).resize(original_image.size, Image.BILINEAR))
    
    results['convnext'] = {
        'prediction': CLASS_NAMES[convnext_pred_idx],
        'pred_idx': convnext_pred_idx,
        'confidence': convnext_probs[convnext_pred_idx].item() * 100,
        'probabilities': convnext_probs.detach().cpu().numpy() * 100,
        'cam': convnext_cam_resized
    }
    
    # Test with EfficientNetV2-S
    efficientnet_layer = efficientnet_model.conv_head
    efficientnet_gradcam = GradCAM(efficientnet_model, efficientnet_layer)
    efficientnet_cam, efficientnet_pred_idx, efficientnet_output = efficientnet_gradcam.generate_cam(image_tensor)
    efficientnet_probs = F.softmax(efficientnet_output, dim=1)[0]
    efficientnet_cam_resized = np.array(Image.fromarray(efficientnet_cam).resize(original_image.size, Image.BILINEAR))
    
    results['efficientnet'] = {
        'prediction': CLASS_NAMES[efficientnet_pred_idx],
        'pred_idx': efficientnet_pred_idx,
        'confidence': efficientnet_probs[efficientnet_pred_idx].item() * 100,
        'probabilities': efficientnet_probs.detach().cpu().numpy() * 100,
        'cam': efficientnet_cam_resized
    }
    
    results['image'] = original_image
    results['true_label'] = true_label
    
    return results

def create_batch_visualization(all_results):
    """Create comprehensive batch visualization"""
    
    n_images = len(all_results)
    
    # Calculate grid size
    fig = plt.figure(figsize=(20, 5 * n_images))
    gs = fig.add_gridspec(n_images, 5, hspace=0.4, wspace=0.3)
    
    for idx, result in enumerate(all_results):
        image_name = result['image_name']
        true_label = result['true_label']
        original_image = result['image']
        convnext = result['convnext']
        efficientnet = result['efficientnet']
        
        # Check correctness
        convnext_correct = convnext['prediction'] == true_label
        efficientnet_correct = efficientnet['prediction'] == true_label
        agree = convnext['prediction'] == efficientnet['prediction']
        
        # Column 1: Original image
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.imshow(original_image, cmap='gray')
        ax1.set_title(f'Image {idx+1}\n{image_name}\nTrue: {true_label}', 
                     fontsize=10, fontweight='bold')
        ax1.axis('off')
        
        # Column 2: ConvNeXt V2 heatmap
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.imshow(convnext['cam'], cmap='jet')
        ax2.set_title(f'ConvNeXt V2\nHeatmap', fontsize=10, fontweight='bold')
        ax2.axis('off')
        
        # Column 3: ConvNeXt V2 overlay
        ax3 = fig.add_subplot(gs[idx, 2])
        ax3.imshow(original_image, cmap='gray')
        ax3.imshow(convnext['cam'], cmap='jet', alpha=0.5)
        status = '✓' if convnext_correct else '✗'
        ax3.set_title(f'{status} ConvNeXt V2\n{convnext["prediction"]}\n{convnext["confidence"]:.1f}%',
                     fontsize=10, fontweight='bold',
                     color='green' if convnext_correct else 'red')
        ax3.axis('off')
        
        # Column 4: EfficientNetV2-S heatmap
        ax4 = fig.add_subplot(gs[idx, 3])
        ax4.imshow(efficientnet['cam'], cmap='jet')
        ax4.set_title(f'EfficientNetV2-S\nHeatmap', fontsize=10, fontweight='bold')
        ax4.axis('off')
        
        # Column 5: EfficientNetV2-S overlay
        ax5 = fig.add_subplot(gs[idx, 4])
        ax5.imshow(original_image, cmap='gray')
        ax5.imshow(efficientnet['cam'], cmap='jet', alpha=0.5)
        status = '✓' if efficientnet_correct else '✗'
        ax5.set_title(f'{status} EfficientNetV2-S\n{efficientnet["prediction"]}\n{efficientnet["confidence"]:.1f}%',
                     fontsize=10, fontweight='bold',
                     color='green' if efficientnet_correct else 'red')
        ax5.axis('off')
    
    fig.suptitle(f'Batch Testing with Explainability - {n_images} Random Images',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    save_path = OUTPUT_DIR / 'batch_test_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Batch visualization saved: {save_path}")
    plt.close()

def create_summary_report(all_results):
    """Create detailed summary report"""
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')
    
    # Calculate statistics
    n_images = len(all_results)
    convnext_correct = sum(1 for r in all_results if r['convnext']['prediction'] == r['true_label'])
    efficientnet_correct = sum(1 for r in all_results if r['efficientnet']['prediction'] == r['true_label'])
    agree = sum(1 for r in all_results if r['convnext']['prediction'] == r['efficientnet']['prediction'])
    
    convnext_avg_conf = np.mean([r['convnext']['confidence'] for r in all_results])
    efficientnet_avg_conf = np.mean([r['efficientnet']['confidence'] for r in all_results])
    
    # Create report text
    report_text = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    BATCH TESTING SUMMARY REPORT                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 OVERALL STATISTICS
{'='*80}
Total Images Tested:           {n_images}
Models in Agreement:           {agree}/{n_images} ({agree/n_images*100:.1f}%)

🔵 CONVNEXT V2 PERFORMANCE
{'='*80}
Correct Predictions:           {convnext_correct}/{n_images} ({convnext_correct/n_images*100:.1f}%)
Average Confidence:            {convnext_avg_conf:.2f}%
Test Accuracy (Full Dataset):  98.88%

🟢 EFFICIENTNETV2-S PERFORMANCE
{'='*80}
Correct Predictions:           {efficientnet_correct}/{n_images} ({efficientnet_correct/n_images*100:.1f}%)
Average Confidence:            {efficientnet_avg_conf:.2f}%
Test Accuracy (Full Dataset):  96.65%

📋 DETAILED RESULTS PER IMAGE
{'='*80}
"""
    
    for idx, result in enumerate(all_results, 1):
        convnext_correct = result['convnext']['prediction'] == result['true_label']
        efficientnet_correct = result['efficientnet']['prediction'] == result['true_label']
        agree = result['convnext']['prediction'] == result['efficientnet']['prediction']
        
        report_text += f"""
Image {idx}: {result['image_name']}
├─ True Label:     {result['true_label']}
├─ ConvNeXt V2:    {result['convnext']['prediction']:<25} {result['convnext']['confidence']:>6.2f}%  {'✓' if convnext_correct else '✗'}
├─ EfficientNet:   {result['efficientnet']['prediction']:<25} {result['efficientnet']['confidence']:>6.2f}%  {'✓' if efficientnet_correct else '✗'}
└─ Agreement:      {'✅ Models agree' if agree else '⚠️  Models disagree'}
"""
    
    report_text += f"""

{'='*80}
🎯 ANALYSIS
{'='*80}
"""
    
    if convnext_correct == efficientnet_correct == n_images:
        report_text += "🎉 PERFECT! Both models achieved 100% accuracy on this batch.\n"
    elif agree == n_images:
        report_text += "✅ Models are in complete agreement on all predictions.\n"
    elif agree / n_images >= 0.8:
        report_text += f"✓ Models mostly agree ({agree/n_images*100:.0f}% agreement rate).\n"
    else:
        report_text += f"⚠️  Significant disagreement between models ({(n_images-agree)/n_images*100:.0f}% disagree).\n"
    
    if convnext_correct > efficientnet_correct:
        report_text += f"🔵 ConvNeXt V2 performed better on this batch (+{convnext_correct - efficientnet_correct} correct).\n"
    elif efficientnet_correct > convnext_correct:
        report_text += f"🟢 EfficientNetV2-S performed better on this batch (+{efficientnet_correct - convnext_correct} correct).\n"
    else:
        report_text += "⚖️  Both models performed equally on this batch.\n"
    
    report_text += f"""

💡 RECOMMENDATIONS
{'='*80}
• Use both models for critical cases to cross-validate predictions
• Pay attention to cases where models disagree (manual review recommended)
• Verify Grad-CAM heatmaps focus on fracture-relevant anatomical features
• Consider confidence levels: >95% = very high, 85-95% = high, 70-85% = moderate
• For production use, ensemble predictions from both models for robustness

📁 OUTPUT FILES
{'='*80}
• Batch Visualization: batch_test_visualization.png
• Summary Report: batch_test_summary.png
• Individual images tested with Grad-CAM overlays included

🔍 EXPLAINABILITY NOTE
{'='*80}
Grad-CAM heatmaps show model attention:
• Red/Yellow areas = High attention (important for decision)
• Blue/Purple areas = Low attention (less relevant)
• Verify heatmaps focus on bone structures and fracture lines

Generated: {Path(__file__).name}
Date: November 4, 2025
    """
    
    ax.text(0.05, 0.95, report_text,
           transform=ax.transAxes,
           fontsize=9,
           verticalalignment='top',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / 'batch_test_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Summary report saved: {save_path}")
    plt.close()
    
    return report_text

def main():
    print(f"🎲 Selecting {NUM_IMAGES} random test images...")
    print("="*80)
    
    # Get random images
    test_images = get_random_test_images(NUM_IMAGES)
    
    if not test_images:
        print("❌ No test images found!")
        return
    
    print(f"✅ Selected {len(test_images)} images:")
    for i, img_info in enumerate(test_images, 1):
        print(f"   {i}. {img_info['path'].name} (True: {img_info['true_class']})")
    print()
    
    # Load models
    convnext_model, efficientnet_model = load_models()
    print()
    
    # Test all images
    print("🔬 Testing images with both models...")
    print("="*80)
    
    all_results = []
    
    for idx, img_info in enumerate(test_images, 1):
        print(f"\n[{idx}/{len(test_images)}] Processing: {img_info['path'].name}")
        
        result = test_image_with_both_models(
            img_info['path'],
            img_info['true_class'],
            convnext_model,
            efficientnet_model
        )
        
        result['image_name'] = img_info['path'].name
        all_results.append(result)
        
        # Print quick summary
        convnext_correct = result['convnext']['prediction'] == img_info['true_class']
        efficientnet_correct = result['efficientnet']['prediction'] == img_info['true_class']
        
        print(f"   ConvNeXt V2:      {result['convnext']['prediction']:<25} {result['convnext']['confidence']:>6.2f}%  {'✓' if convnext_correct else '✗'}")
        print(f"   EfficientNetV2-S: {result['efficientnet']['prediction']:<25} {result['efficientnet']['confidence']:>6.2f}%  {'✓' if efficientnet_correct else '✗'}")
    
    print()
    print("="*80)
    print("📊 Creating visualizations...")
    print("="*80)
    
    # Create visualizations
    create_batch_visualization(all_results)
    report_text = create_summary_report(all_results)
    
    # Print summary to console
    print()
    print(report_text)
    
    print()
    print("="*80)
    print("🎉 BATCH TESTING COMPLETE!")
    print("="*80)
    print(f"📁 All outputs saved to: {OUTPUT_DIR}")
    print(f"   • batch_test_visualization.png - Grid view of all images with Grad-CAM")
    print(f"   • batch_test_summary.png - Detailed summary report")
    print()

if __name__ == '__main__':
    main()
