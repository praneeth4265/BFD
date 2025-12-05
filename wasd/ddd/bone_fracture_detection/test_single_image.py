"""
Test Single Image with Trained Model
Predict bone fracture type for any X-ray image
"""

import argparse
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import timm
import json

def test_single_image(image_path, model_path=None):
    """Test a single image with the trained model"""
    
    # Default model path
    if model_path is None:
        model_path = Path("/home/praneeth4265/wasd/ddd/bone_fracture_detection/models/convnext_v2_improved_best.pth")
    
    results_path = model_path.parent / "convnext_v2_improved_results.json"
    
    print("="*80)
    print("🔬 BONE FRACTURE DETECTION - SINGLE IMAGE TEST")
    print("="*80)
    print()
    
    # Load config
    with open(results_path, 'r') as f:
        results = json.load(f)
        config = results['config']
    
    print(f"📊 Model Info:")
    print(f"   Model: {config['model_name']}")
    print(f"   Test Accuracy: {results['test_acc']:.2f}%")
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    print()
    
    # Load model
    print("🏗️  Loading model...")
    model = timm.create_model(config['model_name'], pretrained=False, num_classes=config['num_classes'])
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✅ Model loaded")
    print()
    
    # Load image
    print(f"📷 Loading image: {image_path}")
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"   Image size: {image.size}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    print()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    print("🔍 Analyzing X-ray...")
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        comminuted_prob = probabilities[0][0].item() * 100
        simple_prob = probabilities[0][1].item() * 100
        confidence_pct = confidence.item() * 100
    
    print()
    print("="*80)
    print("📊 PREDICTION RESULTS")
    print("="*80)
    print()
    
    # Display prediction
    if predicted.item() == 0:
        print("🔴 COMMINUTED FRACTURE")
        print(f"   Confidence: {confidence_pct:.2f}%")
    else:
        print("🔵 SIMPLE FRACTURE")
        print(f"   Confidence: {confidence_pct:.2f}%")
    
    print()
    print("📈 Probability Distribution:")
    print(f"   Comminuted Fracture: {comminuted_prob:.2f}%")
    print(f"   Simple Fracture: {simple_prob:.2f}%")
    print()
    
    # Interpretation
    print("💡 Interpretation:")
    if confidence_pct >= 95:
        print("   Very high confidence - Strong prediction")
    elif confidence_pct >= 85:
        print("   High confidence - Reliable prediction")
    elif confidence_pct >= 75:
        print("   Moderate confidence - Consider additional review")
    else:
        print("   Low confidence - Manual review recommended")
    
    print()
    print("="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    import sys
    
    parser = argparse.ArgumentParser(description='Test bone fracture detection model on a single image')
    parser.add_argument('--image', type=str, help='Path to the X-ray image')
    parser.add_argument('--model', type=str, default=None, help='Path to the model checkpoint (optional)')
    
    # Parse args
    args = parser.parse_args()
    
    # If no image provided, show usage and pick a random test image
    if args.image is None:
        print("="*80)
        print("⚠️  No image specified!")
        print("="*80)
        print("\nUsage:")
        print("  python test_single_image.py --image path/to/xray.jpg")
        print("\nOr test with a random image from test set:")
        
        # Try to find a random test image
        test_dir = Path("/home/praneeth4265/wasd/ddd/bone_fracture_detection/data_original/test")
        test_images = []
        
        if test_dir.exists():
            for class_name in ['comminuted_fracture', 'simple_fracture']:
                class_dir = test_dir / class_name
                if class_dir.exists():
                    for ext in ['*.png', '*.jpg', '*.jpeg']:
                        test_images.extend(list(class_dir.glob(ext)))
        
        if test_images:
            import random
            random_image = random.choice(test_images)
            print(f"\n🎲 Testing with random image: {random_image.name}")
            print("="*80)
            print()
            test_single_image(str(random_image), args.model)
        else:
            print("\n❌ No test images found in data_original/test/")
            sys.exit(1)
    else:
        test_single_image(args.image, args.model)
