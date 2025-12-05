"""
Compare ConvNeXt V2 and EfficientNetV2-S Models on Random Test Images
Tests both models side-by-side with confidence scores and timing
"""

import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

print("="*100)
print("🔬 MODEL COMPARISON: ConvNeXt V2 vs EfficientNetV2-S")
print("="*100)
print()

# Configuration
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Comminuted Fracture', 'Simple Fracture']

# Paths
BASE_DIR = Path(__file__).parent
TEST_DIR = BASE_DIR / 'data_original' / 'test'
MODELS_DIR = BASE_DIR / 'models'

CONVNEXT_MODEL_PATH = MODELS_DIR / 'convnext_v2_improved_best.pth'
EFFICIENTNET_MODEL_PATH = MODELS_DIR / 'efficientnetv2_s_improved_best.pth'

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_name, model_path, num_classes=2, dropout=0.3):
    """Load a trained model"""
    if 'convnext' in model_name:
        model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', 
                                 pretrained=False, num_classes=num_classes, 
                                 drop_rate=dropout, drop_path_rate=0.2)
    else:  # efficientnet
        model = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', 
                                 pretrained=False, num_classes=num_classes, 
                                 drop_rate=dropout, drop_path_rate=0.2)
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    return model

def predict_image(model, image_path):
    """Predict single image with timing"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    confidence, predicted = torch.max(probabilities, 1)
    
    return {
        'predicted_class': CLASS_NAMES[predicted.item()],
        'predicted_idx': predicted.item(),
        'confidence': confidence.item() * 100,
        'probabilities': probabilities[0].cpu().numpy() * 100,
        'inference_time_ms': inference_time
    }

def get_random_test_images(n=5):
    """Get random test images with their true labels"""
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
    
    # Shuffle and select random samples
    random.shuffle(test_images)
    return test_images[:n]

def main():
    print(f"📍 Device: {DEVICE}")
    print(f"📂 Test directory: {TEST_DIR}")
    print()
    
    # Check if models exist
    if not CONVNEXT_MODEL_PATH.exists():
        print(f"❌ ConvNeXt model not found: {CONVNEXT_MODEL_PATH}")
        return
    if not EFFICIENTNET_MODEL_PATH.exists():
        print(f"❌ EfficientNet model not found: {EFFICIENTNET_MODEL_PATH}")
        return
    
    print("🔄 Loading models...")
    print("-" * 100)
    
    # Load ConvNeXt V2
    print("Loading ConvNeXt V2 Base...")
    convnext_start = time.time()
    convnext_model = load_model('convnext', CONVNEXT_MODEL_PATH)
    convnext_load_time = time.time() - convnext_start
    print(f"✅ ConvNeXt V2 loaded in {convnext_load_time:.2f}s")
    print(f"   Parameters: {sum(p.numel() for p in convnext_model.parameters()):,}")
    
    # Load EfficientNetV2-S
    print("\nLoading EfficientNetV2-S...")
    efficientnet_start = time.time()
    efficientnet_model = load_model('efficientnet', EFFICIENTNET_MODEL_PATH)
    efficientnet_load_time = time.time() - efficientnet_start
    print(f"✅ EfficientNetV2-S loaded in {efficientnet_load_time:.2f}s")
    print(f"   Parameters: {sum(p.numel() for p in efficientnet_model.parameters()):,}")
    print()
    
    # Get random test images
    print("🎲 Selecting random test images...")
    print("-" * 100)
    test_images = get_random_test_images(n=5)
    print(f"✅ Selected {len(test_images)} random images from test set")
    print()
    
    # Test each image on both models
    convnext_correct = 0
    efficientnet_correct = 0
    convnext_total_time = 0
    efficientnet_total_time = 0
    
    for idx, test_item in enumerate(test_images, 1):
        img_path = test_item['path']
        true_class = test_item['true_class']
        true_idx = test_item['true_idx']
        
        print("=" * 100)
        print(f"📷 TEST IMAGE {idx}/{len(test_images)}")
        print("=" * 100)
        print(f"Image: {img_path.name}")
        print(f"True Label: {true_class}")
        print()
        
        # ConvNeXt V2 prediction
        print("🔵 ConvNeXt V2 Prediction:")
        print("-" * 100)
        convnext_result = predict_image(convnext_model, img_path)
        convnext_total_time += convnext_result['inference_time_ms']
        
        print(f"Predicted: {convnext_result['predicted_class']}")
        print(f"Confidence: {convnext_result['confidence']:.2f}%")
        print(f"Inference Time: {convnext_result['inference_time_ms']:.2f}ms")
        print(f"Probabilities:")
        for i, class_name in enumerate(CLASS_NAMES):
            print(f"  {class_name}: {convnext_result['probabilities'][i]:.2f}%")
        
        convnext_correct_pred = convnext_result['predicted_idx'] == true_idx
        if convnext_correct_pred:
            convnext_correct += 1
            print("✅ CORRECT")
        else:
            print("❌ INCORRECT")
        print()
        
        # EfficientNetV2-S prediction
        print("🟢 EfficientNetV2-S Prediction:")
        print("-" * 100)
        efficientnet_result = predict_image(efficientnet_model, img_path)
        efficientnet_total_time += efficientnet_result['inference_time_ms']
        
        print(f"Predicted: {efficientnet_result['predicted_class']}")
        print(f"Confidence: {efficientnet_result['confidence']:.2f}%")
        print(f"Inference Time: {efficientnet_result['inference_time_ms']:.2f}ms")
        print(f"Probabilities:")
        for i, class_name in enumerate(CLASS_NAMES):
            print(f"  {class_name}: {efficientnet_result['probabilities'][i]:.2f}%")
        
        efficientnet_correct_pred = efficientnet_result['predicted_idx'] == true_idx
        if efficientnet_correct_pred:
            efficientnet_correct += 1
            print("✅ CORRECT")
        else:
            print("❌ INCORRECT")
        print()
        
        # Comparison
        print("⚖️  COMPARISON:")
        print("-" * 100)
        if convnext_correct_pred and efficientnet_correct_pred:
            print("✅ Both models predicted correctly")
            conf_diff = abs(convnext_result['confidence'] - efficientnet_result['confidence'])
            print(f"Confidence difference: {conf_diff:.2f}%")
            if convnext_result['confidence'] > efficientnet_result['confidence']:
                print(f"ConvNeXt V2 is more confident (+{conf_diff:.2f}%)")
            elif efficientnet_result['confidence'] > convnext_result['confidence']:
                print(f"EfficientNetV2-S is more confident (+{conf_diff:.2f}%)")
            else:
                print("Both models equally confident")
        elif not convnext_correct_pred and not efficientnet_correct_pred:
            print("❌ Both models predicted incorrectly")
        elif convnext_correct_pred:
            print("🔵 Only ConvNeXt V2 predicted correctly")
        else:
            print("🟢 Only EfficientNetV2-S predicted correctly")
        
        time_diff = abs(convnext_result['inference_time_ms'] - efficientnet_result['inference_time_ms'])
        if convnext_result['inference_time_ms'] < efficientnet_result['inference_time_ms']:
            print(f"ConvNeXt V2 is faster by {time_diff:.2f}ms")
        elif efficientnet_result['inference_time_ms'] < convnext_result['inference_time_ms']:
            print(f"EfficientNetV2-S is faster by {time_diff:.2f}ms")
        print()
    
    # Final Summary
    print("=" * 100)
    print("📊 FINAL SUMMARY")
    print("=" * 100)
    print()
    
    print("🎯 ACCURACY:")
    print("-" * 100)
    convnext_acc = (convnext_correct / len(test_images)) * 100
    efficientnet_acc = (efficientnet_correct / len(test_images)) * 100
    print(f"ConvNeXt V2:        {convnext_correct}/{len(test_images)} correct ({convnext_acc:.1f}%)")
    print(f"EfficientNetV2-S:   {efficientnet_correct}/{len(test_images)} correct ({efficientnet_acc:.1f}%)")
    
    if convnext_acc > efficientnet_acc:
        print(f"\n🏆 Winner: ConvNeXt V2 (+{convnext_acc - efficientnet_acc:.1f}%)")
    elif efficientnet_acc > convnext_acc:
        print(f"\n🏆 Winner: EfficientNetV2-S (+{efficientnet_acc - convnext_acc:.1f}%)")
    else:
        print("\n🤝 Tie: Both models performed equally")
    print()
    
    print("⚡ SPEED:")
    print("-" * 100)
    convnext_avg_time = convnext_total_time / len(test_images)
    efficientnet_avg_time = efficientnet_total_time / len(test_images)
    print(f"ConvNeXt V2 Average:        {convnext_avg_time:.2f}ms per image")
    print(f"EfficientNetV2-S Average:   {efficientnet_avg_time:.2f}ms per image")
    
    if convnext_avg_time < efficientnet_avg_time:
        speedup = (efficientnet_avg_time / convnext_avg_time)
        print(f"\n⚡ Winner: ConvNeXt V2 ({speedup:.2f}x faster)")
    elif efficientnet_avg_time < convnext_avg_time:
        speedup = (convnext_avg_time / efficientnet_avg_time)
        print(f"\n⚡ Winner: EfficientNetV2-S ({speedup:.2f}x faster)")
    else:
        print("\n⚖️  Tie: Same inference speed")
    print()
    
    print("📦 MODEL SIZE:")
    print("-" * 100)
    convnext_params = sum(p.numel() for p in convnext_model.parameters())
    efficientnet_params = sum(p.numel() for p in efficientnet_model.parameters())
    print(f"ConvNeXt V2:        {convnext_params:,} parameters")
    print(f"EfficientNetV2-S:   {efficientnet_params:,} parameters")
    size_ratio = convnext_params / efficientnet_params
    print(f"\n📊 ConvNeXt V2 is {size_ratio:.1f}x larger than EfficientNetV2-S")
    print()
    
    print("=" * 100)
    print("🎉 COMPARISON COMPLETE!")
    print("=" * 100)

if __name__ == '__main__':
    main()
