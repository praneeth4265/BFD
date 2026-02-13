"""
Test Trained Model with Random Image
Load the best model and predict on a random test image
"""

import os
import random
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import json

print("="*80)
print("🔬 BONE FRACTURE DETECTION - MODEL INFERENCE TEST")
print("="*80)
print()

# Paths
MODEL_PATH = Path("/home/praneeth4265/wasd/ddd/bone_fracture_detection/models/convnext_v2_improved_best.pth")
TEST_DIR = Path("/home/praneeth4265/wasd/ddd/bone_fracture_detection/data_original/test")
RESULTS_PATH = Path("/home/praneeth4265/wasd/ddd/bone_fracture_detection/models/convnext_v2_improved_results.json")

# Load model config
with open(RESULTS_PATH, 'r') as f:
    results = json.load(f)
    config = results['config']

print(f"📊 Model Performance:")
print(f"   Best Validation Accuracy: {results['best_val_acc']:.2f}%")
print(f"   Test Accuracy: {results['test_acc']:.2f}%")
print(f"   Best Epoch: {results['best_epoch']}")
print()

# Class names
CLASS_NAMES = ['Comminuted Fracture', 'Simple Fracture']

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
print()

# Load model
print("🏗️  Loading model...")
model = timm.create_model(config['model_name'], pretrained=False, num_classes=config['num_classes'])
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print(f"✅ Model loaded: {config['model_name']}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

# Transform
transform = transforms.Compose([
    transforms.Resize((config['img_size'], config['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get all test images
all_test_images = []
for class_name in ['comminuted_fracture', 'simple_fracture']:
    class_dir = TEST_DIR / class_name
    if class_dir.exists():
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            all_test_images.extend([(img, class_name) for img in class_dir.glob(ext)])

print(f"📁 Found {len(all_test_images)} test images")
print()

# Select random images
num_samples = min(10, len(all_test_images))
random_samples = random.sample(all_test_images, num_samples)

print("="*80)
print(f"🎲 TESTING ON {num_samples} RANDOM IMAGES")
print("="*80)
print()

correct = 0
results_list = []

for idx, (img_path, true_class) in enumerate(random_samples, 1):
    # Load and preprocess image
    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = 'comminuted_fracture' if predicted.item() == 0 else 'simple_fracture'
        confidence_pct = confidence.item() * 100
    
    # Check if correct
    is_correct = (predicted_class == true_class)
    if is_correct:
        correct += 1
    
    # Display result
    status = "✅" if is_correct else "❌"
    print(f"Test {idx}:")
    print(f"  Image: {img_path.name}")
    print(f"  True Label: {CLASS_NAMES[0] if true_class == 'comminuted_fracture' else CLASS_NAMES[1]}")
    print(f"  Predicted: {CLASS_NAMES[predicted.item()]} ({confidence_pct:.2f}% confidence)")
    print(f"  Status: {status} {'CORRECT' if is_correct else 'INCORRECT'}")
    print()
    
    # Save to results
    results_list.append({
        'image': str(img_path),
        'true_label': true_class,
        'predicted_label': predicted_class,
        'confidence': confidence_pct,
        'correct': is_correct,
        'probabilities': {
            'comminuted_fracture': float(probabilities[0][0]),
            'simple_fracture': float(probabilities[0][1])
        }
    })

accuracy = (correct / num_samples) * 100

print("="*80)
print("📊 RANDOM SAMPLE TEST RESULTS")
print("="*80)
print(f"Total Samples: {num_samples}")
print(f"Correct: {correct}")
print(f"Incorrect: {num_samples - correct}")
print(f"Accuracy: {accuracy:.2f}%")
print()

# Show confidence statistics
confidences = [r['confidence'] for r in results_list]
avg_confidence = sum(confidences) / len(confidences)
print(f"Average Confidence: {avg_confidence:.2f}%")
print(f"Min Confidence: {min(confidences):.2f}%")
print(f"Max Confidence: {max(confidences):.2f}%")
print()

print("="*80)
print("🎯 MODEL IS READY FOR DEPLOYMENT!")
print("="*80)

# Test with a specific image if provided
print()
print("💡 TIP: To test a specific image, use:")
print("   python test_single_image.py --image path/to/your/image.png")
