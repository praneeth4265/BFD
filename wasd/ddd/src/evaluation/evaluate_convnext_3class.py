"""
Evaluate the Best Saved ConvNeXt V2 Model on Test Set
Loads the best checkpoint and runs comprehensive evaluation
"""

import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

print("="*80)
print("🧪 EVALUATING CONVNEXT V2 - 3-CLASS AUGMENTED DATASET")
print("="*80)
print()

# Paths
BASE_DIR = Path(__file__).parent
AUGMENTED_DIR = BASE_DIR / 'datasets' / 'augmented'
TEST_DIR = AUGMENTED_DIR / 'test'
MODEL_PATH = BASE_DIR / 'models' / 'checkpoints' / 'convnextv2_3class_augmented_best.pth'
RESULTS_PATH = BASE_DIR / 'models' / 'checkpoints' / 'convnextv2_3class_augmented_results.json'

# Configuration
CONFIG = {
    'model_name': 'convnextv2_base.fcmae_ft_in22k_in1k',
    'img_size': 224,
    'batch_size': 32,
    'num_classes': 3,
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"📁 Test Data: {TEST_DIR}")
print(f"💾 Model Checkpoint: {MODEL_PATH}")
print(f"🖥️  Device: {CONFIG['device']}")
print()

# Dataset class
class BoneFractureDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = ['comminuted_fracture', 'no_fracture', 'simple_fracture']
        
        self.images = []
        self.labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.png'):
                    self.images.append(img_path)
                    self.labels.append(class_idx)
                for img_path in class_dir.glob('*.jpg'):
                    self.images.append(img_path)
                    self.labels.append(class_idx)
        
        print(f"   Loaded {len(self.images)} test images")
        from collections import Counter
        label_counts = Counter(self.labels)
        for class_idx, class_name in enumerate(self.classes):
            print(f"      {class_name}: {label_counts[class_idx]} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Load test dataset
print("📦 Loading test dataset...")
test_transform = transforms.Compose([
    transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = BoneFractureDataset(TEST_DIR, transform=test_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=CONFIG['num_workers'],
    pin_memory=True
)
print()

# Build model
print("🏗️  Building model...")
model = timm.create_model(
    CONFIG['model_name'],
    pretrained=False,  # Don't load ImageNet weights
    num_classes=CONFIG['num_classes']
)

# Load checkpoint
print("💾 Loading best checkpoint...")
checkpoint = torch.load(MODEL_PATH, map_location=CONFIG['device'])
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(CONFIG['device'])
model.eval()

print(f"✅ Model loaded successfully")
print(f"   Checkpoint from Epoch: {checkpoint['epoch']}")
print(f"   Validation Accuracy: {checkpoint['val_acc']:.2f}%")
print()

# Evaluation function
def evaluate_detailed(model, loader, device):
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    correct = 0
    total = 0
    
    print("🧪 Evaluating on test set...")
    with torch.no_grad():
        pbar = tqdm(loader, desc='Testing')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            pbar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})
    
    accuracy = 100. * correct / total
    
    return accuracy, all_preds, all_labels, all_probs

# Run evaluation
start_time = time.time()
test_acc, predictions, labels, probabilities = evaluate_detailed(
    model, test_loader, CONFIG['device']
)
eval_time = time.time() - start_time

print()
print("="*80)
print("📊 EVALUATION RESULTS")
print("="*80)
print()

# Basic metrics
print(f"✅ Test Accuracy: {test_acc:.2f}%")
print(f"⏱️  Evaluation Time: {eval_time:.2f} seconds")
print()

# Confusion Matrix
print("📋 Confusion Matrix:")
cm = confusion_matrix(labels, predictions)
class_names = ['comminuted_fracture', 'no_fracture', 'simple_fracture']
print(f"{'':25} {'Predicted':^50}")
print(f"{'':25} {' | '.join([f'{c[:10]:^10}' for c in class_names])}")
print("-" * 80)
for i, actual in enumerate(class_names):
    row = f"{actual[:20]:20} | " + " | ".join([f"{cm[i][j]:^10}" for j in range(len(class_names))])
    print(row)
print()

# Classification Report
print("📊 Detailed Classification Report:")
report = classification_report(
    labels, predictions,
    target_names=class_names,
    digits=4
)
print(report)
print()

# Per-class accuracy
print("🎯 Per-Class Accuracy:")
for i, class_name in enumerate(class_names):
    class_correct = sum((np.array(labels) == i) & (np.array(predictions) == i))
    class_total = sum(np.array(labels) == i)
    class_acc = 100. * class_correct / class_total if class_total > 0 else 0
    print(f"   {class_name[:20]:20}: {class_acc:.2f}% ({class_correct}/{class_total})")
print()

# Save results
results = {
    'model': CONFIG['model_name'],
    'num_classes': CONFIG['num_classes'],
    'classes': class_names,
    'img_size': CONFIG['img_size'],
    'checkpoint_epoch': int(checkpoint['epoch']),
    'checkpoint_val_acc': float(checkpoint['val_acc']),
    'test_acc': float(test_acc),
    'test_samples': len(test_dataset),
    'evaluation_time_seconds': eval_time,
    'confusion_matrix': cm.tolist(),
    'classification_report': classification_report(
        labels, predictions,
        target_names=class_names,
        output_dict=True
    ),
    'evaluated_at': datetime.now().isoformat()
}

with open(RESULTS_PATH, 'w') as f:
    json.dump(results, f, indent=2)

print(f"💾 Results saved to: {RESULTS_PATH}")
print()
print("="*80)
print("🎉 EVALUATION COMPLETE!")
print("="*80)
print()
print(f"📌 Summary:")
print(f"   ✅ Test Accuracy: {test_acc:.2f}%")
print(f"   📊 Test Samples: {len(test_dataset)}")
print(f"   🏆 Model: {CONFIG['model_name']}")
print(f"   💾 Results: {RESULTS_PATH.name}")
print()
