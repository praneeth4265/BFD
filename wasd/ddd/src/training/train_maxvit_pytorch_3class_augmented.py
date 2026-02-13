"""
PyTorch Training Script - MaxViT on 3-Class Augmented Dataset
High-performance GPU training with Hybrid CNN-Transformer architecture
Model: MaxViT-Tiny (Multi-axis Vision Transformer)
Dataset: 20,530 augmented images (3 classes: comminuted, simple, no_fracture)
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm

print("="*80)
print("🚀 PYTORCH MAXVIT TRAINING - 3-CLASS AUGMENTED DATASET")
print("="*80)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("="*80)
print()

# Configuration
CONFIG = {
    'model_name': 'maxvit_tiny_tf_224.in1k',  # MaxViT-Tiny with ImageNet-1k pretrained weights
    'img_size': 224,
    'batch_size': 24,  # Slightly smaller batch size due to transformer memory requirements
    'learning_rate': 1e-4,
    'epochs': 30,
    'num_classes': 3,
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_best_only': True,
    'early_stopping_patience': 5,
}

# Paths
BASE_DIR = Path(__file__).parent
AUGMENTED_DIR = BASE_DIR / 'datasets' / 'augmented'
TRAIN_DIR = AUGMENTED_DIR / 'train'
VAL_DIR = AUGMENTED_DIR / 'val'
TEST_DIR = AUGMENTED_DIR / 'test'
MODEL_DIR = BASE_DIR / 'models' / 'checkpoints'
LOGS_DIR = BASE_DIR / 'logs'

MODEL_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)

print(f"📁 Dataset Paths:")
print(f"   Train: {TRAIN_DIR}")
print(f"   Val:   {VAL_DIR}")
print(f"   Test:  {TEST_DIR}")
print()

# Custom Dataset
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
        
        print(f"   Loaded {len(self.images)} images from {data_dir.name}")
        if len(self.labels) > 0:
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

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
print("📦 Loading datasets...")
train_dataset = BoneFractureDataset(TRAIN_DIR, transform=train_transform)
val_dataset = BoneFractureDataset(VAL_DIR, transform=val_transform)
test_dataset = BoneFractureDataset(TEST_DIR, transform=val_transform)
print()

# Data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=CONFIG['num_workers'],
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=CONFIG['num_workers'],
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=CONFIG['num_workers'],
    pin_memory=True
)

print(f"📊 Dataset Statistics:")
print(f"   Train: {len(train_dataset)} images ({len(train_loader)} batches)")
print(f"   Val:   {len(val_dataset)} images ({len(val_loader)} batches)")
print(f"   Test:  {len(test_dataset)} images ({len(test_loader)} batches)")
print()

# Build model
print("🏗️  Building MaxViT-Tiny model...")
print("   Architecture: Hybrid CNN-Transformer with Multi-Axis Attention")
model = timm.create_model(
    CONFIG['model_name'],
    pretrained=True,
    num_classes=CONFIG['num_classes']
)
model = model.to(CONFIG['device'])

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✅ Model built successfully")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(loader):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(loader):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Training loop
print("="*80)
print("🚀 STARTING TRAINING")
print("="*80)
print()

best_val_acc = 0.0
best_epoch = 0
patience_counter = 0
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'lr': []
}

start_time = time.time()

for epoch in range(CONFIG['epochs']):
    epoch_start = time.time()
    
    print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
    print("-" * 80)
    
    # Train
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, CONFIG['device']
    )
    
    # Validate
    val_loss, val_acc = validate(
        model, val_loader, criterion, CONFIG['device']
    )
    
    # Update learning rate
    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)
    
    epoch_time = time.time() - epoch_start
    
    print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    print(f"   LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        patience_counter = 0
        
        if CONFIG['save_best_only']:
            model_path = MODEL_DIR / 'maxvit_3class_augmented_best.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': CONFIG
            }, model_path)
            print(f"   ✅ Best model saved! (Val Acc: {val_acc:.2f}%)")
    else:
        patience_counter += 1
        if patience_counter >= CONFIG['early_stopping_patience']:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break
    
    print()

training_time = time.time() - start_time

print("="*80)
print("✅ TRAINING COMPLETE")
print("="*80)
print(f"   Total time: {training_time/60:.2f} minutes")
print(f"   Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
print()

# Evaluate on test set
print("📊 Evaluating on test set...")
model_path = MODEL_DIR / 'maxvit_3class_augmented_best.pth'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])

test_loss, test_acc = validate(model, test_loader, criterion, CONFIG['device'])

print(f"   Test Loss: {test_loss:.4f}")
print(f"   Test Acc:  {test_acc:.2f}%")
print()

# Save results
results = {
    'model': CONFIG['model_name'],
    'architecture': 'Hybrid CNN-Transformer (MaxViT)',
    'num_classes': CONFIG['num_classes'],
    'classes': ['comminuted_fracture', 'no_fracture', 'simple_fracture'],
    'img_size': CONFIG['img_size'],
    'batch_size': CONFIG['batch_size'],
    'epochs_trained': len(history['train_loss']),
    'best_epoch': best_epoch,
    'best_val_acc': best_val_acc,
    'test_acc': test_acc,
    'test_loss': test_loss,
    'training_time_minutes': training_time / 60,
    'total_parameters': total_params,
    'trainable_parameters': trainable_params,
    'history': history,
    'dataset': {
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'total_size': len(train_dataset) + len(val_dataset) + len(test_dataset)
    }
}

results_path = MODEL_DIR / 'maxvit_3class_augmented_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"💾 Results saved to: {results_path}")
print()
print("="*80)
print("🎉 MAXVIT TRAINING DONE!")
print("="*80)
