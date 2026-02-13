"""
PyTorch Training Script - EfficientNetV2-S (IMPROVED WITH REGULARIZATION)
Prevents overfitting with: dropout, weight decay, better augmentation, early stopping
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

print("="*80)
print("🚀 PYTORCH EFFICIENTNETV2-S TRAINING (IMPROVED - ANTI-OVERFITTING)")
print("="*80)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("="*80)
print()

# Configuration with ANTI-OVERFITTING measures
CONFIG = {
    'model_name': 'tf_efficientnetv2_s.in21k_ft_in1k',
    'img_size': 224,
    'batch_size': 16,  # Optimized for GPU memory
    'learning_rate': 5e-5,  # Lower learning rate
    'weight_decay': 0.01,  # L2 regularization
    'dropout_rate': 0.3,  # Dropout for regularization
    'epochs': 30,  # Reduced epochs
    'num_classes': 2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'early_stopping_patience': 8,
    'label_smoothing': 0.1  # Label smoothing
}

# Paths
DATA_DIR = Path(__file__).parent / 'data_original'
TRAIN_DIR = DATA_DIR / 'train'
VAL_DIR = DATA_DIR / 'val'
TEST_DIR = DATA_DIR / 'test'
MODEL_DIR = Path(__file__).parent / 'models'
LOGS_DIR = Path(__file__).parent / 'logs'

MODEL_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Custom Dataset
class BoneFractureDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = ['comminuted_fracture', 'simple_fracture']
        
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
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ENHANCED Data transforms with MORE augmentation to prevent overfitting
train_transform = transforms.Compose([
    transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
])

val_transform = transforms.Compose([
    transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("📦 LOADING DATA")
print("="*80)

# Create datasets
train_dataset = BoneFractureDataset(TRAIN_DIR, transform=train_transform)
val_dataset = BoneFractureDataset(VAL_DIR, transform=val_transform)
test_dataset = BoneFractureDataset(TEST_DIR, transform=val_transform)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                         shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                       shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                        shuffle=False, num_workers=4, pin_memory=True)

print(f"✅ Data loaded successfully")
print()

print("🏗️  BUILDING EFFICIENTNETV2-S MODEL WITH REGULARIZATION")
print("="*80)

# Create model with dropout
device = torch.device(CONFIG['device'])
model = timm.create_model(CONFIG['model_name'], pretrained=True, num_classes=CONFIG['num_classes'], 
                         drop_rate=CONFIG['dropout_rate'], drop_path_rate=0.2)
model = model.to(device)

print(f"✅ Model: {CONFIG['model_name']}")
print(f"✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"✅ Dropout rate: {CONFIG['dropout_rate']}")
print(f"✅ Weight decay: {CONFIG['weight_decay']}")
print(f"✅ Device: {device}")
print()

# Loss with LABEL SMOOTHING (anti-overfitting)
criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])

# Optimizer with WEIGHT DECAY (L2 regularization)
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], 
                       weight_decay=CONFIG['weight_decay'])

# Cosine annealing scheduler (smoother learning rate decay)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

print("⚙️  Training Configuration (ANTI-OVERFITTING)")
print("="*80)
print(f"Optimizer: AdamW (lr={CONFIG['learning_rate']}, weight_decay={CONFIG['weight_decay']})")
print(f"Loss: CrossEntropyLoss (label_smoothing={CONFIG['label_smoothing']})")
print(f"Scheduler: CosineAnnealingWarmRestarts")
print(f"Batch size: {CONFIG['batch_size']}")
print(f"Epochs: {CONFIG['epochs']}")
print(f"Early stopping patience: {CONFIG['early_stopping_patience']}")
print(f"Dropout: {CONFIG['dropout_rate']}")
print(f"Enhanced augmentation: RandomFlip, Rotation, ColorJitter, Affine, Erasing")
print("="*80)
print()

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(loader), 100 * correct / total

# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(loader), 100 * correct / total

print("🚀 STARTING TRAINING (WITH OVERFITTING PREVENTION)")
print("="*80)
print()

# Training loop with early stopping
best_val_acc = 0.0
patience_counter = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
start_time = time.time()

for epoch in range(CONFIG['epochs']):
    epoch_start = time.time()
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Update scheduler
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)
    
    epoch_time = time.time() - epoch_start
    
    # Calculate overfitting metric
    overfit_gap = train_acc - val_acc
    
    print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] ({epoch_time:.1f}s, LR={current_lr:.2e})")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"  Overfit Gap: {overfit_gap:.2f}% ", end="")
    
    if overfit_gap > 15:
        print("🚨 (HIGH OVERFITTING!)")
    elif overfit_gap > 10:
        print("⚠️  (Moderate overfitting)")
    elif overfit_gap > 5:
        print("✓ (Acceptable)")
    else:
        print("✅ (Good generalization)")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'config': CONFIG
        }, MODEL_DIR / 'efficientnetv2_s_improved_best.pth')
        print(f"  ✅ New best model saved! (Val Acc: {val_acc:.2f}%)")
    else:
        patience_counter += 1
        print(f"  ⏳ No improvement ({patience_counter}/{CONFIG['early_stopping_patience']})")
    
    print()
    
    # Early stopping
    if patience_counter >= CONFIG['early_stopping_patience']:
        print(f"⏹️  Early stopping triggered after {epoch+1} epochs")
        print(f"   Best validation accuracy: {best_val_acc:.2f}%")
        break

training_time = (time.time() - start_time) / 60

print("="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"Training time: {training_time:.1f} minutes")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"Total epochs: {len(history['train_loss'])}")
print()

# Load best model for testing
print("📊 EVALUATING BEST MODEL ON TEST SET")
print("="*80)

checkpoint = torch.load(MODEL_DIR / 'efficientnetv2_s_improved_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
test_loss, test_acc = validate(model, test_loader, criterion, device)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Best epoch: {checkpoint['epoch'] + 1}")
print("="*80)
print()

# Save results
results = {
    'config': CONFIG,
    'training_time_minutes': training_time,
    'best_val_acc': best_val_acc,
    'best_epoch': checkpoint['epoch'] + 1,
    'test_loss': test_loss,
    'test_acc': test_acc,
    'total_epochs_trained': len(history['train_loss']),
    'history': history,
    'improvements': [
        'Weight decay (L2 regularization)',
        'Dropout layers',
        'Label smoothing',
        'Enhanced data augmentation',
        'Gradient clipping',
        'Lower learning rate',
        'CosineAnnealing scheduler',
        'Early stopping'
    ]
}

with open(MODEL_DIR / 'efficientnetv2_s_improved_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Results saved to: {MODEL_DIR / 'efficientnetv2_s_improved_results.json'}")
print(f"✅ Best model saved to: {MODEL_DIR / 'efficientnetv2_s_improved_best.pth'}")
print()
print("="*80)
print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
print("="*80)
