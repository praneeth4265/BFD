"""
PyTorch Training Script - ConvNeXt V2
High-performance GPU training with proper CUDA support
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
print("🚀 PYTORCH CONVNEXT V2 TRAINING")
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
    'model_name': 'convnextv2_base.fcmae_ft_in22k_in1k',
    'img_size': 224,
    'batch_size': 16,
    'learning_rate': 1e-4,
    'epochs': 50,
    'num_classes': 2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Paths
DATA_DIR = Path(__file__).parent / 'data'
TRAIN_DIR = DATA_DIR / 'train_augmented'
VAL_DIR = DATA_DIR / 'val_processed'
TEST_DIR = DATA_DIR / 'test_processed'
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
                         shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                       shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                        shuffle=False, num_workers=4, pin_memory=True)

print(f"✅ Data loaded successfully")
print()

print("🏗️  BUILDING CONVNEXT V2 MODEL")
print("="*80)

# Create model
device = torch.device(CONFIG['device'])
model = timm.create_model(CONFIG['model_name'], pretrained=True, num_classes=CONFIG['num_classes'])
model = model.to(device)

print(f"✅ Model: {CONFIG['model_name']}")
print(f"✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"✅ Device: {device}")
print()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

print("⚙️  Training Configuration")
print("="*80)
print(f"Optimizer: Adam (lr={CONFIG['learning_rate']})")
print(f"Loss: CrossEntropyLoss")
print(f"Scheduler: ReduceLROnPlateau")
print(f"Batch size: {CONFIG['batch_size']}")
print(f"Epochs: {CONFIG['epochs']}")
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

print("🚀 STARTING TRAINING")
print("="*80)
print()

# Training loop
best_val_acc = 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
start_time = time.time()

for epoch in range(CONFIG['epochs']):
    epoch_start = time.time()
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Update scheduler
    scheduler.step(val_acc)
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    epoch_time = time.time() - epoch_start
    
    print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] ({epoch_time:.1f}s)")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_DIR / 'convnext_v2_best.pth')
        print(f"  ✅ New best model saved! (Val Acc: {val_acc:.2f}%)")
    
    print()

training_time = (time.time() - start_time) / 60

print("="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"Training time: {training_time:.1f} minutes")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print()

# Test evaluation
print("📊 EVALUATING ON TEST SET")
print("="*80)

model.load_state_dict(torch.load(MODEL_DIR / 'convnext_v2_best.pth'))
test_loss, test_acc = validate(model, test_loader, criterion, device)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
print("="*80)
print()

# Save results
results = {
    'config': CONFIG,
    'training_time_minutes': training_time,
    'best_val_acc': best_val_acc,
    'test_loss': test_loss,
    'test_acc': test_acc,
    'history': history
}

with open(MODEL_DIR / 'convnext_v2_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Results saved to: {MODEL_DIR / 'convnext_v2_results.json'}")
print(f"✅ Best model saved to: {MODEL_DIR / 'convnext_v2_best.pth'}")
print()
print("="*80)
print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
print("="*80)
