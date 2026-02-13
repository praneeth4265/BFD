"""
Reorganize Original Dataset
Move and split the original dataset into proper train/val/test structure
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

print("="*80)
print("📂 REORGANIZING ORIGINAL DATASET")
print("="*80)
print()

# Set random seed for reproducibility
random.seed(42)

# Source directory
SOURCE_DIR = Path("/home/praneeth4265/wasd/ddd/Bone Fracture X-ray Dataset Simple vs. Comminuted Fractures/Bone Fracture/Bone Fracture/Orginal")

# Destination directory
DEST_DIR = Path("/home/praneeth4265/wasd/ddd/bone_fracture_detection/data_original")

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Class mapping
CLASS_MAPPING = {
    'Comminuted Bone Fracture': 'comminuted_fracture',
    'Simple Bone Fracture': 'simple_fracture'
}

print("📊 Source Directory:")
print(f"   {SOURCE_DIR}")
print()
print("📁 Destination Directory:")
print(f"   {DEST_DIR}")
print()
print(f"📐 Split Ratios: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}")
print("="*80)
print()

# Create destination directories
for split in ['train', 'val', 'test']:
    for class_name in CLASS_MAPPING.values():
        dest_path = DEST_DIR / split / class_name
        dest_path.mkdir(parents=True, exist_ok=True)

# Process each class
stats = {
    'train': {'comminuted_fracture': 0, 'simple_fracture': 0},
    'val': {'comminuted_fracture': 0, 'simple_fracture': 0},
    'test': {'comminuted_fracture': 0, 'simple_fracture': 0}
}

for source_class, dest_class in CLASS_MAPPING.items():
    print(f"Processing: {source_class} → {dest_class}")
    
    source_class_dir = SOURCE_DIR / source_class
    
    # Get all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(list(source_class_dir.glob(ext)))
    
    image_files = sorted(image_files)
    total_images = len(image_files)
    
    print(f"   Found {total_images} images")
    
    # Split into train/val/test
    train_files, temp_files = train_test_split(image_files, test_size=(VAL_RATIO + TEST_RATIO), random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=TEST_RATIO/(VAL_RATIO + TEST_RATIO), random_state=42)
    
    print(f"   Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # Copy files
    for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        dest_split_dir = DEST_DIR / split_name / dest_class
        
        for idx, img_path in enumerate(files):
            # Create new filename with index
            new_filename = f"{dest_class}_{split_name}_{idx:04d}{img_path.suffix}"
            dest_path = dest_split_dir / new_filename
            
            # Copy file
            shutil.copy2(img_path, dest_path)
            stats[split_name][dest_class] += 1
    
    print(f"   ✅ Completed\n")

print("="*80)
print("✅ REORGANIZATION COMPLETE!")
print("="*80)
print()

# Print statistics
print("📊 DATASET STATISTICS:")
print("="*80)
print()

for split in ['train', 'val', 'test']:
    total = sum(stats[split].values())
    print(f"{split.upper()}:")
    print(f"  Comminuted Fracture: {stats[split]['comminuted_fracture']:,}")
    print(f"  Simple Fracture: {stats[split]['simple_fracture']:,}")
    print(f"  Total: {total:,}")
    print()

grand_total = sum(sum(stats[split].values()) for split in stats)
print(f"GRAND TOTAL: {grand_total:,} images")
print()

print("="*80)
print("📁 Directory Structure:")
print("="*80)
print(f"{DEST_DIR}/")
print("├── train/")
print("│   ├── comminuted_fracture/")
print("│   └── simple_fracture/")
print("├── val/")
print("│   ├── comminuted_fracture/")
print("│   └── simple_fracture/")
print("└── test/")
print("    ├── comminuted_fracture/")
print("    └── simple_fracture/")
print()
print("="*80)
print("🎉 READY FOR TRAINING!")
print("="*80)
