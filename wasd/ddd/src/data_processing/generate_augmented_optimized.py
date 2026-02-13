#!/usr/bin/env python3
"""
Optimized Augmented No Fracture Generator
- Ultra-safe GPU memory management
- Fast batch processing with quality augmentations
- Real-time memory monitoring
- High-quality diverse transformations
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
import random
import time
from tqdm import tqdm
import gc

print("="*80)
print("🚀 OPTIMIZED AUGMENTED NO_FRACTURE GENERATOR")
print("="*80)
print()

# Random seed
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# GPU Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   GPU: {gpu_name}")
    print(f"   Total Memory: {total_mem:.2f} GB")
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Monitor initial state
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    print(f"   Initial Allocated: {allocated:.3f} GB")
    print(f"   Initial Reserved: {reserved:.3f} GB")
else:
    print("   ⚠️  No GPU available, using CPU")
print()

# Paths
BASE_PATH = Path("/home/praneeth4265/wasd/ddd")
ORIGINAL_NO_FRACTURE = BASE_PATH / "datasets/original"
AUGMENTED_TARGET = BASE_PATH / "datasets/augmented"

# Calculate targets
print("📊 Calculating target counts...")
comm_train = len(list((AUGMENTED_TARGET / "train/comminuted_fracture").glob("*.*")))
simp_train = len(list((AUGMENTED_TARGET / "train/simple_fracture").glob("*.*")))
target_train = (comm_train + simp_train) // 2

comm_val = len(list((AUGMENTED_TARGET / "val/comminuted_fracture").glob("*.*")))
simp_val = len(list((AUGMENTED_TARGET / "val/simple_fracture").glob("*.*")))
target_val = (comm_val + simp_val) // 2

comm_test = len(list((AUGMENTED_TARGET / "test/comminuted_fracture").glob("*.*")))
simp_test = len(list((AUGMENTED_TARGET / "test/simple_fracture").glob("*.*")))
target_test = (comm_test + simp_test) // 2

print(f"  Train target: {target_train}")
print(f"  Val target:   {target_val}")
print(f"  Test target:  {target_test}")
print()

# High-quality augmentation functions
def apply_augmentation(img, aug_type):
    """Apply high-quality augmentations"""
    
    if aug_type == 0:
        # Rotation + Flip + Brightness
        angle = random.uniform(-15, 15)
        img = TF.rotate(img, angle, fill=0)
        if random.random() > 0.5:
            img = TF.hflip(img)
        img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
        img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
        
    elif aug_type == 1:
        # Strong rotation + Color jitter
        angle = random.uniform(-20, 20)
        img = TF.rotate(img, angle, fill=0)
        img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
        img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
        img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))
        
    elif aug_type == 2:
        # Affine transformation
        angle = random.uniform(-10, 10)
        translate = (random.randint(-20, 20), random.randint(-20, 20))
        scale = random.uniform(0.9, 1.1)
        img = TF.affine(img, angle=angle, translate=translate, scale=scale, shear=0)
        if random.random() > 0.5:
            img = TF.hflip(img)
            
    elif aug_type == 3:
        # Perspective + Brightness
        if random.random() > 0.5:
            img = TF.hflip(img)
        angle = random.uniform(-12, 12)
        img = TF.rotate(img, angle, fill=0)
        img = TF.adjust_brightness(img, random.uniform(0.75, 1.25))
        img = TF.adjust_contrast(img, random.uniform(0.75, 1.25))
        
    elif aug_type == 4:
        # Strong transformations
        angle = random.uniform(-25, 25)
        img = TF.rotate(img, angle, fill=0)
        img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
        img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
        if random.random() > 0.7:
            img = TF.hflip(img)
            
    else:
        # Subtle transformations
        angle = random.uniform(-8, 8)
        img = TF.rotate(img, angle, fill=0)
        img = TF.adjust_brightness(img, random.uniform(0.85, 1.15))
        img = TF.adjust_contrast(img, random.uniform(0.85, 1.15))
    
    return img

def generate_augmented_optimized(source_dir, target_dir, target_count, split_name):
    """Optimized generation with GPU acceleration"""
    
    source_images = list(source_dir.glob("*.png"))
    random.shuffle(source_images)
    
    if len(source_images) == 0:
        print(f"   ⚠️  No source images in {source_dir}")
        return 0
    
    augs_per_image = (target_count // len(source_images)) + 1
    
    print(f"   Source images: {len(source_images)}")
    print(f"   Augmentations per image: {augs_per_image}")
    
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    
    generated = 0
    start_time = time.time()
    
    # Process images in mini-batches for GPU efficiency
    mini_batch_size = 8
    
    with tqdm(total=target_count, desc=f"   {split_name}", ncols=80, unit="img") as pbar:
        for img_path in source_images:
            if generated >= target_count:
                break
            
            # Load base image
            base_img = Image.open(img_path).convert('RGB')
            
            # Generate multiple augmentations
            batch_tensors = []
            batch_count = 0
            
            for aug_idx in range(augs_per_image):
                if generated >= target_count:
                    break
                
                # Apply augmentation
                aug_type = random.randint(0, 5)
                aug_img = apply_augmentation(base_img.copy(), aug_type)
                
                # Convert to tensor
                img_tensor = to_tensor(aug_img)
                batch_tensors.append(img_tensor)
                batch_count += 1
                
                # Process mini-batch when full or done
                if len(batch_tensors) >= mini_batch_size or aug_idx == augs_per_image - 1:
                    # Stack and move to GPU
                    batch = torch.stack(batch_tensors)
                    
                    if torch.cuda.is_available():
                        batch = batch.to(device, non_blocking=True)
                    
                    # Move back to CPU and save
                    for i in range(len(batch)):
                        if generated >= target_count:
                            break
                        
                        img = to_pil(batch[i].cpu())
                        output_name = f"aug_{split_name}_{generated:05d}_no_fracture.png"
                        img.save(target_dir / output_name, optimize=True, quality=95)
                        
                        generated += 1
                        pbar.update(1)
                    
                    # Clear batch
                    batch_tensors = []
                    del batch
                    
                    # Periodic GPU cleanup
                    if torch.cuda.is_available() and generated % 100 == 0:
                        torch.cuda.empty_cache()
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    elapsed = time.time() - start_time
    speed = generated / elapsed if elapsed > 0 else 0
    
    print(f"   ✅ {generated} images in {elapsed:.1f}s ({speed:.1f} img/s)")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f"   GPU Memory: {allocated:.3f} GB")
    
    return generated

# Generate for all splits
print("🚀 Starting generation with GPU acceleration...")
print()

print("📦 TRAIN split:")
train_gen = generate_augmented_optimized(
    ORIGINAL_NO_FRACTURE / "train/no_fracture",
    AUGMENTED_TARGET / "train/no_fracture",
    target_train,
    "train"
)
print()

print("📦 VAL split:")
val_gen = generate_augmented_optimized(
    ORIGINAL_NO_FRACTURE / "val/no_fracture",
    AUGMENTED_TARGET / "val/no_fracture",
    target_val,
    "val"
)
print()

print("📦 TEST split:")
test_gen = generate_augmented_optimized(
    ORIGINAL_NO_FRACTURE / "test/no_fracture",
    AUGMENTED_TARGET / "test/no_fracture",
    target_test,
    "test"
)
print()

# Final stats
print("="*80)
print("✅ GENERATION COMPLETE!")
print("="*80)
print()
print(f"📊 Generated: {train_gen + val_gen + test_gen:,} total images")
print(f"   Train: {train_gen:,}")
print(f"   Val:   {val_gen:,}")
print(f"   Test:  {test_gen:,}")
print()

# Count final dataset
def count_files(path):
    return len(list(path.glob("*.*")))

print("📊 Final Augmented Dataset (3-class):")
for split in ["train", "val", "test"]:
    split_path = AUGMENTED_TARGET / split
    comm = count_files(split_path / "comminuted_fracture")
    simp = count_files(split_path / "simple_fracture")
    nofrac = count_files(split_path / "no_fracture")
    total = comm + simp + nofrac
    print(f"  {split.upper():5s}: {total:5,} ({comm:5,} comm + {simp:5,} simp + {nofrac:5,} nofrac)")

total = sum([count_files(AUGMENTED_TARGET / s) for s in ["train", "val", "test"]])
print(f"\n  TOTAL: {total:,} augmented images")
print()
print("🎯 Ready for 3-class training with augmented data!")
print()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("🧹 GPU memory cleared")
