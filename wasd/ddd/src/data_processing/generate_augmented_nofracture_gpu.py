#!/usr/bin/env python3
"""
Generate Augmented No Fracture Images Using GPU (Safe Memory Management)
- Uses PyTorch with CUDA for efficient GPU utilization
- Safe batch processing with memory monitoring
- Generates augmented no_fracture images to match comminuted & simple counts
- Creates balanced 3-class augmented dataset
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import random
import time
from tqdm import tqdm
import gc

print("="*80)
print("🎨 GENERATING AUGMENTED NO_FRACTURE IMAGES WITH GPU (SAFE MODE)")
print("="*80)
print()

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Check GPU and set memory limits
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   Total Memory: {total_memory:.2f} GB")
    
    # Clear any existing GPU cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Set memory fraction to use (80% for safety)
    torch.cuda.set_per_process_memory_fraction(0.8, 0)
    print(f"   Using: {total_memory * 0.8:.2f} GB (80% for safety)")
else:
    print("   ⚠️  GPU not available, using CPU (slower)")
print()

# Paths
BASE_PATH = Path("/home/praneeth4265/wasd/ddd")
ORIGINAL_NO_FRACTURE = BASE_PATH / "datasets/original"
AUGMENTED_TARGET = BASE_PATH / "datasets/augmented"

# Calculate target counts (match average of comminuted and simple)
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

print(f"  Comminuted: Train {comm_train}, Val {comm_val}, Test {comm_test}")
print(f"  Simple:     Train {simp_train}, Val {simp_val}, Test {simp_test}")
print()
print(f"🎯 Target no_fracture counts:")
print(f"  Train: {target_train}")
print(f"  Val:   {target_val}")
print(f"  Test:  {target_test}")
print()

# Define augmentation transforms (GPU-compatible)
augmentation_transforms = [
    transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ]),
    transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(p=0.8),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.Resize(256),
    ]),
    transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    ]),
    transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.RandomRotation(12),
        transforms.RandomAffine(degrees=0, shear=5),
    ]),
    transforms.Compose([
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        transforms.RandomHorizontalFlip(p=0.3),
    ]),
    transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
    ]),
]

# Custom dataset class for batch processing
class ImageAugmentationDataset(Dataset):
    def __init__(self, image_paths, num_augmentations_per_image, transforms_list):
        self.image_paths = image_paths
        self.num_augmentations = num_augmentations_per_image
        self.transforms_list = transforms_list
        
    def __len__(self):
        return len(self.image_paths) * self.num_augmentations
    
    def __getitem__(self, idx):
        img_idx = idx // self.num_augmentations
        aug_idx = idx % self.num_augmentations
        
        img_path = self.image_paths[img_idx]
        img = Image.open(img_path).convert('RGB')
        
        # Apply random transformation
        transform = random.choice(self.transforms_list)
        augmented_img = transform(img)
        
        return augmented_img, img_idx, aug_idx

def generate_augmented_batch(source_dir, target_dir, target_count, split_name, batch_size=16):
    """Generate augmented images using GPU batch processing with safe memory management"""
    
    source_images = list(source_dir.glob("*.png"))
    random.shuffle(source_images)
    
    if len(source_images) == 0:
        print(f"   ⚠️  No source images found in {source_dir}")
        return 0
    
    # Calculate augmentations per image
    augs_per_image = (target_count // len(source_images)) + 1
    total_to_generate = min(len(source_images) * augs_per_image, target_count)
    
    print(f"   Source images: {len(source_images)}")
    print(f"   Augmentations per image: {augs_per_image}")
    print(f"   Total to generate: {total_to_generate}")
    print(f"   Batch size: {batch_size} (optimized for memory)")
    
    # Create dataset and dataloader with optimized settings
    dataset = ImageAugmentationDataset(source_images, augs_per_image, augmentation_transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,  # Reduced for safety
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=2
    )
    
    generated = 0
    start_time = time.time()
    
    # Convert to tensor transform
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    
    with tqdm(total=total_to_generate, desc=f"   Generating {split_name}", ncols=80) as pbar:
        for batch_idx, (batch_imgs, img_indices, aug_indices) in enumerate(dataloader):
            if generated >= target_count:
                break
            
            # Move to GPU if available
            if torch.cuda.is_available():
                batch_imgs = batch_imgs.to(device, non_blocking=True)
            
            # Process each image in batch
            for i in range(len(batch_imgs)):
                if generated >= target_count:
                    break
                
                # Convert back to PIL for saving
                img = to_pil(batch_imgs[i].cpu())
                
                # Save
                output_name = f"aug_{split_name}_{generated:05d}_no_fracture.png"
                img.save(target_dir / output_name)
                
                generated += 1
                pbar.update(1)
            
            # Clear GPU cache every 10 batches for safety
            if torch.cuda.is_available() and batch_idx % 10 == 0:
                torch.cuda.empty_cache()
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    elapsed = time.time() - start_time
    print(f"   ✅ Generated {generated} images in {elapsed:.2f}s ({generated/elapsed:.1f} imgs/sec)")
    
    return generated

# Generate for each split
print("🚀 Starting augmentation generation...")
print()

print("📦 Processing TRAIN split...")
train_generated = generate_augmented_batch(
    ORIGINAL_NO_FRACTURE / "train/no_fracture",
    AUGMENTED_TARGET / "train/no_fracture",
    target_train,
    "train",
    batch_size=16  # Safe batch size
)
print()

print("📦 Processing VAL split...")
val_generated = generate_augmented_batch(
    ORIGINAL_NO_FRACTURE / "val/no_fracture",
    AUGMENTED_TARGET / "val/no_fracture",
    target_val,
    "val",
    batch_size=16  # Safe batch size
)
print()

print("📦 Processing TEST split...")
test_generated = generate_augmented_batch(
    ORIGINAL_NO_FRACTURE / "test/no_fracture",
    AUGMENTED_TARGET / "test/no_fracture",
    target_test,
    "test",
    batch_size=16  # Safe batch size
)
print()

# Final GPU cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f"🧹 GPU Memory cleaned")
    print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"   Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    print()

# Final verification
print("="*80)
print("✅ AUGMENTATION GENERATION COMPLETE!")
print("="*80)
print()
print("📊 Generated:")
print(f"   Train: {train_generated} images")
print(f"   Val:   {val_generated} images")
print(f"   Test:  {test_generated} images")
print(f"   Total: {train_generated + val_generated + test_generated} images")
print()

# Count all augmented images
def count_files(path):
    return len(list(path.glob("*.*")))

print("📊 Final Augmented Dataset:")
for split in ["train", "val", "test"]:
    split_path = AUGMENTED_TARGET / split
    comm = count_files(split_path / "comminuted_fracture")
    simp = count_files(split_path / "simple_fracture")
    nofrac = count_files(split_path / "no_fracture")
    total = comm + simp + nofrac
    print(f"  {split:5s}: {comm:5d} comminuted, {simp:5d} simple, {nofrac:5d} no_fracture = {total:6d} total")

total_aug = sum([count_files(AUGMENTED_TARGET / split) for split in ["train", "val", "test"]])
print(f"\n  Total augmented: {total_aug:,} images")
print()
print("🎯 Ready for 3-class training with augmented data!")
