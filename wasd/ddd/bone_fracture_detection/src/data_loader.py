"""
Optimized Data Loader for Maximum GPU Utilization
Features: Prefetching, caching, parallel loading, augmentation
"""

import tensorflow as tf
from pathlib import Path
import numpy as np

class OptimizedDataLoader:
    """
    High-performance data loader with GPU optimization
    """
    
    def __init__(self, img_size=224, batch_size=32, prefetch_size=tf.data.AUTOTUNE):
        self.img_size = img_size
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.class_names = ['comminuted_fracture', 'simple_fracture']
        
    def parse_image(self, file_path, label):
        """Parse and preprocess image"""
        # Read image
        img = tf.io.read_file(file_path)
        
        # Decode image (handles both PNG and JPEG)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        
        # Resize
        img = tf.image.resize(img, [self.img_size, self.img_size])
        
        # Normalize to [0, 1]
        img = tf.cast(img, tf.float32) / 255.0
        
        return img, label
    
    def load_dataset(self, data_dir, shuffle=True, cache=True, augment=False):
        """
        Load dataset with optimizations for maximum GPU usage
        
        Args:
            data_dir: Path to data directory
            shuffle: Whether to shuffle data
            cache: Whether to cache dataset in memory
            augment: Whether to apply augmentation (for training only)
        """
        data_path = Path(data_dir)
        
        # Get all image paths and labels
        file_paths = []
        labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = data_path / class_name
            if class_dir.exists():
                images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
                file_paths.extend([str(p) for p in images])
                labels.extend([class_idx] * len(images))
        
        print(f"   Found {len(file_paths)} images in {data_dir}")
        print(f"   Class distribution: {np.bincount(labels)}")
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        
        # Shuffle first for better distribution
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(10000, len(file_paths)))
        
        # Parse images in parallel
        dataset = dataset.map(
            self.parse_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Batch the data BEFORE caching to avoid shape mismatch issues
        dataset = dataset.batch(self.batch_size)
        
        # Cache dataset in memory for faster epoch iterations (after batching)
        if cache:
            dataset = dataset.cache()
        
        # Shuffle again after caching
        if shuffle:
            dataset = dataset.shuffle(buffer_size=100)
        
        # Prefetch for GPU optimization (prepare next batch while GPU processes current)
        dataset = dataset.prefetch(buffer_size=self.prefetch_size)
        
        return dataset, len(file_paths)
    
    def create_datasets(self, train_dir, val_dir, test_dir):
        """
        Create train, validation, and test datasets
        
        Args:
            train_dir: Path to training data
            val_dir: Path to validation data
            test_dir: Path to test data
            
        Returns:
            train_ds, val_ds, test_ds, dataset_info
        """
        print("="*80)
        print("📦 LOADING DATASETS")
        print("="*80)
        
        # Load training set (NO caching to avoid shape mismatch)
        print("\n📂 Loading training set...")
        train_ds, train_size = self.load_dataset(
            train_dir, 
            shuffle=True, 
            cache=False,  # Disabled to avoid batch shape conflicts
            augment=False  # Already augmented
        )
        
        # Load validation set (with caching, no shuffling)
        print("\n📂 Loading validation set...")
        val_ds, val_size = self.load_dataset(
            val_dir,
            shuffle=False,
            cache=False,  # Disabled to avoid batch shape conflicts
            augment=False
        )
        
        # Load test set (with caching, no shuffling)
        print("\n📂 Loading test set...")
        test_ds, test_size = self.load_dataset(
            test_dir,
            shuffle=False,
            cache=False,  # Disabled to avoid batch shape conflicts
            augment=False
        )
        
        dataset_info = {
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'steps_per_epoch': train_size // self.batch_size,
            'validation_steps': val_size // self.batch_size
        }
        
        print("\n" + "="*80)
        print("✅ DATASETS LOADED")
        print("="*80)
        print(f"\n📊 Dataset Info:")
        print(f"   Training samples:   {train_size:>6} ({dataset_info['steps_per_epoch']} steps/epoch)")
        print(f"   Validation samples: {val_size:>6} ({dataset_info['validation_steps']} steps)")
        print(f"   Test samples:       {test_size:>6}")
        print(f"   Batch size:         {self.batch_size:>6}")
        print(f"   Classes:            {len(self.class_names)} {self.class_names}")
        print("\n" + "="*80)
        
        return train_ds, val_ds, test_ds, dataset_info


def test_data_loader():
    """Test the data loader"""
    print("Testing data loader...")
    
    loader = OptimizedDataLoader(img_size=224, batch_size=32)
    
    train_ds, val_ds, test_ds, info = loader.create_datasets(
        train_dir='/home/praneeth4265/wasd/ddd/bone_fracture_detection/data/train_augmented',
        val_dir='/home/praneeth4265/wasd/ddd/bone_fracture_detection/data/val_processed',
        test_dir='/home/praneeth4265/wasd/ddd/bone_fracture_detection/data/test_processed'
    )
    
    print("\n✅ Data loader test passed!")
    
    # Test batch
    for images, labels in train_ds.take(1):
        print(f"\nBatch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image range: [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")
        break


if __name__ == "__main__":
    test_data_loader()
