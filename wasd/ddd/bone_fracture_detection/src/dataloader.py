"""
Data loading and dataset management for bone fracture detection.
Handles data organization, splitting, and balanced sampling.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import yaml
from PIL import Image
import albumentations as A

from preprocessing import BoneImagePreprocessor


class BoneFractureDataset(Dataset):
    """
    Custom dataset for bone fracture detection with medical image preprocessing.
    """
    
    def __init__(
        self, 
        image_paths: List[str], 
        labels: List[int], 
        transform: Optional[A.Compose] = None,
        preprocessor: Optional[BoneImagePreprocessor] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels (0: No Fracture, 1: Fracture)
            transform: Albumentations transform pipeline
            preprocessor: Image preprocessor instance
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.preprocessor = preprocessor or BoneImagePreprocessor()
        
        # Validate data
        assert len(image_paths) == len(labels), "Mismatch between images and labels"
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (preprocessed_image, label)
        """
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms if provided
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Use default preprocessing
            image = self.preprocessor.preprocess_single_image(image, mode='inference')
        
        label = self.labels[idx]
        
        return image, label


class BoneFractureDataManager:
    """
    Manages data organization, splitting, and loading for the bone fracture detection project.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the data manager.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
            
        self.preprocessor = BoneImagePreprocessor(config_path)
        
    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            'data': {
                'dataset_path': './data',
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15,
                'image_size': [224, 224]
            },
            'model': {
                'batch_size': 32
            },
            'hardware': {
                'num_workers': 4,
                'pin_memory': True
            }
        }
    
    def organize_dataset_from_source(self, source_path: str, target_path: str) -> None:
        """
        Organize the original dataset structure into train/val/test splits.
        
        Args:
            source_path: Path to the original dataset
            target_path: Path where organized dataset will be saved
        """
        source_path = Path(source_path)
        target_path = Path(target_path)
        
        # Create target directories
        for split in ['train', 'val', 'test']:
            for class_name in ['fracture', 'no_fracture']:
                (target_path / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # Collect all images and labels
        all_images = []
        all_labels = []
        
        # Process fracture images
        fracture_dirs = [
            source_path / "Bone Fracture" / "Bone Fracture" / "Augmented" / "Comminuted Bone Fracture",
            source_path / "Bone Fracture" / "Bone Fracture" / "Orginal" / "Comminuted Bone Fracture",
            source_path / "Bone Fracture" / "Bone Fracture" / "Augmented" / "Simple Bone Fracture",
            source_path / "Bone Fracture" / "Bone Fracture" / "Orginal" / "Simple Bone Fracture"
        ]
        
        for fracture_dir in fracture_dirs:
            if fracture_dir.exists():
                for img_file in fracture_dir.glob("*.jpg"):
                    all_images.append(str(img_file))
                    all_labels.append(1)  # Fracture
                for img_file in fracture_dir.glob("*.png"):
                    all_images.append(str(img_file))
                    all_labels.append(1)  # Fracture
        
        # For this example, we'll create synthetic "no fracture" data
        # In practice, you would have actual no-fracture X-ray images
        print(f"Found {len(all_images)} fracture images")
        print("Note: No-fracture images need to be added separately")
        
        # Split the data
        if len(all_images) > 0:
            train_images, temp_images, train_labels, temp_labels = train_test_split(
                all_images, all_labels, 
                test_size=(1 - self.config['data']['train_split']),
                stratify=all_labels,
                random_state=42
            )
            
            val_size = self.config['data']['val_split'] / (1 - self.config['data']['train_split'])
            val_images, test_images, val_labels, test_labels = train_test_split(
                temp_images, temp_labels,
                test_size=(1 - val_size),
                stratify=temp_labels,
                random_state=42
            )
            
            # Copy files to organized structure
            self._copy_files_to_split(train_images, train_labels, target_path / 'train')
            self._copy_files_to_split(val_images, val_labels, target_path / 'val')
            self._copy_files_to_split(test_images, test_labels, target_path / 'test')
            
            print(f"Data organized: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    def _copy_files_to_split(self, images: List[str], labels: List[int], split_path: Path) -> None:
        """Copy files to the appropriate split directory."""
        import shutil
        
        for img_path, label in zip(images, labels):
            src_file = Path(img_path)
            class_name = 'fracture' if label == 1 else 'no_fracture'
            dst_file = split_path / class_name / src_file.name
            
            # Avoid copying if file already exists
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
    
    def load_data_from_directory(self, data_path: str) -> Tuple[List[str], List[int]]:
        """
        Load image paths and labels from organized directory structure.
        
        Args:
            data_path: Path to data directory
            
        Returns:
            Tuple of (image_paths, labels)
        """
        data_path = Path(data_path)
        image_paths = []
        labels = []
        
        # Define class mapping
        class_mapping = {'no_fracture': 0, 'fracture': 1}
        
        for class_name, label in class_mapping.items():
            class_dir = data_path / class_name
            if class_dir.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    for img_file in class_dir.glob(ext):
                        image_paths.append(str(img_file))
                        labels.append(label)
        
        return image_paths, labels
    
    def create_balanced_sampler(self, labels: List[int]) -> WeightedRandomSampler:
        """
        Create a weighted sampler for handling class imbalance.
        
        Args:
            labels: List of labels
            
        Returns:
            WeightedRandomSampler instance
        """
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(labels), 
            y=labels
        )
        
        sample_weights = [class_weights[label] for label in labels]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def get_data_loaders(
        self, 
        data_path: str,
        use_balanced_sampling: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get data loaders for train, validation, and test sets.
        
        Args:
            data_path: Path to organized data directory
            use_balanced_sampling: Whether to use balanced sampling for training
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        data_path = Path(data_path)
        batch_size = self.config['model']['batch_size']
        num_workers = self.config['hardware']['num_workers']
        pin_memory = self.config['hardware']['pin_memory']
        
        # Load data for each split
        train_images, train_labels = self.load_data_from_directory(data_path / 'train')
        val_images, val_labels = self.load_data_from_directory(data_path / 'val')
        test_images, test_labels = self.load_data_from_directory(data_path / 'test')
        
        # Create datasets
        train_dataset = BoneFractureDataset(
            train_images, train_labels, 
            transform=self.preprocessor.get_train_transform(),
            preprocessor=self.preprocessor
        )
        
        val_dataset = BoneFractureDataset(
            val_images, val_labels,
            transform=self.preprocessor.get_val_transform(),
            preprocessor=self.preprocessor
        )
        
        test_dataset = BoneFractureDataset(
            test_images, test_labels,
            transform=self.preprocessor.get_val_transform(),
            preprocessor=self.preprocessor
        )
        
        # Create samplers
        train_sampler = None
        if use_balanced_sampling and len(train_labels) > 0:
            train_sampler = self.create_balanced_sampler(train_labels)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        print(f"Data loaders created:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")
        
        return train_loader, val_loader, test_loader
    
    def get_stratified_kfold_splits(
        self, 
        data_path: str, 
        k_folds: int = 5
    ) -> List[Tuple[DataLoader, DataLoader]]:
        """
        Get stratified k-fold cross-validation splits.
        
        Args:
            data_path: Path to data directory
            k_folds: Number of folds
            
        Returns:
            List of (train_loader, val_loader) for each fold
        """
        # Load all training data
        train_path = Path(data_path) / 'train'
        images, labels = self.load_data_from_directory(train_path)
        
        if len(images) == 0:
            raise ValueError("No training data found")
        
        # Create stratified k-fold splits
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_loaders = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
            print(f"Creating fold {fold + 1}/{k_folds}")
            
            # Split data
            fold_train_images = [images[i] for i in train_idx]
            fold_train_labels = [labels[i] for i in train_idx]
            fold_val_images = [images[i] for i in val_idx]
            fold_val_labels = [labels[i] for i in val_idx]
            
            # Create datasets
            train_dataset = BoneFractureDataset(
                fold_train_images, fold_train_labels,
                transform=self.preprocessor.get_train_transform(),
                preprocessor=self.preprocessor
            )
            
            val_dataset = BoneFractureDataset(
                fold_val_images, fold_val_labels,
                transform=self.preprocessor.get_val_transform(),
                preprocessor=self.preprocessor
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['model']['batch_size'],
                shuffle=True,
                num_workers=self.config['hardware']['num_workers'],
                pin_memory=self.config['hardware']['pin_memory']
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['model']['batch_size'],
                shuffle=False,
                num_workers=self.config['hardware']['num_workers'],
                pin_memory=self.config['hardware']['pin_memory']
            )
            
            fold_loaders.append((train_loader, val_loader))
        
        return fold_loaders


def test_data_loader():
    """Test the data loading functionality."""
    # Initialize data manager
    data_manager = BoneFractureDataManager()
    
    # Test with dummy data
    dummy_data_path = Path("./test_data")
    dummy_data_path.mkdir(exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        for class_name in ['fracture', 'no_fracture']:
            class_dir = dummy_data_path / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a dummy image
            dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            dummy_pil = Image.fromarray(dummy_img)
            dummy_pil.save(class_dir / f"dummy_{split}_{class_name}.jpg")
    
    try:
        # Test data loading
        train_loader, val_loader, test_loader = data_manager.get_data_loaders(str(dummy_data_path))
        
        print("Data loader test successful!")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test a batch
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}: Images shape {images.shape}, Labels shape {labels.shape}")
            break
            
    except Exception as e:
        print(f"Data loader test failed: {e}")
    
    # Cleanup
    import shutil
    if dummy_data_path.exists():
        shutil.rmtree(dummy_data_path)


if __name__ == "__main__":
    test_data_loader()