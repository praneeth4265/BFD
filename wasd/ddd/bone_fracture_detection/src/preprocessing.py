"""
Medical-grade image preprocessing pipeline for bone fracture detection.
Includes CLAHE, advanced augmentations, edge enhancement, and normalization.
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional, Dict, Any
import yaml
from pathlib import Path


class BoneImagePreprocessor:
    """
Advanced preprocessing pipeline for bone fracture X-ray images.
Includes medical-grade image enhancement, normalization, and augmentation.
Enhanced with automated batch processing and real-time augmentation.
"""

import cv2
import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional, List, Union, Dict, Any
import logging
import os
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')
    
    def __init__(self, config_path: str = None):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
            
        self.image_size = tuple(self.config['data']['image_size'])
        self.mean = self.config['data']['normalize_mean']
        self.std = self.config['data']['normalize_std']
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration if no config file provided."""
        return {
            'data': {
                'image_size': [224, 224],
                'normalize_mean': [0.485, 0.456, 0.406],
                'normalize_std': [0.229, 0.224, 0.225],
                'augmentation': {
                    'rotation_limit': 20,
                    'brightness_limit': 0.2,
                    'contrast_limit': 0.2,
                    'blur_limit': 3,
                    'noise_limit': 0.05,
                    'clahe_clip_limit': 2.0
                }
            }
        }
    
    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        Essential for medical X-ray image enhancement.
        
        Args:
            image: Input grayscale or RGB image
            clip_limit: Threshold for contrast limiting
            
        Returns:
            CLAHE enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space for better results
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            
            # Merge back
            lab[:, :, 0] = l_channel
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            
        return enhanced
    
    def enhance_bone_edges(self, image: np.ndarray, method: str = 'canny') -> np.ndarray:
        """
        Enhance bone edges using various edge detection methods.
        
        Args:
            image: Input image
            method: Edge detection method ('canny', 'sobel', 'laplacian')
            
        Returns:
            Edge-enhanced image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        if method == 'canny':
            edges = cv2.Canny(gray, 50, 150)
        elif method == 'sobel':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = np.uint8(edges / edges.max() * 255)
        elif method == 'laplacian':
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.uint8(np.absolute(edges))
        else:
            raise ValueError(f"Unknown edge detection method: {method}")
            
        # Combine original with edges
        if len(image.shape) == 3:
            edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            enhanced = cv2.addWeighted(image, 0.8, edges_3ch, 0.2, 0)
        else:
            enhanced = cv2.addWeighted(image, 0.8, edges, 0.2, 0)
            
        return enhanced
    
    def get_train_transform(self) -> A.Compose:
        """
        Get training data augmentation pipeline with medical-specific augmentations.
        
        Returns:
            Albumentations composition for training
        """
        aug_config = self.config['data']['augmentation']
        
        return A.Compose([
            # Resize first
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            
            # Medical-specific enhancements
            A.CLAHE(clip_limit=aug_config['clahe_clip_limit'], p=0.7),
            
            # Geometric augmentations (conservative for medical images)
            A.Rotate(limit=aug_config['rotation_limit'], p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=aug_config['rotation_limit'], 
                p=0.5
            ),
            
            # Intensity augmentations
            A.RandomBrightnessContrast(
                brightness_limit=aug_config['brightness_limit'],
                contrast_limit=aug_config['contrast_limit'],
                p=0.6
            ),
            A.GaussianBlur(blur_limit=aug_config['blur_limit'], p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            
            # Sharpening for better bone structure visibility
            A.UnsharpMask(blur_limit=3, sigma_limit=1.0, alpha=0.2, p=0.4),
            
            # Final normalization
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
    
    def get_val_transform(self) -> A.Compose:
        """
        Get validation/test data preprocessing pipeline.
        
        Returns:
            Albumentations composition for validation/testing
        """
        return A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.CLAHE(clip_limit=2.0, p=1.0),  # Always apply CLAHE for consistency
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
    
    def get_inference_transform(self) -> A.Compose:
        """
        Get inference preprocessing pipeline for deployment.
        
        Returns:
            Albumentations composition for inference
        """
        return A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.CLAHE(clip_limit=2.0, p=1.0),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
    
    def preprocess_single_image(self, image: np.ndarray, mode: str = 'inference') -> np.ndarray:
        """
        Preprocess a single image for inference.
        
        Args:
            image: Input image as numpy array
            mode: Processing mode ('train', 'val', 'inference')
            
        Returns:
            Preprocessed image tensor
        """
        if mode == 'train':
            transform = self.get_train_transform()
        elif mode == 'val':
            transform = self.get_val_transform()
        else:
            transform = self.get_inference_transform()
            
        # Apply CLAHE enhancement before augmentation for better results
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = self.apply_clahe(image)
        
        # Apply transformations
        transformed = transform(image=image)
        return transformed['image']
    
    def compute_dataset_stats(self, image_paths: list) -> Tuple[list, list]:
        """
        Compute mean and std statistics for the dataset.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Tuple of (mean, std) values for normalization
        """
        pixel_sum = np.zeros(3)
        pixel_squared_sum = np.zeros(3)
        total_pixels = 0
        
        for img_path in image_paths:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.image_size)
            
            # Apply CLAHE for consistent statistics
            image = self.apply_clahe(image)
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Flatten and accumulate statistics
            image_flat = image.reshape(-1, 3)
            pixel_sum += np.sum(image_flat, axis=0)
            pixel_squared_sum += np.sum(image_flat**2, axis=0)
            total_pixels += image_flat.shape[0]
        
        mean = pixel_sum / total_pixels
        std = np.sqrt((pixel_squared_sum / total_pixels) - (mean**2))
        
        return mean.tolist(), std.tolist()


def test_preprocessor():
    """Test the preprocessor with a sample image."""
    # Create a dummy X-ray-like image
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Initialize preprocessor
    preprocessor = BoneImagePreprocessor()
    
    # Test different preprocessing modes
    train_tensor = preprocessor.preprocess_single_image(dummy_image, mode='train')
    val_tensor = preprocessor.preprocess_single_image(dummy_image, mode='val')
    inference_tensor = preprocessor.preprocess_single_image(dummy_image, mode='inference')
    
    print(f"Train tensor shape: {train_tensor.shape}")
    print(f"Val tensor shape: {val_tensor.shape}")
    print(f"Inference tensor shape: {inference_tensor.shape}")
    
    # Test CLAHE enhancement
    enhanced = preprocessor.apply_clahe(dummy_image)
    print(f"CLAHE enhanced image shape: {enhanced.shape}")
    
    # Test edge enhancement
    edge_enhanced = preprocessor.enhance_bone_edges(dummy_image)
    print(f"Edge enhanced image shape: {edge_enhanced.shape}")
    
    print("Preprocessor tests passed!")


if __name__ == "__main__":
    test_preprocessor()