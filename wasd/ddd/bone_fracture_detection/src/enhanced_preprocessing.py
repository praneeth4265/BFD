"""
Enhanced preprocessing pipeline for bone fracture X-ray images.
Includes medical-grade image enhancement, real-time augmentation, and automated batch processing.
Optimized for 224x224 standardization with 3-5x dataset augmentation.
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
import yaml
from tqdm import tqdm
import multiprocessing as mp

warnings.filterwarnings('ignore')


class EnhancedBoneImageProcessor:
    """
    Enhanced medical image processor for bone fracture detection.
    Features:
    - Image standardization (224x224, RGB, normalized)
    - Advanced enhancement (CLAHE, noise reduction, contrast adjustment)
    - Real-time augmentation (geometric + photometric)
    - Automated batch processing
    - Error handling for corrupted files
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the enhanced preprocessor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
        # Image standardization parameters
        self.target_size = (224, 224)
        self.pixel_range = [0, 1]  # Normalized range
        
        # Enhancement parameters
        self.clahe_params = {
            'clipLimit': 2.0,
            'tileGridSize': (8, 8)
        }
        
        # Noise reduction parameters
        self.denoise_params = {
            'h': 10,
            'templateWindowSize': 7,
            'searchWindowSize': 21
        }
        
        # Augmentation multiplier (3-5x dataset increase)
        self.augmentation_factor = 5
        
        # Statistics tracking
        self.processing_stats = {
            'processed_count': 0,
            'corrupted_count': 0,
            'enhancement_time': 0,
            'total_time': 0,
            'batch_sizes': []
        }
        
        # Initialize CLAHE
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_params['clipLimit'],
            tileGridSize=self.clahe_params['tileGridSize']
        )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for preprocessing."""
        return {
            'image_size': 224,
            'normalize_mean': [0.485, 0.456, 0.406],
            'normalize_std': [0.229, 0.224, 0.225],
            'clahe_clip_limit': 2.0,
            'clahe_tile_grid_size': [8, 8],
            'augmentation_probability': 0.8,
            'batch_size': 32,
            'num_workers': mp.cpu_count()
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the preprocessor."""
        logger = logging.getLogger('BoneImageProcessor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def standardize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Standardize image to 224x224 RGB format with normalized pixels.
        
        Args:
            image: Input image array
            
        Returns:
            Standardized image array
        """
        try:
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Already RGB
                rgb_image = image
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # RGBA to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 2:
                # Grayscale to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                # BGR to RGB (OpenCV default)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to 224x224
            resized_image = cv2.resize(
                rgb_image, 
                self.target_size, 
                interpolation=cv2.INTER_LANCZOS4
            )
            
            # Normalize pixel values to [0, 1] range
            normalized_image = resized_image.astype(np.float32) / 255.0
            
            return normalized_image
            
        except Exception as e:
            self.logger.error(f"Error in image standardization: {e}")
            raise
    
    def enhance_bone_visibility(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced bone visibility using advanced techniques.
        
        Args:
            image: Input image (0-1 normalized)
            
        Returns:
            Enhanced image
        """
        try:
            start_time = time.time()
            
            # Convert to uint8 for OpenCV operations
            img_uint8 = (image * 255).astype(np.uint8)
            
            # Apply CLAHE for contrast enhancement
            if len(img_uint8.shape) == 3:
                # Apply CLAHE to each channel
                enhanced_channels = []
                for i in range(3):
                    enhanced_channel = self.clahe.apply(img_uint8[:, :, i])
                    enhanced_channels.append(enhanced_channel)
                enhanced_img = np.stack(enhanced_channels, axis=2)
            else:
                enhanced_img = self.clahe.apply(img_uint8)
                enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2RGB)
            
            # Noise reduction using Non-local Means Denoising
            denoised_img = cv2.fastNlMeansDenoisingColored(
                enhanced_img,
                None,
                h=self.denoise_params['h'],
                hColor=self.denoise_params['h'],
                templateWindowSize=self.denoise_params['templateWindowSize'],
                searchWindowSize=self.denoise_params['searchWindowSize']
            )
            
            # Brightness and contrast adjustment
            alpha = 1.2  # Contrast factor
            beta = 10    # Brightness factor
            adjusted_img = cv2.convertScaleAbs(denoised_img, alpha=alpha, beta=beta)
            
            # Edge enhancement for bone structures
            # Create a sharpening kernel
            sharpening_kernel = np.array([[-1, -1, -1],
                                        [-1,  9, -1],
                                        [-1, -1, -1]])
            sharpened_img = cv2.filter2D(adjusted_img, -1, sharpening_kernel)
            
            # Blend original and sharpened images
            blend_factor = 0.3
            final_img = cv2.addWeighted(adjusted_img, 1-blend_factor, sharpened_img, blend_factor, 0)
            
            # Convert back to normalized float
            result = final_img.astype(np.float32) / 255.0
            
            # Update timing statistics
            self.processing_stats['enhancement_time'] += time.time() - start_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in bone enhancement: {e}")
            return image  # Return original if enhancement fails
    
    def create_real_time_augmentation(self, for_training: bool = True) -> A.Compose:
        """
        Create real-time augmentation pipeline for 3-5x dataset increase.
        
        Args:
            for_training: Whether augmentations are for training (more aggressive)
            
        Returns:
            Albumentations composition
        """
        if for_training:
            # Aggressive augmentation for training (3-5x increase)
            augmentations = A.Compose([
                # Geometric augmentations
                A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),  # Less likely for X-rays
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,  # 0.9-1.1x scaling
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.6
                ),
                A.Affine(
                    shear=(-10, 10),  # Shearing transformation
                    p=0.3,
                    mode=cv2.BORDER_CONSTANT,
                    cval=0
                ),
                
                # Photometric augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.6
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.4),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    p=0.3
                ),
                
                # Noise and blur
                A.GaussNoise(var_limit=(10, 50), p=0.2),
                A.GaussianBlur(blur_limit=(1, 3), p=0.2),
                
                # Advanced medical-specific augmentations
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=50,
                    p=0.2
                ),
                A.GridDistortion(p=0.2),
                
                # Normalization (ImageNet stats for transfer learning)
                A.Normalize(
                    mean=self.config['normalize_mean'],
                    std=self.config['normalize_std']
                ),
                ToTensorV2()
            ])
        else:
            # Minimal augmentation for validation/test
            augmentations = A.Compose([
                A.Normalize(
                    mean=self.config['normalize_mean'],
                    std=self.config['normalize_std']
                ),
                ToTensorV2()
            ])
        
        return augmentations
    
    def process_single_image(self, image_path: str, apply_enhancement: bool = True) -> Optional[np.ndarray]:
        """
        Process a single image with error handling for corrupted files.
        
        Args:
            image_path: Path to the image file
            apply_enhancement: Whether to apply enhancement
            
        Returns:
            Processed image array or None if corrupted
        """
        try:
            # Try multiple loading methods for robustness
            image = None
            
            # Method 1: OpenCV
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                pass
            
            # Method 2: PIL if OpenCV fails
            if image is None:
                try:
                    pil_image = Image.open(image_path).convert('RGB')
                    image = np.array(pil_image)
                except:
                    pass
            
            # Method 3: Direct numpy loading for some formats
            if image is None:
                try:
                    image = np.load(image_path) if image_path.endswith('.npy') else None
                except:
                    pass
            
            if image is None:
                self.logger.warning(f"Could not load image: {image_path}")
                self.processing_stats['corrupted_count'] += 1
                return None
            
            # Standardize image
            standardized = self.standardize_image(image)
            
            # Apply enhancement if requested
            if apply_enhancement:
                enhanced = self.enhance_bone_visibility(standardized)
            else:
                enhanced = standardized
            
            self.processing_stats['processed_count'] += 1
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            self.processing_stats['corrupted_count'] += 1
            return None
    
    def batch_process_images(
        self, 
        image_paths: List[str], 
        output_dir: str = None,
        apply_enhancement: bool = True,
        save_processed: bool = False,
        num_workers: int = None
    ) -> List[np.ndarray]:
        """
        Automated batch processing of images with parallel processing.
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save processed images (optional)
            apply_enhancement: Whether to apply enhancement
            save_processed: Whether to save processed images
            num_workers: Number of parallel workers
            
        Returns:
            List of processed image arrays
        """
        start_time = time.time()
        
        if num_workers is None:
            num_workers = min(self.config['num_workers'], len(image_paths))
        
        if output_dir and save_processed:
            os.makedirs(output_dir, exist_ok=True)
        
        processed_images = []
        
        self.logger.info(f"Starting batch processing of {len(image_paths)} images with {num_workers} workers")
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.process_single_image, path, apply_enhancement): path 
                for path in image_paths
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_path), total=len(image_paths), desc="Processing images"):
                image_path = future_to_path[future]
                try:
                    processed_image = future.result()
                    if processed_image is not None:
                        processed_images.append(processed_image)
                        
                        # Save if requested
                        if save_processed and output_dir:
                            output_path = os.path.join(
                                output_dir, 
                                f"processed_{os.path.basename(image_path)}"
                            )
                            self._save_processed_image(processed_image, output_path)
                            
                except Exception as e:
                    self.logger.error(f"Failed to process {image_path}: {e}")
        
        # Update statistics
        processing_time = time.time() - start_time
        self.processing_stats['total_time'] += processing_time
        self.processing_stats['batch_sizes'].append(len(image_paths))
        
        self.logger.info(f"Batch processing completed in {processing_time:.2f}s")
        self.logger.info(f"Processed: {len(processed_images)}, Corrupted: {self.processing_stats['corrupted_count']}")
        
        return processed_images
    
    def _save_processed_image(self, image: np.ndarray, output_path: str):
        """Save processed image to disk."""
        try:
            # Convert from [0,1] to [0,255] for saving
            image_uint8 = (image * 255).astype(np.uint8)
            cv2.imwrite(output_path, cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR))
        except Exception as e:
            self.logger.error(f"Failed to save image {output_path}: {e}")
    
    def augment_dataset_realtime(
        self, 
        images: List[np.ndarray], 
        labels: List[int],
        augmentation_factor: int = None
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Real-time dataset augmentation for 3-5x increase.
        
        Args:
            images: List of original images (normalized [0,1] float32)
            labels: List of corresponding labels
            augmentation_factor: Multiplier for dataset size
            
        Returns:
            Augmented images and labels (as uint8 [0,255])
        """
        if augmentation_factor is None:
            augmentation_factor = self.augmentation_factor
        
        # Create augmentation WITHOUT normalization for saving
        augment_transform = A.Compose([
            # Geometric augmentations
            A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.6
            ),
            A.Affine(
                shear=(-10, 10),
                p=0.3,
                mode=cv2.BORDER_CONSTANT,
                cval=0
            ),
            # Photometric augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.6
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.GaussianBlur(blur_limit=(1, 3), p=0.2),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.2
            ),
            A.GridDistortion(p=0.2),
        ])
        
        augmented_images = []
        augmented_labels = []
        
        self.logger.info(f"Augmenting dataset by {augmentation_factor}x")
        
        for img, label in tqdm(zip(images, labels), total=len(images), desc="Augmenting dataset"):
            # Convert from [0,1] to [0,255] for augmentation
            if img.dtype == np.float32 or img.dtype == np.float64:
                img_uint8 = (img * 255).astype(np.uint8)
            else:
                img_uint8 = img
            
            # Include original (as uint8)
            augmented_images.append(img_uint8)
            augmented_labels.append(label)
            
            # Generate augmented versions
            for _ in range(augmentation_factor - 1):
                try:
                    # Apply augmentation (returns uint8)
                    augmented = augment_transform(image=img_uint8)['image']
                    augmented_images.append(augmented)
                    augmented_labels.append(label)
                    
                except Exception as e:
                    self.logger.warning(f"Augmentation failed for image: {e}")
                    continue
        
        self.logger.info(f"Dataset augmented from {len(images)} to {len(augmented_images)} images")
        return augmented_images, augmented_labels
    
    def create_batch_processing_script(self, script_path: str = "batch_process.py"):
        """
        Create automated batch processing script using OpenCV.
        
        Args:
            script_path: Path for the generated script
        """
        script_content = '''#!/usr/bin/env python3
"""
Automated batch processing script for bone fracture X-ray images.
Generated by EnhancedBoneImageProcessor.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from enhanced_preprocessing import EnhancedBoneImageProcessor


def main():
    parser = argparse.ArgumentParser(description='Batch process bone X-ray images')
    parser.add_argument('input_dir', help='Input directory containing images')
    parser.add_argument('output_dir', help='Output directory for processed images')
    parser.add_argument('--enhancement', action='store_true', default=True,
                       help='Apply image enhancement (default: True)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = EnhancedBoneImageProcessor(config_path=args.config)
    
    # Get all image files
    input_path = Path(args.input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    image_paths = [
        str(p) for p in input_path.rglob('*') 
        if p.suffix.lower() in image_extensions
    ]
    
    print(f"Found {len(image_paths)} images to process")
    
    # Process images
    processor.batch_process_images(
        image_paths=image_paths,
        output_dir=args.output_dir,
        apply_enhancement=args.enhancement,
        save_processed=True,
        num_workers=args.workers
    )
    
    # Print statistics
    stats = processor.processing_stats
    print(f"\\nProcessing completed:")
    print(f"  Processed: {stats['processed_count']}")
    print(f"  Corrupted: {stats['corrupted_count']}")
    print(f"  Total time: {stats['total_time']:.2f}s")
    print(f"  Enhancement time: {stats['enhancement_time']:.2f}s")


if __name__ == "__main__":
    main()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        self.logger.info(f"Batch processing script created: {script_path}")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        stats = self.processing_stats.copy()
        
        if stats['processed_count'] > 0:
            stats['avg_enhancement_time'] = stats['enhancement_time'] / stats['processed_count']
            stats['success_rate'] = stats['processed_count'] / (stats['processed_count'] + stats['corrupted_count'])
        
        if stats['batch_sizes']:
            stats['avg_batch_size'] = np.mean(stats['batch_sizes'])
            stats['max_batch_size'] = max(stats['batch_sizes'])
        
        return stats
    
    def reset_statistics(self):
        """Reset processing statistics."""
        self.processing_stats = {
            'processed_count': 0,
            'corrupted_count': 0,
            'enhancement_time': 0,
            'total_time': 0,
            'batch_sizes': []
        }


def create_sample_config(config_path: str = "preprocessing_config.yaml"):
    """Create a sample configuration file."""
    config = {
        'image_size': 224,
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225],
        'clahe_clip_limit': 2.0,
        'clahe_tile_grid_size': [8, 8],
        'augmentation_probability': 0.8,
        'batch_size': 32,
        'num_workers': mp.cpu_count(),
        'enhancement': {
            'noise_reduction': True,
            'contrast_enhancement': True,
            'edge_sharpening': True,
            'brightness_adjustment': True
        },
        'augmentation': {
            'rotation_limit': 15,
            'scale_limit': 0.1,
            'brightness_limit': 0.2,
            'contrast_limit': 0.2,
            'factor': 5
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Sample configuration created: {config_path}")


def test_enhanced_processor():
    """Test the enhanced preprocessor."""
    processor = EnhancedBoneImageProcessor()
    
    # Create a test image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    print("Testing image standardization...")
    standardized = processor.standardize_image(test_image)
    print(f"Original shape: {test_image.shape}, Standardized shape: {standardized.shape}")
    print(f"Pixel range: [{standardized.min():.3f}, {standardized.max():.3f}]")
    
    print("Testing bone enhancement...")
    enhanced = processor.enhance_bone_visibility(standardized)
    print(f"Enhanced shape: {enhanced.shape}")
    
    print("Testing augmentation pipeline...")
    transform = processor.create_real_time_augmentation(for_training=True)
    augmented = transform(image=enhanced)
    print(f"Augmented type: {type(augmented['image'])}")
    
    print("Creating batch processing script...")
    processor.create_batch_processing_script()
    
    print("Enhanced preprocessor test completed successfully!")
    
    # Print statistics
    stats = processor.get_processing_statistics()
    print(f"Processing statistics: {stats}")


if __name__ == "__main__":
    # Create sample configuration
    create_sample_config()
    
    # Test the processor
    test_enhanced_processor()