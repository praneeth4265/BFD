#!/usr/bin/env python3
"""
Run preprocessing and data augmentation on the bone fracture dataset.
Processes both Simple and Comminuted fracture images with enhancement and augmentation.
"""

import os
import sys
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from enhanced_preprocessing import EnhancedBoneImageProcessor


def process_bone_fracture_dataset():
    """Process the complete bone fracture dataset."""
    
    print("=" * 80)
    print("🦴 BONE FRACTURE DATASET PREPROCESSING & AUGMENTATION")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize processor
    config_path = "preprocessing_config.yaml"
    processor = EnhancedBoneImageProcessor(config_path=config_path)
    
    print(f"✅ Processor initialized")
    print(f"   Target size: {processor.target_size}")
    print(f"   Augmentation factor: {processor.augmentation_factor}x")
    
    # Define dataset paths
    dataset_base = "/home/praneeth4265/wasd/ddd/Bone Fracture X-ray Dataset Simple vs. Comminuted Fractures"
    
    # Original images (note: spelled "Orginal" in the dataset)
    simple_orig_path = os.path.join(dataset_base, "Bone Fracture", "Bone Fracture", "Orginal", "Simple Bone Fracture")
    comminuted_orig_path = os.path.join(dataset_base, "Bone Fracture", "Bone Fracture", "Orginal", "Comminuted Bone Fracture")
    
    # Augmented images (if you want to process those too)
    simple_aug_path = os.path.join(dataset_base, "Bone Fracture", "Bone Fracture", "Augmented", "Simple Bone Fracture")
    comminuted_aug_path = os.path.join(dataset_base, "Bone Fracture", "Bone Fracture", "Augmented", "Comminuted Bone Fracture")
    
    # Output directories
    output_base = "processed_dataset"
    processed_train_dir = os.path.join(output_base, "train")
    processed_val_dir = os.path.join(output_base, "validation")
    augmented_output_dir = os.path.join(output_base, "augmented")
    
    # Create output directories
    for dir_path in [processed_train_dir, processed_val_dir, augmented_output_dir]:
        os.makedirs(os.path.join(dir_path, "simple_fracture"), exist_ok=True)
        os.makedirs(os.path.join(dir_path, "comminuted_fracture"), exist_ok=True)
    
    print(f"\n📁 Dataset structure:")
    print(f"   Simple (Original): {simple_orig_path}")
    print(f"   Comminuted (Original): {comminuted_orig_path}")
    print(f"   Output: {output_base}")
    
    # Collect all image files
    results = {
        'simple_fracture': {'original': 0, 'processed': 0, 'augmented': 0, 'corrupted': 0},
        'comminuted_fracture': {'original': 0, 'processed': 0, 'augmented': 0, 'corrupted': 0},
        'total_time': 0,
        'processing_stats': {}
    }
    
    # Process Simple Bone Fractures
    print(f"\n{'='*80}")
    print("📸 PROCESSING SIMPLE BONE FRACTURES")
    print(f"{'='*80}")
    
    if os.path.exists(simple_orig_path):
        simple_files = list(Path(simple_orig_path).glob("*.jpg")) + \
                      list(Path(simple_orig_path).glob("*.png")) + \
                      list(Path(simple_orig_path).glob("*.jpeg"))
        
        results['simple_fracture']['original'] = len(simple_files)
        print(f"Found {len(simple_files)} simple fracture images")
        
        if simple_files:
            # Process with enhancement
            print(f"\n🔧 Stage 1: Preprocessing with enhancement...")
            simple_processed = processor.batch_process_images(
                image_paths=[str(f) for f in simple_files],
                output_dir=os.path.join(processed_train_dir, "simple_fracture"),
                apply_enhancement=True,
                save_processed=True,
                num_workers=4
            )
            results['simple_fracture']['processed'] = len(simple_processed)
            print(f"✅ Processed {len(simple_processed)} images")
            
            # Apply augmentation for 5x increase
            print(f"\n🔄 Stage 2: Real-time augmentation (5x increase)...")
            labels = [0] * len(simple_processed)  # Label 0 for simple fracture
            aug_images, aug_labels = processor.augment_dataset_realtime(
                simple_processed, labels, augmentation_factor=5
            )
            results['simple_fracture']['augmented'] = len(aug_images)
            
            # Save augmented images
            print(f"💾 Saving augmented images...")
            aug_output_dir = os.path.join(augmented_output_dir, "simple_fracture")
            for i, img in enumerate(aug_images):
                output_path = os.path.join(aug_output_dir, f"aug_simple_{i:05d}.png")
                processor._save_processed_image(img, output_path)
            
            print(f"✅ Created {len(aug_images)} augmented images (from {len(simple_processed)})")
            
    else:
        print(f"⚠️ Simple fracture directory not found: {simple_orig_path}")
    
    # Process Comminuted Bone Fractures
    print(f"\n{'='*80}")
    print("📸 PROCESSING COMMINUTED BONE FRACTURES")
    print(f"{'='*80}")
    
    if os.path.exists(comminuted_orig_path):
        comminuted_files = list(Path(comminuted_orig_path).glob("*.jpg")) + \
                          list(Path(comminuted_orig_path).glob("*.png")) + \
                          list(Path(comminuted_orig_path).glob("*.jpeg"))
        
        results['comminuted_fracture']['original'] = len(comminuted_files)
        print(f"Found {len(comminuted_files)} comminuted fracture images")
        
        if comminuted_files:
            # Process with enhancement
            print(f"\n🔧 Stage 1: Preprocessing with enhancement...")
            comminuted_processed = processor.batch_process_images(
                image_paths=[str(f) for f in comminuted_files],
                output_dir=os.path.join(processed_train_dir, "comminuted_fracture"),
                apply_enhancement=True,
                save_processed=True,
                num_workers=4
            )
            results['comminuted_fracture']['processed'] = len(comminuted_processed)
            print(f"✅ Processed {len(comminuted_processed)} images")
            
            # Apply augmentation for 5x increase
            print(f"\n🔄 Stage 2: Real-time augmentation (5x increase)...")
            labels = [1] * len(comminuted_processed)  # Label 1 for comminuted fracture
            aug_images, aug_labels = processor.augment_dataset_realtime(
                comminuted_processed, labels, augmentation_factor=5
            )
            results['comminuted_fracture']['augmented'] = len(aug_images)
            
            # Save augmented images
            print(f"💾 Saving augmented images...")
            aug_output_dir = os.path.join(augmented_output_dir, "comminuted_fracture")
            for i, img in enumerate(aug_images):
                output_path = os.path.join(aug_output_dir, f"aug_comminuted_{i:05d}.png")
                processor._save_processed_image(img, output_path)
            
            print(f"✅ Created {len(aug_images)} augmented images (from {len(comminuted_processed)})")
            
    else:
        print(f"⚠️ Comminuted fracture directory not found: {comminuted_orig_path}")
    
    # Get final statistics
    stats = processor.get_processing_statistics()
    results['processing_stats'] = stats
    results['corrupted'] = stats.get('corrupted_count', 0)
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print("📊 PROCESSING SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n🩺 Simple Bone Fractures:")
    print(f"   Original images: {results['simple_fracture']['original']}")
    print(f"   Processed images: {results['simple_fracture']['processed']}")
    print(f"   Augmented images: {results['simple_fracture']['augmented']}")
    print(f"   Augmentation factor: {results['simple_fracture']['augmented'] / max(1, results['simple_fracture']['processed']):.1f}x")
    
    print(f"\n🦴 Comminuted Bone Fractures:")
    print(f"   Original images: {results['comminuted_fracture']['original']}")
    print(f"   Processed images: {results['comminuted_fracture']['processed']}")
    print(f"   Augmented images: {results['comminuted_fracture']['augmented']}")
    print(f"   Augmentation factor: {results['comminuted_fracture']['augmented'] / max(1, results['comminuted_fracture']['processed']):.1f}x")
    
    total_original = results['simple_fracture']['original'] + results['comminuted_fracture']['original']
    total_processed = results['simple_fracture']['processed'] + results['comminuted_fracture']['processed']
    total_augmented = results['simple_fracture']['augmented'] + results['comminuted_fracture']['augmented']
    
    print(f"\n📈 Total Statistics:")
    print(f"   Total original images: {total_original}")
    print(f"   Total processed images: {total_processed}")
    print(f"   Total augmented images: {total_augmented}")
    print(f"   Overall augmentation: {total_augmented / max(1, total_processed):.1f}x")
    print(f"   Corrupted files: {results['corrupted']}")
    print(f"   Success rate: {100 * total_processed / max(1, total_original):.1f}%")
    
    if stats.get('total_time', 0) > 0:
        print(f"\n⏱️ Performance:")
        print(f"   Total processing time: {stats['total_time']:.2f}s")
        print(f"   Enhancement time: {stats['enhancement_time']:.2f}s")
        print(f"   Average time per image: {stats['total_time'] / max(1, total_processed):.3f}s")
    
    print(f"\n📁 Output Structure:")
    print(f"   Processed (train): {processed_train_dir}")
    print(f"   Augmented: {augmented_output_dir}")
    print(f"      ├── simple_fracture/")
    print(f"      └── comminuted_fracture/")
    
    # Save results to JSON
    results_file = os.path.join(output_base, "processing_results.json")
    with open(results_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                    for k, v in value.items()}
            else:
                json_results[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    print(f"\n{'='*80}")
    print("🎉 PREPROCESSING & AUGMENTATION COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


if __name__ == "__main__":
    try:
        results = process_bone_fracture_dataset()
        
        print(f"\n✅ All processing completed successfully!")
        print(f"🚀 Dataset is ready for training!")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
