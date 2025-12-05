"""
Quick script to monitor training progress
"""

import time
from pathlib import Path
import json

def monitor_training():
    models_path = Path('/home/praneeth4265/wasd/ddd/bone_fracture_detection/models')
    
    print("="*80)
    print("📊 TRAINING MONITOR")
    print("="*80)
    
    # Check for saved models
    convnext_model = models_path / 'convnext_v2_best.h5'
    efficientnet_model = models_path / 'efficientnetv2s_best.h5'
    
    print("\n🔍 Checking for completed models...")
    
    if convnext_model.exists():
        size_mb = convnext_model.stat().st_size / (1024 * 1024)
        print(f"✅ ConvNeXt V2: Model saved ({size_mb:.1f} MB)")
    else:
        print(f"⏳ ConvNeXt V2: Training in progress...")
    
    if efficientnet_model.exists():
        size_mb = efficientnet_model.stat().st_size / (1024 * 1024)
        print(f"✅ EfficientNetV2-S: Model saved ({size_mb:.1f} MB)")
    else:
        print(f"⏳ EfficientNetV2-S: Not started yet")
    
    # Check for results
    print("\n📈 Recent Results:")
    result_files = sorted(models_path.glob('*_results_*.json'), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if result_files:
        for result_file in result_files[:2]:  # Show latest 2
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            print(f"\n   {result_file.name}:")
            print(f"      Model: {results['config']['model_name']}")
            print(f"      Best Val Accuracy: {results['best_val_accuracy']:.4f}")
            print(f"      Test Accuracy: {results['test_results']['accuracy']:.4f}")
            print(f"      Epochs Trained: {results['epochs_trained']}")
    else:
        print("   No results files found yet")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    monitor_training()
