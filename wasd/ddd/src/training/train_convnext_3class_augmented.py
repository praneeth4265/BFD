"""
Train ConvNeXt V2 on 3-Class Augmented Dataset
- Uses augmented dataset (20,530 images)
- 3 classes: comminuted_fracture, simple_fracture, no_fracture
- Transfer learning with ImageNet weights
- Mixed precision training for speed
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import sys
from pathlib import Path
import json
from datetime import datetime

# Add bone_fracture_detection/src to path
sys.path.append(str(Path(__file__).parent / 'src' / 'legacy' / 'v1_models'))

from data_loader import OptimizedDataLoader
from train_convnext_v2 import ConvNeXtV2Trainer


def setup_gpu():
    """Configure GPU for maximum safe utilization"""
    print("="*80)
    print("🔧 GPU CONFIGURATION")
    print("="*80)
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"✅ Found {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            
            tf.config.set_visible_devices(gpus, 'GPU')
            tf.config.optimizer.set_jit(False)
            print("✅ GPU enabled with memory growth")
            
        except RuntimeError as e:
            print(f"⚠️  GPU setup error: {e}")
    else:
        print("⚠️  No GPU found, using CPU")
    
    print()


def main():
    """Main training function"""
    
    # Setup GPU
    setup_gpu()
    
    # Paths
    BASE_DIR = Path(__file__).parent
    AUGMENTED_DIR = BASE_DIR / "datasets/augmented"
    
    print("="*80)
    print("🎯 TRAINING CONVNEXT V2 - 3-CLASS (AUGMENTED DATASET)")
    print("="*80)
    print()
    print(f"📁 Dataset: {AUGMENTED_DIR}")
    print(f"   Train: {AUGMENTED_DIR / 'train'}")
    print(f"   Val:   {AUGMENTED_DIR / 'val'}")
    print(f"   Test:  {AUGMENTED_DIR / 'test'}")
    print()
    
    # Verify paths exist
    if not AUGMENTED_DIR.exists():
        print(f"❌ Error: Augmented dataset not found at {AUGMENTED_DIR}")
        return
    
    # Configuration
    config = {
        'data_dir': str(AUGMENTED_DIR),
        'num_classes': 3,
        'class_names': ['comminuted_fracture', 'simple_fracture', 'no_fracture'],
        'img_size': 224,
        'batch_size': 32,
        'epochs': 30,
        'learning_rate': 1e-4,
        'model_name': 'convnext_v2_3class_augmented',
        'use_mixed_precision': True,
        'early_stopping_patience': 5,
        'reduce_lr_patience': 3,
    }
    
    print("⚙️  Configuration:")
    print(f"   Model: ConvNeXt V2 Base")
    print(f"   Classes: {config['num_classes']} ({', '.join(config['class_names'])})")
    print(f"   Image size: {config['img_size']}x{config['img_size']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Mixed precision: {config['use_mixed_precision']}")
    print()
    
    # Count images
    train_count = len(list((AUGMENTED_DIR / 'train').rglob('*.png'))) + \
                  len(list((AUGMENTED_DIR / 'train').rglob('*.jpg')))
    val_count = len(list((AUGMENTED_DIR / 'val').rglob('*.png'))) + \
                len(list((AUGMENTED_DIR / 'val').rglob('*.jpg')))
    test_count = len(list((AUGMENTED_DIR / 'test').rglob('*.png'))) + \
                 len(list((AUGMENTED_DIR / 'test').rglob('*.jpg')))
    
    print(f"📊 Dataset Statistics:")
    print(f"   Train: {train_count:,} images")
    print(f"   Val:   {val_count:,} images")
    print(f"   Test:  {test_count:,} images")
    print(f"   Total: {train_count + val_count + test_count:,} images")
    print()
    
    # Load data
    print("📦 Loading augmented dataset...")
    data_loader = OptimizedDataLoader(
        img_size=config['img_size'],
        batch_size=config['batch_size'],
        prefetch_size=tf.data.AUTOTUNE
    )
    
    # Override class names for 3-class problem
    data_loader.class_names = ['comminuted_fracture', 'no_fracture', 'simple_fracture']
    
    train_ds, val_ds, test_ds, dataset_info = data_loader.create_datasets(
        train_dir=str(AUGMENTED_DIR / 'train'),
        val_dir=str(AUGMENTED_DIR / 'val'),
        test_dir=str(AUGMENTED_DIR / 'test')
    )
    print()
    
    print("✅ Dataset loaded successfully")
    print(f"   Classes: {dataset_info['class_names']}")
    print(f"   Train: {dataset_info['train_size']} images ({dataset_info['steps_per_epoch']} steps/epoch)")
    print(f"   Val: {dataset_info['val_size']} images ({dataset_info['validation_steps']} steps/epoch)")
    print(f"   Test: {dataset_info['test_size']} images")
    print()
    
    # Create trainer
    print("🏗️  Building ConvNeXt V2 model...")
    trainer = ConvNeXtV2Trainer(
        img_size=config['img_size'],
        num_classes=config['num_classes'],
        learning_rate=config['learning_rate']
    )
    
    # Build model
    model = trainer.build_model()
    print("✅ Model built successfully")
    print()
    
    # Compile model
    trainer.compile_model()
    print()
    
    print(f"📊 Model Summary:")
    print(f"   Total params: {model.count_params():,}")
    trainable = sum([tf.size(v).numpy() for v in model.trainable_variables])
    print(f"   Trainable params: {trainable:,}")
    print()
    
    # Train
    print("="*80)
    print("🚀 STARTING TRAINING")
    print("="*80)
    print()
    
    start_time = datetime.now()
    
    history = trainer.train(train_ds, val_ds)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print()
    print("="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"⏱️  Duration: {duration}")
    print()
    
    # Evaluate on test set
    print("📊 Evaluating on test set...")
    test_results = trainer.evaluate(test_ds)
    
    print()
    print("="*80)
    print("📈 FINAL RESULTS")
    print("="*80)
    print()
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test Loss: {test_results['loss']:.4f}")
    print()
    
    # Save results
    results = {
        'model': 'ConvNeXt V2 Base',
        'dataset': 'augmented_3class',
        'num_images': train_count + val_count + test_count,
        'train_images': train_count,
        'val_images': val_count,
        'test_images': test_count,
        'num_classes': config['num_classes'],
        'class_names': config['class_names'],
        'config': config,
        'test_accuracy': float(test_results['accuracy']),
        'test_loss': float(test_results['loss']),
        'training_duration': str(duration),
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = Path(__file__).parent / 'models' / f"{config['model_name']}_results.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {results_file}")
    print()
    print("🎉 Training completed successfully!")


if __name__ == '__main__':
    main()
