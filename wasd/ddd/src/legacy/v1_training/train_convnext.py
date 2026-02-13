"""
Main Training Script - ConvNeXt V2
High-performance training with maximum GPU utilization
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
# Enable GPU with TensorFlow 2.15
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader import OptimizedDataLoader
from train_convnext_v2 import ConvNeXtV2Trainer


def setup_gpu():
    """Configure GPU for maximum safe utilization"""
    print("="*80)
    print("🔧 GPU CONFIGURATION")
    print("="*80)
    
    # Get GPU info
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth to prevent OOM
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"✅ Found {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            
            # Set visible devices
            tf.config.set_visible_devices(gpus, 'GPU')
            
            # Disable XLA (requires CUDA toolkit which may not be available)
            # GPU will still be used with mixed precision for speed
            tf.config.optimizer.set_jit(False)
            print("✅ GPU enabled (XLA disabled - using eager mode with mixed precision)")
            
        except RuntimeError as e:
            print(f"⚠️  GPU setup error: {e}")
    else:
        print("⚠️  No GPU found, using CPU")
    
    print("="*80)


def main():
    """Main training pipeline for ConvNeXt V2"""
    
    # Setup
    setup_gpu()
    
    # Paths
    base_path = Path('/home/praneeth4265/wasd/ddd/bone_fracture_detection')
    data_path = base_path / 'data'
    models_path = base_path / 'models'
    logs_path = base_path / 'logs'
    
    # Create directories
    models_path.mkdir(exist_ok=True)
    logs_path.mkdir(exist_ok=True)
    
    # Training configuration - Optimized for RTX 4060 (6GB)
    config = {
        'model_name': 'ConvNeXt_V2',
        'img_size': 224,
        'batch_size': 16,  # Reduced for 6GB GPU memory
        'learning_rate': 1e-4,
        'epochs': 50,
        'num_classes': 2
    }
    
    print("\n" + "="*80)
    print("⚙️  TRAINING CONFIGURATION - CONVNEXT V2")
    print("="*80)
    for key, value in config.items():
        print(f"   {key}: {value}")
    print("="*80)
    
    # Step 1: Load Data
    print("\n" + "="*80)
    print("📦 STEP 1/4: LOADING DATA")
    print("="*80)
    
    data_loader = OptimizedDataLoader(
        img_size=config['img_size'],
        batch_size=config['batch_size'],
        prefetch_size=tf.data.AUTOTUNE
    )
    
    train_ds, val_ds, test_ds, dataset_info = data_loader.create_datasets(
        train_dir=str(data_path / 'train_augmented'),
        val_dir=str(data_path / 'val_processed'),
        test_dir=str(data_path / 'test_processed')
    )
    
    # Step 2: Build Model
    print("\n" + "="*80)
    print("🏗️  STEP 2/4: BUILDING MODEL")
    print("="*80)
    
    trainer = ConvNeXtV2Trainer(
        img_size=config['img_size'],
        num_classes=config['num_classes'],
        learning_rate=config['learning_rate']
    )
    
    model = trainer.build_model(freeze_base=False)
    trainer.compile_model()
    
    # Step 3: Create Callbacks
    print("\n" + "="*80)
    print("📋 STEP 3/4: SETTING UP CALLBACKS")
    print("="*80)
    
    callbacks = trainer.create_callbacks(
        model_dir=models_path,
        logs_dir=logs_path
    )
    
    print(f"   ✅ ModelCheckpoint: Save best model to {models_path}")
    print(f"   ✅ EarlyStopping: Patience=10 epochs")
    print(f"   ✅ ReduceLROnPlateau: Factor=0.5, Patience=5")
    print(f"   ✅ TensorBoard: Logs to {logs_path}")
    
    # Step 4: Train Model
    print("\n" + "="*80)
    print("🚀 STEP 4/4: TRAINING MODEL")
    print("="*80)
    
    print(f"\n   Training on: {dataset_info['train_size']:,} images")
    print(f"   Validating on: {dataset_info['val_size']:,} images")
    print(f"   Steps per epoch: {dataset_info['steps_per_epoch']}")
    print(f"   Max epochs: {config['epochs']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"\n   🔥 Starting training...\n")
    
    history = trainer.train(
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=config['epochs'],
        callbacks=callbacks
    )
    
    # Step 5: Evaluate on Test Set
    print("\n" + "="*80)
    print("📊 FINAL EVALUATION ON TEST SET")
    print("="*80)
    
    test_results = trainer.evaluate(test_ds)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save history
    history_path = models_path / f'convnext_v2_history_{timestamp}.json'
    trainer.save_history(history_path)
    
    # Save test results
    results_path = models_path / f'convnext_v2_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump({
            'config': config,
            'test_results': test_results,
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'epochs_trained': len(history.history['accuracy'])
        }, f, indent=2)
    
    print(f"\n✅ Results saved to {results_path}")
    
    # Final Summary
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE - CONVNEXT V2")
    print("="*80)
    print(f"\n📊 Final Metrics:")
    print(f"   Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"   Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"   Test Precision: {test_results['precision']:.4f}")
    print(f"   Test Recall: {test_results['recall']:.4f}")
    print(f"   Test AUC: {test_results['auc']:.4f}")
    print(f"\n📁 Saved Files:")
    print(f"   Best Model: {models_path / 'convnext_v2_best.h5'}")
    print(f"   History: {history_path}")
    print(f"   Results: {results_path}")
    print(f"   TensorBoard Logs: {logs_path}")
    print("\n" + "="*80)
    
    return trainer, history, test_results


if __name__ == "__main__":
    try:
        trainer, history, results = main()
        print("\n✅ SUCCESS! ConvNeXt V2 training completed successfully!")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
