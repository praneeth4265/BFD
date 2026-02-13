"""
Simple CNN Training - Fast and Reliable
Train a custom CNN that works without GPU compilation issues
Target: Complete training in ~2 hours
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU only to avoid CUDA issues

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
from datetime import datetime

print("="*80)
print("🚀 SIMPLE CNN TRAINING - FAST & RELIABLE")
print("="*80)
print(f"TensorFlow version: {tf.__version__}")
print(f"Training on: CPU (avoiding GPU CUDA issues)")
print("="*80)
print()

# Configuration
CONFIG = {
    'img_size': 224,
    'batch_size': 64,  # Increased for faster training
    'learning_rate': 0.001,
    'epochs': 15,  # Reduced for 2-hour target
    'num_classes': 2
}

# Paths
DATA_DIR = Path(__file__).parent / 'data'
TRAIN_DIR = DATA_DIR / 'train_augmented'
VAL_DIR = DATA_DIR / 'val_processed'
TEST_DIR = DATA_DIR / 'test_processed'
MODEL_DIR = Path(__file__).parent / 'models'
LOGS_DIR = Path(__file__).parent / 'logs'

MODEL_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

print("📦 LOADING DATA")
print("="*80)

# Create datasets
def load_dataset(directory, batch_size, shuffle=False, augment=False):
    """Load dataset from directory"""
    ds = keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=(CONFIG['img_size'], CONFIG['img_size']),
        batch_size=batch_size,
        label_mode='int',
        shuffle=shuffle
    )
    
    # Normalize
    normalization = layers.Rescaling(1./255)
    ds = ds.map(lambda x, y: (normalization(x), y))
    
    # Prefetch
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return ds

print(f"Loading from: {TRAIN_DIR}")
train_ds = load_dataset(TRAIN_DIR, CONFIG['batch_size'], shuffle=True)
val_ds = load_dataset(VAL_DIR, CONFIG['batch_size'], shuffle=False)
test_ds = load_dataset(TEST_DIR, CONFIG['batch_size'], shuffle=False)

print("✅ Data loaded successfully")
print()

print("🏗️  BUILDING SIMPLE CNN MODEL")
print("="*80)

def create_simple_cnn():
    """Create a simple but effective CNN"""
    model = keras.Sequential([
        # Input
        layers.Input(shape=(224, 224, 3)),
        
        # Block 1
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Classifier
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(CONFIG['num_classes'], activation='softmax')
    ], name='SimpleCNN')
    
    return model

model = create_simple_cnn()
model.summary()

print(f"\n✅ Model created")
print(f"   Total parameters: {model.count_params():,}")
print()

print("⚙️  COMPILING MODEL")
print("="*80)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']  # Simplified metrics to avoid batch size caching issues
)

print("✅ Model compiled")
print()

print("📋 SETTING UP CALLBACKS")
print("="*80)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        MODEL_DIR / 'simple_cnn_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.CSVLogger(
        LOGS_DIR / 'training_history.csv'
    )
]

print("✅ Callbacks configured")
print()

print("🚀 STARTING TRAINING")
print("="*80)
print(f"Epochs: {CONFIG['epochs']}")
print(f"Batch size: {CONFIG['batch_size']}")
print(f"Learning rate: {CONFIG['learning_rate']}")
print("="*80)
print()

start_time = datetime.now()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CONFIG['epochs'],
    callbacks=callbacks,
    verbose=1
)

end_time = datetime.now()
training_time = (end_time - start_time).total_seconds() / 60

print()
print("="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"Training time: {training_time:.1f} minutes")
print()

print("📊 EVALUATING ON TEST SET")
print("="*80)

results = model.evaluate(test_ds, verbose=1)
test_results = {
    'test_loss': float(results[0]),
    'test_accuracy': float(results[1])
}

print()
print("="*80)
print("FINAL RESULTS")
print("="*80)
for metric, value in test_results.items():
    print(f"   {metric}: {value:.4f}")
print("="*80)

# Save results
results_file = MODEL_DIR / 'simple_cnn_results.json'
with open(results_file, 'w') as f:
    json.dump({
        'config': CONFIG,
        'training_time_minutes': training_time,
        'test_results': test_results,
        'history': {k: [float(v) for v in vals] for k, vals in history.history.items()}
    }, f, indent=2)

print(f"\n✅ Results saved to: {results_file}")
print(f"✅ Best model saved to: {MODEL_DIR / 'simple_cnn_best.h5'}")
print()
print("="*80)
print("🎉 ALL DONE!")
print("="*80)
