"""
ConvNeXt V2 Training Script
Optimized for maximum GPU usage with mixed precision training
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ConvNeXtBase
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime

class ConvNeXtV2Trainer:
    """
    ConvNeXt V2 trainer with GPU optimization
    """
    
    def __init__(self, img_size=224, num_classes=2, learning_rate=1e-4):
        self.img_size = img_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
        # Enable mixed precision for faster training
        self.setup_mixed_precision()
        
    def setup_mixed_precision(self):
        """Enable mixed precision training for better GPU utilization"""
        # Disable mixed precision to save memory on 6GB GPU
        policy = tf.keras.mixed_precision.Policy('float32')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✅ Using float32 (mixed precision disabled for memory efficiency)")
        
    def build_model(self, freeze_base=False):
        """
        Build ConvNeXt V2 model with custom classification head
        
        Args:
            freeze_base: Whether to freeze base layers initially
        """
        print("\n" + "="*80)
        print("🏗️  BUILDING CONVNEXT V2 MODEL")
        print("="*80)
        
        # Input layer
        inputs = keras.Input(shape=(self.img_size, self.img_size, 3))
        
        # Apply ConvNeXt preprocessing (expects [0, 255] range, so multiply by 255)
        x = layers.Rescaling(scale=255.0)(inputs)  # Convert [0,1] back to [0,255] for ConvNeXt preprocessing
        
        # Base model (ConvNeXt Base as V2 equivalent)
        base_model = ConvNeXtBase(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_size, self.img_size, 3),
            pooling='avg'
        )
        
        # Pass preprocessed input through base model
        x = base_model(x)
        
        # Freeze base layers if specified
        if freeze_base:
            base_model.trainable = False
            print("   🔒 Base model frozen")
        else:
            base_model.trainable = True
            print("   🔓 Base model trainable")
        
        # Custom classification head
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer (float32 for numerical stability)
        outputs = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(x)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs, name='ConvNeXtV2')
        
        print(f"\n   Model: {self.model.name}")
        print(f"   Input shape: {self.img_size}x{self.img_size}x3")
        print(f"   Output classes: {self.num_classes}")
        print(f"   Total parameters: {self.model.count_params():,}")
        
        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        print(f"   Trainable parameters: {trainable_params:,}")
        
        print("="*80)
        
        return self.model
    
    def compile_model(self):
        """Compile model with specified optimizer and loss"""
        print("\n⚙️  Compiling model...")
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ],
            jit_compile=False  # Disable XLA JIT compilation to avoid shape issues
        )
        
        print(f"   ✅ Optimizer: Adam (lr={self.learning_rate})")
        print(f"   ✅ Loss: Categorical Cross-Entropy")
        print(f"   ✅ Metrics: Accuracy, Precision, Recall, AUC")
        print(f"   ✅ JIT Compile: Disabled")
    
    def create_callbacks(self, model_dir, logs_dir):
        """Create training callbacks"""
        callbacks = []
        
        # Model checkpoint - save best model
        checkpoint_path = model_dir / 'convnext_v2_best.h5'
        checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard
        tensorboard = TensorBoard(
            log_dir=str(logs_dir / f'convnext_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard)
        
        return callbacks
    
    def train(self, train_ds, val_ds, epochs=50, callbacks=None):
        """
        Train the model
        
        Args:
            train_ds: Training dataset
            val_ds: Validation dataset
            epochs: Number of epochs
            callbacks: List of callbacks
        """
        print("\n" + "="*80)
        print("🚀 STARTING TRAINING - CONVNEXT V2")
        print("="*80)
        
        start_time = time.time()
        
        # Train model
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE - CONVNEXT V2")
        print("="*80)
        print(f"   Training time: {training_time/60:.2f} minutes")
        print(f"   Final train accuracy: {self.history.history['accuracy'][-1]:.4f}")
        print(f"   Final val accuracy: {self.history.history['val_accuracy'][-1]:.4f}")
        print(f"   Best val accuracy: {max(self.history.history['val_accuracy']):.4f}")
        print("="*80)
        
        return self.history
    
    def evaluate(self, test_ds):
        """Evaluate model on test set"""
        print("\n" + "="*80)
        print("📊 EVALUATING MODEL - CONVNEXT V2")
        print("="*80)
        
        results = self.model.evaluate(test_ds, verbose=1)
        
        print("\n✅ Test Results:")
        for name, value in zip(self.model.metrics_names, results):
            print(f"   {name}: {value:.4f}")
        
        print("="*80)
        
        return dict(zip(self.model.metrics_names, results))
    
    def save_history(self, save_path):
        """Save training history"""
        history_dict = {k: [float(v) for v in vals] for k, vals in self.history.history.items()}
        
        with open(save_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"✅ History saved to {save_path}")
    
    def save_model(self, save_path):
        """Save complete model"""
        self.model.save(save_path)
        print(f"✅ Model saved to {save_path}")


def main():
    """Test training script"""
    print("Testing ConvNeXt V2 trainer...")
    
    trainer = ConvNeXtV2Trainer(img_size=224, num_classes=2, learning_rate=1e-4)
    model = trainer.build_model(freeze_base=False)
    trainer.compile_model()
    
    print("\n✅ ConvNeXt V2 trainer initialized successfully!")
    print(f"   Model ready for training")


if __name__ == "__main__":
    main()
