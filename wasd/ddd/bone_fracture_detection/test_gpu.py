"""
GPU Test - Check if we can train on GPU without CUDA compilation issues
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
    except RuntimeError as e:
        print(f"Error: {e}")

# Create simple model
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(2),
    layers.GlobalAveragePooling2D(),
    layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✅ Model created and compiled")

# Create dummy data
import numpy as np
x_train = np.random.random((100, 224, 224, 3)).astype('float32')
y_train = np.random.randint(0, 2, (100,))

print("\n🚀 Testing training on GPU...")
try:
    history = model.fit(x_train, y_train, batch_size=16, epochs=2, verbose=1)
    print("\n✅ ✅ ✅ GPU TRAINING WORKS! ✅ ✅ ✅")
    print("We can proceed with ConvNeXt V2 and EfficientNetV2!")
except Exception as e:
    print(f"\n❌ GPU Training failed: {e}")
    print("Need to use CPU or fix GPU issue")
