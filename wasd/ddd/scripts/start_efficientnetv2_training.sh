#!/bin/bash
# Wrapper script to start training with fresh CUDA context

echo "🔧 Starting EfficientNetV2 training with fresh CUDA context..."
echo ""

# Clear any CUDA environment issues
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Start training in a fresh process
exec ml_env_linux/bin/python3 train_efficientnetv2_pytorch_3class_augmented.py
