#!/bin/bash
# Monitor PyTorch Training Progress

echo "======================================================================"
echo "🔍 PYTORCH TRAINING MONITOR"
echo "======================================================================"
echo ""

# Check if training process is running
if ps aux | grep -q "[t]rain_convnext_pytorch_3class"; then
    echo "✅ ConvNeXt V2 training is RUNNING"
    ps aux | grep "[t]rain_convnext_pytorch_3class" | awk '{print "   PID:", $2, "| CPU:", $3"% | MEM:", $4"% | TIME:", $10}'
    echo ""
fi

if ps aux | grep -q "[t]rain_efficientnetv2_pytorch_3class"; then
    echo "✅ EfficientNetV2 training is RUNNING"
    ps aux | grep "[t]rain_efficientnetv2_pytorch_3class" | awk '{print "   PID:", $2, "| CPU:", $3"% | MEM:", $4"% | TIME:", $10}'
    echo ""
fi

# Show latest training progress
echo "======================================================================"
echo "📊 LATEST TRAINING PROGRESS"
echo "======================================================================"
echo ""

if [ -f "pytorch_convnext_training.log" ]; then
    echo "--- ConvNeXt V2 ---"
    tail -20 pytorch_convnext_training.log | grep -E "Epoch|Train Loss|Val Loss|Best model|Training is"
    echo ""
fi

if [ -f "pytorch_efficientnetv2_training.log" ]; then
    echo "--- EfficientNetV2 ---"
    tail -20 pytorch_efficientnetv2_training.log | grep -E "Epoch|Train Loss|Val Loss|Best model|Training is"
    echo ""
fi

echo "======================================================================"
echo "💾 MODEL FILES"
echo "======================================================================"
ls -lh bone_fracture_detection/models/*3class_augmented* 2>/dev/null || echo "   No models saved yet"
echo ""

echo "======================================================================"
echo "📈 RESULTS"
echo "======================================================================"
if [ -f "bone_fracture_detection/models/convnextv2_3class_augmented_results.json" ]; then
    echo "--- ConvNeXt V2 Results ---"
    cat bone_fracture_detection/models/convnextv2_3class_augmented_results.json | grep -E "best_val_acc|test_acc|epochs_trained" | head -5
    echo ""
fi

if [ -f "bone_fracture_detection/models/efficientnetv2_3class_augmented_results.json" ]; then
    echo "--- EfficientNetV2 Results ---"
    cat bone_fracture_detection/models/efficientnetv2_3class_augmented_results.json | grep -E "best_val_acc|test_acc|epochs_trained" | head -5
    echo ""
fi

echo "======================================================================"
echo "Run this script again to check latest progress"
echo "======================================================================"
