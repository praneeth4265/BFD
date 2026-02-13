#!/bin/bash
# Quick training status checker

echo "======================================================================"
echo "🚀 PYTORCH TRAINING STATUS"
echo "======================================================================"
echo ""

# Check if running
if ps aux | grep -q "[t]rain_convnext_pytorch"; then
    echo "✅ Status: RUNNING"
    UPTIME=$(ps -p $(pgrep -f train_convnext_pytorch | head -1) -o etime= 2>/dev/null | xargs)
    CPU=$(ps aux | grep "[t]train_convnext_pytorch" | head -1 | awk '{print $3}')
    echo "⏱️  Running for: $UPTIME"
    echo "💻 CPU Usage: ${CPU}%"
    echo ""
else
    echo "⏹️  Status: STOPPED"
    echo ""
fi

echo "======================================================================"
echo "📊 LATEST EPOCHS"
echo "======================================================================"
tail -100 pytorch_convnext_training.log | grep -A 4 "^Epoch" | tail -20

echo ""
echo "======================================================================"
