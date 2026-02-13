#!/bin/bash
# Monitoring script for training progress

echo "=========================================="
echo "TRAINING MONITOR"
echo "=========================================="
echo ""

# Check if training process is running
if ps aux | grep "train_convnext.py" | grep -v grep > /dev/null; then
    echo "✅ Training process is RUNNING"
    ps aux | grep "train_convnext.py" | grep -v grep | awk '{print "   PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "TIME:", $10}'
else
    echo "❌ Training process is NOT running"
fi

echo ""
echo "=========================================="
echo "LAST 30 LINES OF TRAINING LOG:"
echo "=========================================="
tail -30 /home/praneeth4265/wasd/ddd/bone_fracture_detection/training.log

echo ""
echo "=========================================="
echo "GPU STATUS:"
echo "=========================================="
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits
