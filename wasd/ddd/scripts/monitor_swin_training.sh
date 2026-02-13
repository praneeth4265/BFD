#!/bin/bash
# Monitor Swin Transformer Training Progress

echo "=================================================="
echo "🔍 SWIN TRANSFORMER TRAINING MONITOR"
echo "=================================================="
echo ""

echo "📊 GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | head -1 | awk '{print "   GPU Utilization: "$1"% | Memory: "$3"/"$4" MB ("$2"%)"}'
echo ""

echo "📈 Training Progress (Last 30 lines):"
tail -30 pytorch_swin_training.log | grep -v "pydantic" | grep -v "warnings.warn"
echo ""

echo "=================================================="
echo "To watch live: bash watch_swin_live.sh"
echo "=================================================="
