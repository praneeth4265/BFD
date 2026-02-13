#!/bin/bash
# Live Swin Training Monitor - Auto-refresh every 3 seconds

watch -n 3 '
clear
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          🔥 Swin Transformer - LIVE MONITOR 🔥               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

echo "📊 GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
awk "{print \"   GPU Utilization: \"\$1\"%\"}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | \
awk "{print \"   GPU Memory: \"\$1\" / \"\$2\" MB\"}"
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | \
awk "{print \"   Temperature: \"\$1\"°C\"}"
echo ""

echo "📈 Training Progress:"
tail -50 pytorch_swin_training.log | grep -v "pydantic" | grep -v "warnings.warn" | tail -30
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "Refreshing every 3 seconds... (Press Ctrl+C to exit)"
echo "════════════════════════════════════════════════════════════════"
'
