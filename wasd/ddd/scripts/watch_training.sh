#!/bin/bash
# Live Training Progress Monitor with Auto-Refresh

while true; do
    clear
    echo "======================================================================"
    echo "🔥 PYTORCH TRAINING - LIVE MONITOR"
    echo "======================================================================"
    date
    echo ""
    
    # Check if training is running
    if ps aux | grep -q "[t]rain_convnext_pytorch_3class"; then
        echo "✅ Status: TRAINING IN PROGRESS"
        UPTIME=$(ps -p $(pgrep -f train_convnext_pytorch_3class | head -1) -o etime= 2>/dev/null | xargs)
        echo "⏱️  Running Time: $UPTIME"
        echo ""
        
        # Show last 25 lines with color highlighting
        echo "======================================================================"
        echo "📊 LATEST PROGRESS"
        echo "======================================================================"
        tail -25 pytorch_convnext_training.log | grep -v "pydantic\|UnsupportedFieldAttribute" | tail -20
        
    else
        echo "⏹️  Status: TRAINING STOPPED"
        echo ""
        echo "======================================================================"
        echo "📊 FINAL RESULTS (if available)"
        echo "======================================================================"
        
        if [ -f "bone_fracture_detection/models/convnextv2_3class_augmented_results.json" ]; then
            python3 -c "
import json
with open('bone_fracture_detection/models/convnextv2_3class_augmented_results.json', 'r') as f:
    results = json.load(f)
    print(f\"✅ Training Complete!\")
    print(f\"   Best Val Acc: {results.get('best_val_acc', 0):.2f}%\")
    print(f\"   Test Acc: {results.get('test_acc', 0):.2f}%\")
    print(f\"   Epochs: {results.get('epochs_trained', 0)}\")
    print(f\"   Training Time: {results.get('training_time_minutes', 0):.1f} minutes\")
" 2>/dev/null || tail -30 pytorch_convnext_training.log
        else
            tail -30 pytorch_convnext_training.log
        fi
    fi
    
    echo ""
    echo "======================================================================"
    echo "💡 Press Ctrl+C to exit | Auto-refreshing every 5 seconds..."
    echo "======================================================================"
    
    sleep 5
done
