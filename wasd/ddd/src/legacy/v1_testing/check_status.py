#!/usr/bin/env python3
"""
Quick status checker for training progress
"""

import os
import time

log_file = "/home/praneeth4265/wasd/ddd/bone_fracture_detection/training_convnext.log"

print("=" * 80)
print("CONVNEXT V2 TRAINING STATUS")
print("=" * 80)
print()

# Check if process is running
pid = 14530
try:
    os.kill(pid, 0)
    print(f"✅ Training process (PID {pid}) is RUNNING")
except:
    print(f"❌ Training process (PID {pid}) is NOT running")

print()
print("=" * 80)
print("LAST 50 LINES OF LOG:")
print("=" * 80)
print()

# Read last 50 lines
try:
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines[-50:]:
            print(line, end='')
except FileNotFoundError:
    print("Log file not found yet...")

print()
print("=" * 80)
print(f"Full log: {log_file}")
print("=" * 80)
