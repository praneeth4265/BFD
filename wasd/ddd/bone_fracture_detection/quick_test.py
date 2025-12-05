#!/usr/bin/env python3
"""
Quick Test - Bone Fracture Detection
Run this to quickly test the model with random images
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_single_image import test_single_image
from pathlib import Path
import random

print("="*80)
print("🚀 QUICK TEST - BONE FRACTURE DETECTION MODEL")
print("="*80)
print()

# Find test images
test_dir = Path(__file__).parent / "data_original" / "test"
test_images = {
    'comminuted': [],
    'simple': []
}

comminuted_dir = test_dir / "comminuted_fracture"
simple_dir = test_dir / "simple_fracture"

if comminuted_dir.exists():
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        test_images['comminuted'].extend(list(comminuted_dir.glob(ext)))

if simple_dir.exists():
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        test_images['simple'].extend(list(simple_dir.glob(ext)))

total_images = len(test_images['comminuted']) + len(test_images['simple'])

if total_images == 0:
    print("❌ No test images found!")
    sys.exit(1)

print(f"📁 Found {total_images} test images:")
print(f"   - Comminuted: {len(test_images['comminuted'])}")
print(f"   - Simple: {len(test_images['simple'])}")
print()

# Test 3 random images
num_tests = min(3, total_images)
print(f"🎲 Testing {num_tests} random images...")
print("="*80)
print()

all_images = test_images['comminuted'] + test_images['simple']
random_samples = random.sample(all_images, num_tests)

for idx, img_path in enumerate(random_samples, 1):
    print(f"\n{'='*80}")
    print(f"TEST {idx}/{num_tests}")
    print('='*80)
    print()
    
    # Determine true label
    if 'comminuted' in str(img_path):
        true_label = "Comminuted Fracture"
    else:
        true_label = "Simple Fracture"
    
    print(f"📸 Image: {img_path.name}")
    print(f"🏷️  True Label: {true_label}")
    print()
    
    # Test
    test_single_image(str(img_path))
    
    print()

print("\n" + "="*80)
print("🎉 ALL TESTS COMPLETE!")
print("="*80)
print()
print("💡 To test a specific image:")
print("   python test_single_image.py --image path/to/your/xray.jpg")
print()
