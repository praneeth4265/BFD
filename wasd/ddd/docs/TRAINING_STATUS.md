# Training Status Report
**Date:** February 7, 2026  
**Task:** Train models on 3-class augmented dataset

## Dataset Status ✅
- **Augmented Dataset:** 20,530 images
  - Train: 14,370 images (5,163 comm, 4,790 nofrac, 4,417 simp)
  - Val: 3,078 images (1,106 comm, 1,026 nofrac, 946 simp)
  - Test: 3,082 images (1,107 comm, 1,027 nofrac, 948 simp)
- **Classes:** 3 (comminuted_fracture, simple_fracture, no_fracture)
- **Balance:** ✅ Well balanced (~33% each class, within 6% variance)
- **Quality:** ✅ High quality augmentations with 6 diverse transforms

## Technical Issues Encountered ❌

### ConvNeXt V2 Training
**Status:** FAILED  
**Error:** JIT compilation failure in preprocessing normalization layer
```
JIT compilation failed.
[[{{node ConvNeXtV2_1/convnext_base_prestem_normalization_1/Sqrt}}]]
```
**Root Cause:** ConvNeXt Base model includes preprocessing normalization layers that fail with TensorFlow's XLA JIT compilation on this setup.

### EfficientNetV2-S Training (Attempt 1)
**Status:** FAILED  
**Error:** GPU Out of Memory
**Configuration:** 384x384 images, batch_size=16
```
Resource exhaustedError: failed to allocate memory
[[{{node EfficientNetV2S_1/block6f_expand_activation_1/Sigmoid}}]]
```
**Root Cause:** 20M parameter model + 384x384 images exceeded 6.16 GB available GPU memory

### EfficientNetV2-S Training (Attempt 2)
**Status:** FAILED  
**Error:** JIT compilation failure in Adam optimizer
**Configuration:** 224x224 images, batch_size=24
```
error: libdevice not found at ./libdevice.10.bc
JIT compilation failed.
[[{{node adam/Pow_747}}]]
```
**Root Cause:** TensorFlow trying to use XLA JIT compilation despite it being disabled. Missing CUDA libdevice file.

## Root Cause Analysis
The core issue is **TensorFlow/Keras's automatic graph optimization and XLA JIT compilation** which:
1. Attempts to compile operations even when `jit_compile=False` is set
2. Fails due to missing or incompatible CUDA libraries (libdevice.10.bc)
3. Incompatible with ConvNeXt's preprocessing normalization layers
4. Triggers on complex models like ConvNeXt and EfficientNetV2

## System Environment
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU
  - Memory: 6.16 GB available
  - Compute Capability: 8.9
- **TensorFlow:** 2.15.x (Keras 3.x)
- **Python:** 3.10
- **CUDA:** Present but libdevice.10.bc missing or not in search path

## Solutions Attempted
1. ✅ Fixed import paths for training modules
2. ✅ Fixed data loader initialization (use `create_datasets()`)
3. ✅ Fixed trainer initialization (pass individual params, not dict)
4. ✅ Added `compile_model()` call before training
5. ✅ Reduced image size (384→224) for EfficientNetV2
6. ✅ Adjusted batch size for memory
7. ❌ Set `jit_compile=False` in model compile (still triggered internally)
8. ❌ Set environment variables `TF_XLA_FLAGS="--tf_xla_auto_jit=0"` (insufficient)
9. ❌ Modified ConvNeXt preprocessing with Rescaling layer (still failed)

## Recommendations for Resolution

### Option 1: Use PyTorch (Recommended) ⭐
**Pros:**
- No JIT compilation issues
- Better GPU memory management
- More control over training loop
- Existing PyTorch training scripts in `bone_fracture_detection/`
- Simpler debugging

**Action:** Adapt existing PyTorch training scripts for 3-class augmented dataset

### Option 2: Fix TensorFlow Environment
**Actions needed:**
1. Locate/install proper CUDA libdevice file
2. Set `XLA_FLAGS` to point to CUDA directory
3. Completely disable graph optimization: `TF_DISABLE_MKL=1 TF_DISABLE_SEGMENT_REDUCTION=1`
4. Use eager execution mode: `tf.config.run_functions_eagerly(True)`
5. Try older TensorFlow version (2.10.x) with Keras 2.x

**Complexity:** High, may require CUDA reinstallation

### Option 3: Use Simpler Architecture
**Action:** Train with ResNet50 or MobileNetV2 (lighter models)
**Pros:** Less likely to trigger JIT compilation issues
**Cons:** May have lower accuracy than ConvNeXt/EfficientNetV2

## Next Steps
1. **Immediate:** Try PyTorch training scripts (already exist and working)
2. **Alternative:** Switch to ResNet50 with TensorFlow
3. **Long-term:** Fix CUDA/TensorFlow environment for future compatibility

## Files Created
- `train_convnext_3class_augmented.py` - ConvNeXt training script (not working)
- `train_efficientnetv2_3class_augmented.py` - EfficientNetV2 training script (not working)
- `convnext_training.log` - Error logs
- `efficientnetv2_training.log` - Error logs

## Files Modified
- `bone_fracture_detection/src/train_convnext_v2.py` - Added Rescaling layer (didn't resolve issue)
- `bone_fracture_detection/src/train_efficientnet_v2.py` - No changes needed

## Conclusion
Training failed due to TensorFlow/XLA JIT compilation issues on this system. **Recommended path forward: Use PyTorch** which has proven more reliable in this environment and already has working training infrastructure.
