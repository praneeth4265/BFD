"""
Model optimization and deployment utilities for bone fracture detection.
Includes quantization, ONNX conversion, pruning, and performance benchmarking.
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
import onnx
import onnxruntime as ort
import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import yaml
import json
import warnings
import logging

from utils import Timer, DeviceManager, ConfigManager


class ModelOptimizer:
    """
    Comprehensive model optimization for deployment.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any] = None,
        device: str = 'auto'
    ):
        """
        Initialize model optimizer.
        
        Args:
            model: PyTorch model to optimize
            config: Configuration dictionary
            device: Device to use for optimization
        """
        self.model = model
        self.config = config or {}
        
        if device == 'auto':
            self.device = DeviceManager.get_device(prefer_gpu=False)  # CPU for optimization
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.logger = logging.getLogger(__name__)
    
    def quantize_model(
        self,
        calibration_loader=None,
        quantization_type: str = 'dynamic'
    ) -> nn.Module:
        """
        Apply quantization to reduce model size and improve inference speed.
        
        Args:
            calibration_loader: Data loader for static quantization calibration
            quantization_type: Type of quantization ('dynamic', 'static', 'qat')
            
        Returns:
            Quantized model
        """
        self.logger.info(f"Applying {quantization_type} quantization...")
        
        if quantization_type == 'dynamic':
            # Dynamic quantization (no calibration data needed)
            quantized_model = quantization.quantize_dynamic(
                self.model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            
        elif quantization_type == 'static':
            # Static quantization (requires calibration data)
            if calibration_loader is None:
                raise ValueError("Calibration data required for static quantization")
            
            # Prepare model for static quantization
            self.model.qconfig = quantization.get_default_qconfig('fbgemm')
            quantization.prepare(self.model, inplace=True)
            
            # Calibrate with representative data
            self.logger.info("Calibrating model...")
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(calibration_loader):
                    if batch_idx >= 100:  # Limit calibration samples
                        break
                    data = data.to(self.device)
                    self.model(data)
            
            # Convert to quantized model
            quantized_model = quantization.convert(self.model, inplace=False)
            
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
        
        self.logger.info("Quantization completed")
        return quantized_model
    
    def prune_model(
        self,
        pruning_ratio: float = 0.2,
        structured: bool = False
    ) -> nn.Module:
        """
        Apply pruning to reduce model parameters.
        
        Args:
            pruning_ratio: Fraction of parameters to prune
            structured: Whether to use structured pruning
            
        Returns:
            Pruned model
        """
        import torch.nn.utils.prune as prune
        
        self.logger.info(f"Applying pruning with ratio {pruning_ratio}...")
        
        # Collect parameters to prune
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        if structured:
            # Structured pruning (prune entire channels/neurons)
            for module, param_name in parameters_to_prune:
                if isinstance(module, nn.Conv2d):
                    prune.ln_structured(
                        module, param_name, amount=pruning_ratio, n=2, dim=0
                    )
                elif isinstance(module, nn.Linear):
                    prune.ln_structured(
                        module, param_name, amount=pruning_ratio, n=2, dim=0
                    )
        else:
            # Unstructured pruning
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio
            )
        
        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        self.logger.info("Pruning completed")
        return self.model
    
    def export_to_onnx(
        self,
        save_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        opset_version: int = 11,
        dynamic_axes: Dict[str, Dict[int, str]] = None
    ) -> str:
        """
        Export model to ONNX format.
        
        Args:
            save_path: Path to save ONNX model
            input_shape: Input tensor shape
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes for variable input sizes
            
        Returns:
            Path to saved ONNX model
        """
        self.logger.info(f"Exporting model to ONNX: {save_path}")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Dynamic axes for batch size flexibility
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        
        self.logger.info("ONNX export completed and verified")
        return save_path
    
    def optimize_onnx_model(
        self,
        onnx_path: str,
        optimization_level: str = 'all'
    ) -> str:
        """
        Optimize ONNX model for better performance.
        
        Args:
            onnx_path: Path to ONNX model
            optimization_level: Optimization level ('basic', 'extended', 'all')
            
        Returns:
            Path to optimized ONNX model
        """
        import onnxruntime as ort
        
        self.logger.info(f"Optimizing ONNX model with level: {optimization_level}")
        
        # Set optimization level
        if optimization_level == 'basic':
            graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        elif optimization_level == 'extended':
            graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        else:  # 'all'
            graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Create optimized model path
        optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
        
        # Optimize
        sess_options = ort.SessionOptions()
        sess_options.optimized_model_filepath = optimized_path
        sess_options.graph_optimization_level = graph_optimization_level
        
        # Create session to trigger optimization
        _ = ort.InferenceSession(onnx_path, sess_options)
        
        self.logger.info(f"Optimized ONNX model saved: {optimized_path}")
        return optimized_path


class ModelBenchmark:
    """
    Benchmark model performance across different formats and optimizations.
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize benchmarker.
        
        Args:
            device: Device to use for benchmarking
        """
        if device == 'auto':
            self.device = DeviceManager.get_device()
        else:
            self.device = torch.device(device)
        
        self.logger = logging.getLogger(__name__)
    
    def benchmark_pytorch_model(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        num_warmup: int = 10,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark PyTorch model inference time.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            num_warmup: Number of warmup runs
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        model = model.to(self.device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)
        
        # Synchronize GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(dummy_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'fps': 1000 / np.mean(times)
        }
    
    def benchmark_onnx_model(
        self,
        onnx_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        num_warmup: int = 10,
        num_runs: int = 100,
        providers: List[str] = None
    ) -> Dict[str, float]:
        """
        Benchmark ONNX model inference time.
        
        Args:
            onnx_path: Path to ONNX model
            input_shape: Input tensor shape
            num_warmup: Number of warmup runs
            num_runs: Number of benchmark runs
            providers: ONNX execution providers
            
        Returns:
            Benchmark results
        """
        if providers is None:
            providers = ['CPUExecutionProvider']
            if torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        input_name = session.get_inputs()[0].name
        
        # Warmup
        for _ in range(num_warmup):
            _ = session.run(None, {input_name: dummy_input})
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = session.run(None, {input_name: dummy_input})
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'fps': 1000 / np.mean(times),
            'providers': providers
        }
    
    def compare_model_formats(
        self,
        pytorch_model: nn.Module,
        onnx_path: str = None,
        quantized_model: nn.Module = None,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare performance across different model formats.
        
        Args:
            pytorch_model: Original PyTorch model
            onnx_path: Path to ONNX model
            quantized_model: Quantized PyTorch model
            input_shape: Input tensor shape
            
        Returns:
            Comparison results
        """
        results = {}
        
        # Benchmark PyTorch model
        self.logger.info("Benchmarking PyTorch model...")
        results['pytorch'] = self.benchmark_pytorch_model(pytorch_model, input_shape)
        
        # Benchmark ONNX model
        if onnx_path and Path(onnx_path).exists():
            self.logger.info("Benchmarking ONNX model...")
            results['onnx'] = self.benchmark_onnx_model(onnx_path, input_shape)
        
        # Benchmark quantized model
        if quantized_model is not None:
            self.logger.info("Benchmarking quantized model...")
            results['quantized'] = self.benchmark_pytorch_model(quantized_model, input_shape)
        
        # Calculate speedups
        if 'pytorch' in results:
            baseline_time = results['pytorch']['mean_ms']
            for format_name, format_results in results.items():
                if format_name != 'pytorch':
                    speedup = baseline_time / format_results['mean_ms']
                    format_results['speedup'] = speedup
        
        return results
    
    def get_model_size(self, model_path: str) -> Dict[str, float]:
        """
        Get model file size information.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Size information
        """
        path = Path(model_path)
        if not path.exists():
            return {'error': 'File not found'}
        
        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        
        return {
            'size_bytes': size_bytes,
            'size_mb': size_mb,
            'size_kb': size_bytes / 1024
        }


class DeploymentPackager:
    """
    Package models for deployment with necessary preprocessing and postprocessing.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize deployment packager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def create_onnx_package(
        self,
        model: nn.Module,
        save_dir: str,
        model_name: str = 'bone_fracture_model',
        include_preprocessing: bool = True
    ) -> Dict[str, str]:
        """
        Create a complete ONNX deployment package.
        
        Args:
            model: PyTorch model
            save_dir: Directory to save package
            model_name: Name for the model
            include_preprocessing: Whether to include preprocessing info
            
        Returns:
            Dictionary with package file paths
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        package_files = {}
        
        # Initialize optimizer
        optimizer = ModelOptimizer(model, self.config)
        
        # Export to ONNX
        onnx_path = save_dir / f"{model_name}.onnx"
        optimizer.export_to_onnx(str(onnx_path))
        package_files['onnx_model'] = str(onnx_path)
        
        # Optimize ONNX model
        optimized_path = optimizer.optimize_onnx_model(str(onnx_path))
        package_files['optimized_onnx'] = optimized_path
        
        # Create quantized version if possible
        try:
            quantized_model = optimizer.quantize_model(quantization_type='dynamic')
            quantized_path = save_dir / f"{model_name}_quantized.pt"
            torch.save(quantized_model.state_dict(), quantized_path)
            package_files['quantized_model'] = str(quantized_path)
        except Exception as e:
            self.logger.warning(f"Quantization failed: {e}")
        
        # Save model metadata
        metadata = {
            'model_name': model_name,
            'input_shape': [1, 3, 224, 224],
            'output_classes': ['No Fracture', 'Fracture'],
            'preprocessing': {
                'resize': [224, 224],
                'normalize_mean': [0.485, 0.456, 0.406],
                'normalize_std': [0.229, 0.224, 0.225],
                'apply_clahe': True
            } if include_preprocessing else {},
            'version': '1.0.0'
        }
        
        metadata_path = save_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        package_files['metadata'] = str(metadata_path)
        
        # Create inference script template
        inference_script = self._create_inference_script(model_name)
        script_path = save_dir / f"{model_name}_inference.py"
        with open(script_path, 'w') as f:
            f.write(inference_script)
        package_files['inference_script'] = str(script_path)
        
        # Create README
        readme = self._create_readme(model_name, package_files)
        readme_path = save_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme)
        package_files['readme'] = str(readme_path)
        
        self.logger.info(f"Deployment package created in: {save_dir}")
        return package_files
    
    def _create_inference_script(self, model_name: str) -> str:
        """Create a template inference script."""
        return f'''"""
Inference script for {model_name} bone fracture detection model.
"""

import onnxruntime as ort
import numpy as np
import cv2
import json
from pathlib import Path


class BoneFractureInference:
    def __init__(self, model_path: str, metadata_path: str = None):
        """Initialize inference engine."""
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        # Load metadata
        if metadata_path:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            # Default values
            self.metadata = {{
                'input_shape': [1, 3, 224, 224],
                'preprocessing': {{
                    'resize': [224, 224],
                    'normalize_mean': [0.485, 0.456, 0.406],
                    'normalize_std': [0.229, 0.224, 0.225]
                }},
                'output_classes': ['No Fracture', 'Fracture']
            }}
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess input image."""
        # Resize
        target_size = tuple(self.metadata['preprocessing']['resize'])
        image = cv2.resize(image, target_size)
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array(self.metadata['preprocessing']['normalize_mean'])
        std = np.array(self.metadata['preprocessing']['normalize_std'])
        image = (image - mean) / std
        
        # Add batch dimension and transpose to CHW
        image = np.expand_dims(image, axis=0)  # Add batch
        image = np.transpose(image, (0, 3, 1, 2))  # BHWC to BCHW
        
        return image
    
    def predict(self, image: np.ndarray) -> dict:
        """Make prediction on input image."""
        # Preprocess
        processed_image = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(None, {{self.input_name: processed_image}})
        logits = outputs[0]
        
        # Apply softmax
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        
        # Get prediction
        predicted_class = np.argmax(probabilities, axis=1)[0]
        confidence = probabilities[0][predicted_class]
        
        return {{
            'predicted_class': int(predicted_class),
            'predicted_label': self.metadata['output_classes'][predicted_class],
            'confidence': float(confidence),
            'probabilities': probabilities[0].tolist()
        }}


# Example usage
if __name__ == "__main__":
    # Initialize inference engine
    inference = BoneFractureInference(
        model_path="{model_name}.onnx",
        metadata_path="{model_name}_metadata.json"
    )
    
    # Load and predict on an image
    image_path = "sample_xray.jpg"  # Replace with actual image path
    image = cv2.imread(image_path)
    
    if image is not None:
        result = inference.predict(image)
        print(f"Prediction: {{result['predicted_label']}}")
        print(f"Confidence: {{result['confidence']:.3f}}")
    else:
        print("Could not load image")
'''
    
    def _create_readme(self, model_name: str, package_files: Dict[str, str]) -> str:
        """Create README for the deployment package."""
        return f'''# {model_name} Deployment Package

## Overview
This package contains a trained bone fracture detection model optimized for deployment.

## Files Included
- `{model_name}.onnx`: Main ONNX model
- `{model_name}_optimized.onnx`: Optimized ONNX model (recommended)
- `{model_name}_metadata.json`: Model metadata and preprocessing parameters
- `{model_name}_inference.py`: Inference script template
- `README.md`: This file

## Quick Start

### Requirements
```bash
pip install onnxruntime opencv-python numpy
```

### Basic Usage
```python
from {model_name}_inference import BoneFractureInference
import cv2

# Initialize model
model = BoneFractureInference(
    model_path="{model_name}_optimized.onnx",
    metadata_path="{model_name}_metadata.json"
)

# Load and predict
image = cv2.imread("xray_image.jpg")
result = model.predict(image)

print(f"Prediction: {{result['predicted_label']}}")
print(f"Confidence: {{result['confidence']:.3f}}")
```

## Model Details
- **Input**: RGB image, resized to 224x224
- **Output**: Binary classification (Fracture/No Fracture)
- **Preprocessing**: CLAHE enhancement, normalization
- **Format**: ONNX (compatible with ONNX Runtime)

## Performance
- Optimized for CPU inference
- Typical inference time: <100ms on modern CPU
- Model size: <50MB

## Integration Notes
1. Ensure proper image preprocessing (see metadata.json)
2. Use the optimized ONNX model for best performance
3. Consider batch processing for multiple images
4. Implement proper error handling for production use

## Support
For questions or issues, please refer to the project documentation.
'''


def optimize_and_package_model(
    model: nn.Module,
    config_path: str,
    save_dir: str,
    model_name: str = 'bone_fracture_model'
) -> Dict[str, Any]:
    """
    Complete model optimization and packaging pipeline.
    
    Args:
        model: Trained PyTorch model
        config_path: Path to configuration file
        save_dir: Directory to save optimized models
        model_name: Name for the model
        
    Returns:
        Dictionary with optimization results and file paths
    """
    config = ConfigManager.load_config(config_path)
    
    # Initialize components
    optimizer = ModelOptimizer(model, config)
    benchmarker = ModelBenchmark()
    packager = DeploymentPackager(config)
    
    results = {}
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Benchmark original model
    results['original_benchmark'] = benchmarker.benchmark_pytorch_model(model)
    
    # Apply optimizations
    try:
        # Quantization
        quantized_model = optimizer.quantize_model(quantization_type='dynamic')
        results['quantized_benchmark'] = benchmarker.benchmark_pytorch_model(quantized_model)
        
        # Save quantized model
        quantized_path = save_dir / f"{model_name}_quantized.pth"
        torch.save(quantized_model.state_dict(), quantized_path)
        results['quantized_model_path'] = str(quantized_path)
        
    except Exception as e:
        logging.warning(f"Quantization failed: {e}")
        quantized_model = None
    
    # Export to ONNX
    onnx_path = save_dir / f"{model_name}.onnx"
    optimizer.export_to_onnx(str(onnx_path))
    results['onnx_model_path'] = str(onnx_path)
    
    # Optimize ONNX
    optimized_onnx_path = optimizer.optimize_onnx_model(str(onnx_path))
    results['optimized_onnx_path'] = optimized_onnx_path
    
    # Benchmark ONNX models
    results['onnx_benchmark'] = benchmarker.benchmark_onnx_model(str(onnx_path))
    results['optimized_onnx_benchmark'] = benchmarker.benchmark_onnx_model(optimized_onnx_path)
    
    # Compare all formats
    results['comparison'] = benchmarker.compare_model_formats(
        model, str(onnx_path), quantized_model
    )
    
    # Get model sizes
    results['model_sizes'] = {
        'onnx': benchmarker.get_model_size(str(onnx_path)),
        'optimized_onnx': benchmarker.get_model_size(optimized_onnx_path)
    }
    
    if quantized_model is not None:
        results['model_sizes']['quantized'] = benchmarker.get_model_size(str(quantized_path))
    
    # Create deployment package
    package_files = packager.create_onnx_package(
        model, str(save_dir), model_name
    )
    results['package_files'] = package_files
    
    # Save optimization report
    report_path = save_dir / f"{model_name}_optimization_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def test_optimization():
    """Test model optimization functionality."""
    from models import create_bone_cnn
    
    # Create dummy model
    model = create_bone_cnn('compact', num_classes=2)
    
    # Test optimization
    optimizer = ModelOptimizer(model)
    
    # Test quantization
    try:
        quantized_model = optimizer.quantize_model(quantization_type='dynamic')
        print("Quantization successful")
    except Exception as e:
        print(f"Quantization failed: {e}")
    
    # Test ONNX export
    try:
        onnx_path = "test_model.onnx"
        optimizer.export_to_onnx(onnx_path)
        print("ONNX export successful")
        
        # Clean up
        Path(onnx_path).unlink(missing_ok=True)
        
    except Exception as e:
        print(f"ONNX export failed: {e}")
    
    # Test benchmarking
    benchmarker = ModelBenchmark()
    results = benchmarker.benchmark_pytorch_model(model, num_runs=10)
    print(f"Benchmark results: {results}")
    
    print("Optimization tests completed!")


if __name__ == "__main__":
    test_optimization()