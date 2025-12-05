"""
Utility functions for the bone fracture detection project.
Includes device management, metrics, visualization, and helper functions.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from typing import Dict, List, Tuple, Any, Optional
import yaml
import json
from pathlib import Path
import time
import psutil
import GPUtil
from dataclasses import dataclass
import logging


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: np.ndarray
    classification_report: str


class DeviceManager:
    """Manages device selection and memory optimization."""
    
    @staticmethod
    def get_device(prefer_gpu: bool = True) -> torch.device:
        """
        Get the best available device.
        
        Args:
            prefer_gpu: Whether to prefer GPU over CPU
            
        Returns:
            PyTorch device
        """
        if prefer_gpu and torch.cuda.is_available():
            # Select GPU with most free memory
            if torch.cuda.device_count() > 1:
                gpu_memory = []
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                    gpu_memory.append(free_memory)
                best_gpu = gpu_memory.index(max(gpu_memory))
                device = torch.device(f'cuda:{best_gpu}')
            else:
                device = torch.device('cuda:0')
            
            print(f"Using device: {device}")
            print(f"GPU: {torch.cuda.get_device_name(device)}")
            print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
            
        else:
            device = torch.device('cpu')
            print(f"Using device: {device}")
            print(f"CPU cores: {psutil.cpu_count()}")
            print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
        
        return device
    
    @staticmethod
    def optimize_memory():
        """Optimize GPU memory usage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Enable memory efficient attention if available
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except:
                pass
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage."""
        usage = {}
        
        # CPU memory
        cpu_memory = psutil.virtual_memory()
        usage['cpu_used_gb'] = cpu_memory.used / 1e9
        usage['cpu_total_gb'] = cpu_memory.total / 1e9
        usage['cpu_percent'] = cpu_memory.percent
        
        # GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory
                gpu_allocated = torch.cuda.memory_allocated(i)
                usage[f'gpu_{i}_used_gb'] = gpu_allocated / 1e9
                usage[f'gpu_{i}_total_gb'] = gpu_memory / 1e9
                usage[f'gpu_{i}_percent'] = (gpu_allocated / gpu_memory) * 100
        
        return usage


class MetricsCalculator:
    """Calculate and format evaluation metrics."""
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None
    ) -> MetricsResult:
        """
        Calculate comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for AUC)
            class_names: Class names for report
            
        Returns:
            MetricsResult object
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            zero_division=0
        )
        
        # AUC-ROC (if probabilities provided)
        auc = 0.0
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    auc = roc_auc_score(y_true, y_proba[:, 1])
                else:  # Multi-class
                    auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            except ValueError:
                auc = 0.0
        
        return MetricsResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc,
            confusion_matrix=cm,
            classification_report=report
        )
    
    @staticmethod
    def print_metrics(metrics: MetricsResult, title: str = "Evaluation Metrics"):
        """Print formatted metrics."""
        print(f"\n{'='*50}")
        print(f"{title:^50}")
        print(f"{'='*50}")
        print(f"Accuracy:  {metrics.accuracy:.4f}")
        print(f"Precision: {metrics.precision:.4f}")
        print(f"Recall:    {metrics.recall:.4f}")
        print(f"F1-Score:  {metrics.f1_score:.4f}")
        print(f"AUC-ROC:   {metrics.auc_roc:.4f}")
        print(f"\nConfusion Matrix:")
        print(metrics.confusion_matrix)
        print(f"\nClassification Report:")
        print(metrics.classification_report)


class Visualizer:
    """Visualization utilities for training and evaluation."""
    
    @staticmethod
    def plot_training_history(
        train_losses: List[float],
        val_losses: List[float],
        train_metrics: List[float],
        val_metrics: List[float],
        metric_name: str = "Accuracy",
        save_path: Optional[str] = None
    ):
        """
        Plot training history.
        
        Args:
            train_losses: Training losses
            val_losses: Validation losses
            train_metrics: Training metrics
            val_metrics: Validation metrics
            metric_name: Name of the metric
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot metrics
        ax2.plot(epochs, train_metrics, 'b-', label=f'Training {metric_name}')
        ax2.plot(epochs, val_metrics, 'r-', label=f'Validation {metric_name}')
        ax2.set_title(f'Model {metric_name}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(metric_name)
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        class_names: List[str],
        normalize: bool = False,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: Class names
            normalize: Whether to normalize the matrix
            save_path: Path to save the plot
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str],
        save_path: Optional[str] = None
    ):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            class_names: Class names
            save_path: Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        if len(class_names) == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            auc = roc_auc_score(y_true, y_proba[:, 1])
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        else:  # Multi-class
            for i, class_name in enumerate(class_names):
                y_true_binary = (y_true == i).astype(int)
                fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, i])
                auc = roc_auc_score(y_true_binary, y_proba[:, i])
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class ConfigManager:
    """Configuration management utilities."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], save_path: str):
        """Save configuration to YAML file."""
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries."""
        merged = base_config.copy()
        
        def update_nested(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    update_nested(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        update_nested(merged, override_config)
        return merged


class Timer:
    """Simple timer for performance measurement."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
    
    def stop(self):
        """Stop the timer."""
        self.end_time = time.time()
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
        if mode == 'min':
            self.is_better = lambda score, best: score < best - min_delta
        else:
            self.is_better = lambda score, best: score > best + min_delta
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current score (loss or metric)
            model: Model to save weights from
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save model checkpoint."""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup file handler if specified
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True
    )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }


def test_utils():
    """Test utility functions."""
    print("Testing utility functions...")
    
    # Test device manager
    device = DeviceManager.get_device()
    memory_usage = DeviceManager.get_memory_usage()
    print(f"Memory usage: {memory_usage}")
    
    # Test metrics calculator
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    y_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8], [0.7, 0.3], [0.9, 0.1]])
    
    metrics = MetricsCalculator.calculate_metrics(y_true, y_pred, y_proba, ['No Fracture', 'Fracture'])
    MetricsCalculator.print_metrics(metrics, "Test Metrics")
    
    # Test timer
    with Timer() as timer:
        time.sleep(0.1)
    print(f"Timer test: {timer.elapsed():.3f} seconds")
    
    print("All utility tests passed!")


if __name__ == "__main__":
    test_utils()