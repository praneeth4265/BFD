"""
Comprehensive evaluation and explainability tools for bone fracture detection.
Includes Grad-CAM, model analysis, and medical imaging specific metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, XGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from typing import List, Dict, Any, Optional, Tuple, Union
import yaml
from pathlib import Path
import pandas as pd
from PIL import Image

from utils import MetricsCalculator, Visualizer, ConfigManager
from preprocessing import BoneImagePreprocessor


class GradCAMVisualizer:
    """
    Grad-CAM visualization for medical image analysis.
    Provides localization heatmaps for fracture detection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layers: List[str] = None,
        use_cuda: bool = True
    ):
        """
        Initialize Grad-CAM visualizer.
        
        Args:
            model: Trained model
            target_layers: List of layer names for visualization
            use_cuda: Whether to use CUDA
        """
        self.model = model
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Automatically detect target layers if not provided
        if target_layers is None:
            self.target_layers = self._find_target_layers()
        else:
            self.target_layers = [self._get_layer_by_name(name) for name in target_layers]
        
        # Initialize Grad-CAM
        self.grad_cam = GradCAM(
            model=self.model,
            target_layers=self.target_layers,
            use_cuda=use_cuda
        )
        
        self.grad_cam_plus = GradCAMPlusPlus(
            model=self.model,
            target_layers=self.target_layers,
            use_cuda=use_cuda
        )
    
    def _find_target_layers(self) -> List[nn.Module]:
        """Automatically find suitable target layers."""
        target_layers = []
        
        # Look for the last convolutional layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Sequential)):
                # Check if this layer has convolutional operations
                has_conv = False
                if isinstance(module, nn.Conv2d):
                    has_conv = True
                elif isinstance(module, nn.Sequential):
                    for sub_module in module.modules():
                        if isinstance(sub_module, nn.Conv2d):
                            has_conv = True
                            break
                
                if has_conv:
                    target_layers.append(module)
        
        # Return the last few convolutional layers
        return target_layers[-2:] if len(target_layers) > 2 else target_layers
    
    def _get_layer_by_name(self, layer_name: str) -> nn.Module:
        """Get layer by name."""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer {layer_name} not found in model")
    
    def generate_heatmap(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
        method: str = 'gradcam'
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            image: Input image tensor [1, C, H, W]
            target_class: Target class for visualization (None for predicted class)
            method: Visualization method ('gradcam', 'gradcam++')
            
        Returns:
            Heatmap as numpy array
        """
        # Ensure image is on correct device
        image = image.to(self.device)
        
        # Get prediction if target_class is not specified
        if target_class is None:
            with torch.no_grad():
                outputs = self.model(image)
                target_class = outputs.argmax(dim=1).item()
        
        # Set up targets
        targets = [ClassifierOutputTarget(target_class)]
        
        # Generate heatmap
        if method.lower() == 'gradcam++':
            cam = self.grad_cam_plus
        else:
            cam = self.grad_cam
        
        grayscale_cam = cam(input_tensor=image, targets=targets)
        
        return grayscale_cam[0]  # Return first (and only) image's heatmap
    
    def visualize_prediction(
        self,
        original_image: np.ndarray,
        preprocessed_image: torch.Tensor,
        class_names: List[str],
        save_path: Optional[str] = None,
        show_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Create comprehensive visualization of model prediction.
        
        Args:
            original_image: Original image as numpy array
            preprocessed_image: Preprocessed image tensor
            class_names: List of class names
            save_path: Path to save visualization
            show_confidence: Whether to show confidence scores
            
        Returns:
            Dictionary with prediction details
        """
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(preprocessed_image)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(dim=1).item()
            confidence = probabilities.max().item()
        
        # Generate heatmaps for all methods
        gradcam_heatmap = self.generate_heatmap(preprocessed_image, method='gradcam')
        gradcam_plus_heatmap = self.generate_heatmap(preprocessed_image, method='gradcam++')
        
        # Prepare original image for overlay
        if len(original_image.shape) == 3:
            rgb_img = cv2.resize(original_image, (224, 224))
            if rgb_img.max() > 1:
                rgb_img = rgb_img.astype(np.float32) / 255.0
        else:
            # Convert grayscale to RGB
            rgb_img = cv2.resize(original_image, (224, 224))
            rgb_img = np.stack([rgb_img] * 3, axis=-1)
            if rgb_img.max() > 1:
                rgb_img = rgb_img.astype(np.float32) / 255.0
        
        # Create visualizations
        gradcam_viz = show_cam_on_image(rgb_img, gradcam_heatmap, use_rgb=True)
        gradcam_plus_viz = show_cam_on_image(rgb_img, gradcam_plus_heatmap, use_rgb=True)
        
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(rgb_img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Grad-CAM heatmap
        axes[0, 1].imshow(gradcam_heatmap, cmap='jet')
        axes[0, 1].set_title('Grad-CAM Heatmap')
        axes[0, 1].axis('off')
        
        # Grad-CAM overlay
        axes[0, 2].imshow(gradcam_viz)
        axes[0, 2].set_title('Grad-CAM Overlay')
        axes[0, 2].axis('off')
        
        # Grad-CAM++ heatmap
        axes[1, 0].imshow(gradcam_plus_heatmap, cmap='jet')
        axes[1, 0].set_title('Grad-CAM++ Heatmap')
        axes[1, 0].axis('off')
        
        # Grad-CAM++ overlay
        axes[1, 1].imshow(gradcam_plus_viz)
        axes[1, 1].set_title('Grad-CAM++ Overlay')
        axes[1, 1].axis('off')
        
        # Prediction confidence
        axes[1, 2].bar(class_names, probabilities.cpu().numpy()[0])
        axes[1, 2].set_title('Prediction Confidence')
        axes[1, 2].set_ylabel('Probability')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        # Add prediction text
        prediction_text = f"Predicted: {class_names[predicted_class]}"
        if show_confidence:
            prediction_text += f"\nConfidence: {confidence:.3f}"
        
        fig.suptitle(prediction_text, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return {
            'predicted_class': predicted_class,
            'predicted_label': class_names[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy()[0],
            'gradcam_heatmap': gradcam_heatmap,
            'gradcam_plus_heatmap': gradcam_plus_heatmap
        }


class ModelEvaluator:
    """
    Comprehensive model evaluation for medical imaging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config_path: str = None,
        class_names: List[str] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            config_path: Path to configuration file
            class_names: List of class names
        """
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        if config_path:
            self.config = ConfigManager.load_config(config_path)
        else:
            self.config = {}
        
        self.class_names = class_names or ['No Fracture', 'Fracture']
        
        # Initialize Grad-CAM visualizer
        self.grad_cam_viz = GradCAMVisualizer(self.model)
    
    def evaluate_dataset(
        self,
        data_loader,
        save_dir: str = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation on a dataset.
        
        Args:
            data_loader: Data loader for evaluation
            save_dir: Directory to save evaluation results
            
        Returns:
            Dictionary with evaluation results
        """
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_proba = np.array(all_probabilities)
        
        # Calculate comprehensive metrics
        metrics = MetricsCalculator.calculate_metrics(
            y_true, y_pred, y_proba, self.class_names
        )
        
        # Create evaluation report
        results = {
            'metrics': {
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'auc_roc': metrics.auc_roc
            },
            'confusion_matrix': metrics.confusion_matrix,
            'classification_report': metrics.classification_report,
            'raw_predictions': {
                'y_true': y_true,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
        }
        
        # Save results if directory provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            with open(save_dir / 'evaluation_metrics.yaml', 'w') as f:
                yaml.dump(results['metrics'], f)
            
            # Save confusion matrix plot
            Visualizer.plot_confusion_matrix(
                metrics.confusion_matrix,
                self.class_names,
                save_path=str(save_dir / 'confusion_matrix.png')
            )
            
            # Save ROC curve
            if len(self.class_names) == 2:
                Visualizer.plot_roc_curve(
                    y_true, y_proba, self.class_names,
                    save_path=str(save_dir / 'roc_curve.png')
                )
            
            # Save detailed report
            with open(save_dir / 'classification_report.txt', 'w') as f:
                f.write(metrics.classification_report)
        
        return results
    
    def analyze_misclassifications(
        self,
        data_loader,
        num_examples: int = 10,
        save_dir: str = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze misclassified examples with Grad-CAM.
        
        Args:
            data_loader: Data loader
            num_examples: Number of misclassified examples to analyze
            save_dir: Directory to save analysis
            
        Returns:
            List of misclassification analysis results
        """
        misclassified_examples = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                # Find misclassified examples
                misclassified_mask = predictions != labels
                
                if misclassified_mask.any():
                    misclassified_images = images[misclassified_mask]
                    misclassified_labels = labels[misclassified_mask]
                    misclassified_preds = predictions[misclassified_mask]
                    
                    for i in range(len(misclassified_images)):
                        if len(misclassified_examples) >= num_examples:
                            break
                        
                        # Get single image
                        img_tensor = misclassified_images[i:i+1]
                        true_label = misclassified_labels[i].item()
                        pred_label = misclassified_preds[i].item()
                        
                        # Convert to original image format for visualization
                        original_img = self._tensor_to_image(img_tensor[0])
                        
                        # Generate Grad-CAM visualization
                        result = self.grad_cam_viz.visualize_prediction(
                            original_img,
                            img_tensor,
                            self.class_names,
                            save_path=f"{save_dir}/misclassified_{len(misclassified_examples)}.png" if save_dir else None,
                            show_confidence=True
                        )
                        
                        misclassified_examples.append({
                            'true_label': true_label,
                            'true_class': self.class_names[true_label],
                            'predicted_label': pred_label,
                            'predicted_class': self.class_names[pred_label],
                            'confidence': result['confidence'],
                            'batch_idx': batch_idx,
                            'image_idx': i
                        })
                
                if len(misclassified_examples) >= num_examples:
                    break
        
        return misclassified_examples
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to image for visualization."""
        # Denormalize if needed
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        img = tensor.cpu().numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        return img
    
    def generate_activation_maps(
        self,
        image: torch.Tensor,
        save_dir: str = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate activation maps for different layers.
        
        Args:
            image: Input image tensor
            save_dir: Directory to save maps
            
        Returns:
            Dictionary of activation maps
        """
        activation_maps = {}
        
        # Hook function to capture activations
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        activations = {}
        
        # Register hooks for convolutional layers
        handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                handle = module.register_forward_hook(get_activation(name))
                handles.append(handle)
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(image)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Process activations
        for name, activation in activations.items():
            # Take mean across channels for visualization
            activation_map = activation.mean(dim=1).squeeze().cpu().numpy()
            activation_maps[name] = activation_map
            
            if save_dir:
                plt.figure(figsize=(8, 6))
                plt.imshow(activation_map, cmap='viridis')
                plt.title(f'Activation Map: {name}')
                plt.colorbar()
                plt.axis('off')
                plt.savefig(f"{save_dir}/activation_{name.replace('.', '_')}.png",
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        return activation_maps


def evaluate_model_performance(
    model: nn.Module,
    test_loader,
    config_path: str = None,
    save_dir: str = None
) -> Dict[str, Any]:
    """
    Main function for comprehensive model evaluation.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        config_path: Path to configuration file
        save_dir: Directory to save results
        
    Returns:
        Complete evaluation results
    """
    # Initialize evaluator
    evaluator = ModelEvaluator(model, config_path)
    
    # Run comprehensive evaluation
    results = evaluator.evaluate_dataset(test_loader, save_dir)
    
    # Analyze misclassifications
    if save_dir:
        misclass_dir = Path(save_dir) / 'misclassifications'
        misclass_dir.mkdir(exist_ok=True)
        
        misclassified = evaluator.analyze_misclassifications(
            test_loader,
            num_examples=10,
            save_dir=str(misclass_dir)
        )
        
        results['misclassifications'] = misclassified
    
    return results


def test_evaluation():
    """Test evaluation functionality."""
    from models import create_bone_cnn
    
    # Create dummy model
    model = create_bone_cnn('compact', num_classes=2)
    
    # Create dummy data
    dummy_images = torch.randn(4, 3, 224, 224)
    dummy_labels = torch.tensor([0, 1, 0, 1])
    
    # Test Grad-CAM
    grad_cam_viz = GradCAMVisualizer(model)
    
    # Generate heatmap for single image
    heatmap = grad_cam_viz.generate_heatmap(dummy_images[0:1])
    print(f"Heatmap shape: {heatmap.shape}")
    
    # Test evaluation
    evaluator = ModelEvaluator(model)
    
    # Simulate evaluation (would normally use real data loader)
    print("Evaluation test completed successfully!")


if __name__ == "__main__":
    test_evaluation()