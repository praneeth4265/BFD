"""
Ensemble model combining multiple architectures for robust bone fracture detection.
Supports soft voting, hard voting, and weighted ensemble methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union
import numpy as np
from collections import OrderedDict

from .cnn_model import create_bone_cnn
from .transfer_models import create_transfer_model


class EnsembleModel(nn.Module):
    """
    Ensemble model that combines predictions from multiple base models.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        ensemble_method: str = 'soft_voting',
        weights: Optional[List[float]] = None,
        device: str = 'auto'
    ):
        """
        Initialize ensemble model.
        
        Args:
            models: List of base models
            ensemble_method: Ensemble method ('soft_voting', 'hard_voting', 'weighted_voting')
            weights: Weights for each model (used in weighted voting)
            device: Device to run models on
        """
        super(EnsembleModel, self).__init__()
        
        self.ensemble_method = ensemble_method.lower()
        self.num_models = len(models)
        
        if self.num_models == 0:
            raise ValueError("At least one model is required for ensemble")
        
        # Validate ensemble method
        valid_methods = ['soft_voting', 'hard_voting', 'weighted_voting']
        if self.ensemble_method not in valid_methods:
            raise ValueError(f"Invalid ensemble method. Choose from {valid_methods}")
        
        # Store models
        self.models = nn.ModuleList(models)
        
        # Set up weights
        if weights is not None:
            if len(weights) != self.num_models:
                raise ValueError("Number of weights must match number of models")
            weights = torch.tensor(weights, dtype=torch.float32)
            weights = weights / weights.sum()  # Normalize
            self.register_buffer('weights', weights)
        else:
            # Equal weights
            equal_weights = torch.ones(self.num_models) / self.num_models
            self.register_buffer('weights', equal_weights)
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Get number of classes from first model
        self.num_classes = self._get_num_classes()
    
    def _get_num_classes(self) -> int:
        """Get number of classes from the first model."""
        # Create dummy input to get output shape
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            try:
                output = self.models[0](dummy_input)
                return output.shape[1]
            except:
                # Fallback to common case
                return 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Ensemble prediction
        """
        if self.ensemble_method == 'soft_voting':
            return self._soft_voting(x)
        elif self.ensemble_method == 'hard_voting':
            return self._hard_voting(x)
        elif self.ensemble_method == 'weighted_voting':
            return self._weighted_voting(x)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _soft_voting(self, x: torch.Tensor) -> torch.Tensor:
        """Soft voting: average of probabilities."""
        predictions = []
        
        for model in self.models:
            with torch.set_grad_enabled(self.training):
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
        
        # Average probabilities
        ensemble_probs = torch.stack(predictions).mean(dim=0)
        
        # Convert back to logits for loss computation
        ensemble_logits = torch.log(ensemble_probs + 1e-8)
        
        return ensemble_logits
    
    def _hard_voting(self, x: torch.Tensor) -> torch.Tensor:
        """Hard voting: majority vote of predictions."""
        predictions = []
        
        for model in self.models:
            with torch.set_grad_enabled(self.training):
                logits = model(x)
                pred_class = torch.argmax(logits, dim=1)
                predictions.append(pred_class)
        
        # Stack predictions and find mode
        stacked_preds = torch.stack(predictions, dim=0)  # [num_models, batch_size]
        
        # Convert to one-hot for averaging
        batch_size = x.shape[0]
        ensemble_probs = torch.zeros(batch_size, self.num_classes, device=x.device)
        
        for i in range(batch_size):
            class_votes = stacked_preds[:, i]
            for class_idx in range(self.num_classes):
                votes = (class_votes == class_idx).sum().float()
                ensemble_probs[i, class_idx] = votes / self.num_models
        
        # Convert to logits
        ensemble_logits = torch.log(ensemble_probs + 1e-8)
        
        return ensemble_logits
    
    def _weighted_voting(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted voting: weighted average of probabilities."""
        predictions = []
        
        for model in self.models:
            with torch.set_grad_enabled(self.training):
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
        
        # Weighted average
        weighted_probs = torch.zeros_like(predictions[0])
        for i, probs in enumerate(predictions):
            weighted_probs += self.weights[i] * probs
        
        # Convert back to logits
        ensemble_logits = torch.log(weighted_probs + 1e-8)
        
        return ensemble_logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions with ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs = F.softmax(logits, dim=1)
        return probs
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> tuple:
        """
        Make predictions with uncertainty estimation.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (predictions, uncertainty_scores)
        """
        individual_predictions = []
        
        self.eval()
        with torch.no_grad():
            for model in self.models:
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                individual_predictions.append(probs)
        
        # Stack predictions
        all_preds = torch.stack(individual_predictions)  # [num_models, batch_size, num_classes]
        
        # Calculate ensemble prediction
        ensemble_pred = all_preds.mean(dim=0)
        
        # Calculate uncertainty as standard deviation across models
        uncertainty = all_preds.std(dim=0).mean(dim=1)  # [batch_size]
        
        return ensemble_pred, uncertainty
    
    def get_individual_predictions(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get predictions from individual models.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with predictions from each model
        """
        predictions = {}
        
        self.eval()
        with torch.no_grad():
            for i, model in enumerate(self.models):
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                predictions[f'model_{i}'] = probs
        
        return predictions


class AdaptiveEnsemble(EnsembleModel):
    """
    Adaptive ensemble that learns optimal weights during training.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        num_classes: int,
        device: str = 'auto'
    ):
        # Initialize with equal weights first
        super().__init__(models, 'weighted_voting', None, device)
        
        # Learnable weights
        self.adaptive_weights = nn.Parameter(torch.ones(self.num_models))
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with learnable weights."""
        predictions = []
        
        for model in self.models:
            logits = model(x)
            predictions.append(logits)
        
        # Stack predictions
        stacked_logits = torch.stack(predictions, dim=0)  # [num_models, batch_size, num_classes]
        
        # Apply temperature scaling
        stacked_logits = stacked_logits / self.temperature
        
        # Normalize weights with softmax
        normalized_weights = F.softmax(self.adaptive_weights, dim=0)
        
        # Weighted combination
        ensemble_logits = torch.sum(
            normalized_weights.view(-1, 1, 1) * stacked_logits, 
            dim=0
        )
        
        return ensemble_logits


def create_ensemble_from_config(
    config: Dict[str, Any],
    device: str = 'auto'
) -> EnsembleModel:
    """
    Create ensemble model from configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to run on
        
    Returns:
        Ensemble model
    """
    models = []
    
    # Create models based on config
    for model_config in config.get('architectures', []):
        model_name = model_config['name']
        
        if 'efficientnet' in model_name:
            model = create_transfer_model(
                'efficientnet',
                variant=model_name,
                num_classes=config.get('num_classes', 2),
                pretrained=model_config.get('pretrained', True),
                freeze_backbone=model_config.get('freeze_backbone', False)
            )
        elif 'resnet' in model_name:
            model = create_transfer_model(
                'resnet',
                variant=model_name,
                num_classes=config.get('num_classes', 2),
                pretrained=model_config.get('pretrained', True),
                freeze_backbone=model_config.get('freeze_backbone', False)
            )
        elif 'mobilenet' in model_name:
            model = create_transfer_model(
                'mobilenet',
                variant=model_name,
                num_classes=config.get('num_classes', 2),
                pretrained=model_config.get('pretrained', True),
                freeze_backbone=model_config.get('freeze_backbone', False)
            )
        elif model_name == 'custom_cnn':
            model = create_bone_cnn(
                'standard',
                num_classes=config.get('num_classes', 2),
                use_attention=True
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        models.append(model)
    
    # Create ensemble
    ensemble_config = config.get('ensemble', {})
    ensemble_method = ensemble_config.get('voting_method', 'soft')
    weights = ensemble_config.get('weights', None)
    
    if ensemble_method == 'adaptive':
        ensemble = AdaptiveEnsemble(
            models,
            config.get('num_classes', 2),
            device
        )
    else:
        ensemble = EnsembleModel(
            models,
            f"{ensemble_method}_voting",
            weights,
            device
        )
    
    return ensemble


def save_ensemble(ensemble: EnsembleModel, save_path: str):
    """
    Save ensemble model.
    
    Args:
        ensemble: Ensemble model to save
        save_path: Path to save the model
    """
    torch.save({
        'ensemble_method': ensemble.ensemble_method,
        'weights': ensemble.weights,
        'num_models': ensemble.num_models,
        'num_classes': ensemble.num_classes,
        'state_dict': ensemble.state_dict(),
        'model_configs': [model.__class__.__name__ for model in ensemble.models]
    }, save_path)


def load_ensemble(load_path: str, models: List[nn.Module]) -> EnsembleModel:
    """
    Load ensemble model.
    
    Args:
        load_path: Path to load the model from
        models: List of base models (architecture should match saved model)
        
    Returns:
        Loaded ensemble model
    """
    checkpoint = torch.load(load_path, map_location='cpu')
    
    ensemble = EnsembleModel(
        models,
        checkpoint['ensemble_method'],
        checkpoint['weights'].tolist()
    )
    
    ensemble.load_state_dict(checkpoint['state_dict'])
    
    return ensemble


def test_ensemble():
    """Test ensemble functionality."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy models
    models = []
    for i in range(3):
        model = create_bone_cnn('compact', num_classes=2)
        models.append(model)
    
    # Test different ensemble methods
    for method in ['soft_voting', 'hard_voting', 'weighted_voting']:
        print(f"\nTesting {method}...")
        
        ensemble = EnsembleModel(
            models,
            method,
            weights=[0.4, 0.35, 0.25] if method == 'weighted_voting' else None
        )
        ensemble.to(device)
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224).to(device)
        
        with torch.no_grad():
            output = ensemble(x)
            predictions = ensemble.predict(x)
            pred_with_uncertainty = ensemble.predict_with_uncertainty(x)
        
        print(f"  Output shape: {output.shape}")
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Uncertainty shape: {pred_with_uncertainty[1].shape}")
    
    # Test adaptive ensemble
    print("\nTesting adaptive ensemble...")
    adaptive_ensemble = AdaptiveEnsemble(models, num_classes=2)
    adaptive_ensemble.to(device)
    
    with torch.no_grad():
        output = adaptive_ensemble(x)
    
    print(f"  Adaptive output shape: {output.shape}")
    print(f"  Learned weights: {F.softmax(adaptive_ensemble.adaptive_weights, dim=0).detach().cpu().numpy()}")


if __name__ == "__main__":
    test_ensemble()