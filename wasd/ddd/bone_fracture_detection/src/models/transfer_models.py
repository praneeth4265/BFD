"""
Transfer learning models using pre-trained architectures.
Includes EfficientNet, ResNet, and MobileNet with medical imaging adaptations.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Dict, Any
import warnings

from .attention import CBAM, SEBlock, ECA


class TransferLearningModel(nn.Module):
    """
    Base class for transfer learning models with medical imaging adaptations.
    """
    
    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        use_attention: bool = False,
        attention_type: str = 'cbam',
        dropout_rate: float = 0.3,
        input_channels: int = 3
    ):
        super(TransferLearningModel, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Load pre-trained model
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
                global_pool=''  # Remove global pooling
            )
        except Exception as e:
            raise ValueError(f"Failed to load model {model_name}: {e}")
        
        # Handle different input channels
        if input_channels != 3:
            self._adapt_input_channels(input_channels)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        
        # Get feature dimension
        self.feature_dim = self._get_feature_dim()
        
        # Add attention mechanism
        if use_attention:
            if attention_type.lower() == 'cbam':
                self.attention = CBAM(self.feature_dim)
            elif attention_type.lower() == 'se':
                self.attention = SEBlock(self.feature_dim)
            elif attention_type.lower() == 'eca':
                self.attention = ECA(self.feature_dim)
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Custom classifier for medical imaging
        self.classifier = self._build_classifier(dropout_rate)
        
        # Initialize new layers
        self._initialize_new_layers()
    
    def _adapt_input_channels(self, input_channels: int):
        """Adapt the first layer for different input channels."""
        # This is a simplified adaptation - might need model-specific handling
        first_conv = None
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                first_conv = module
                break
        
        if first_conv is not None and first_conv.in_channels != input_channels:
            # Create new first layer
            new_conv = nn.Conv2d(
                input_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
            
            # Initialize weights
            if input_channels == 1:  # Grayscale
                # Average the RGB weights
                with torch.no_grad():
                    new_conv.weight = nn.Parameter(first_conv.weight.mean(dim=1, keepdim=True))
            else:
                # Random initialization for other cases
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            
            # Replace the first conv layer
            # This is model-specific and might need adjustment
            setattr(self.backbone, list(self.backbone.named_children())[0][0], new_conv)
    
    def _freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _get_feature_dim(self) -> int:
        """Get the feature dimension of the backbone."""
        # Create a dummy input to determine feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            return features.shape[1]
    
    def _build_classifier(self, dropout_rate: float) -> nn.Module:
        """Build a medical imaging specific classifier."""
        return nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.feature_dim // 2),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.feature_dim // 4),
            nn.Dropout(dropout_rate / 4),
            nn.Linear(self.feature_dim // 4, self.num_classes)
        )
    
    def _initialize_new_layers(self):
        """Initialize newly added layers."""
        for module in [self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.backbone(x)
        
        # Apply attention if enabled
        if self.use_attention:
            features = self.attention(features)
        
        # Global pooling
        features = self.global_pool(features)
        features = torch.flatten(features, 1)
        
        # Classification
        output = self.classifier(features)
        
        return output
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature maps for Grad-CAM visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature maps from backbone
        """
        return self.backbone(x)
    
    def unfreeze_backbone(self, unfreeze_layers: Optional[int] = None):
        """
        Unfreeze backbone layers for fine-tuning.
        
        Args:
            unfreeze_layers: Number of layers to unfreeze from the end. 
                           If None, unfreezes all layers.
        """
        backbone_params = list(self.backbone.parameters())
        
        if unfreeze_layers is None:
            # Unfreeze all
            for param in backbone_params:
                param.requires_grad = True
        else:
            # Unfreeze last N layers
            for param in backbone_params[-unfreeze_layers:]:
                param.requires_grad = True


class EfficientNetModel(TransferLearningModel):
    """EfficientNet model for bone fracture detection."""
    
    def __init__(
        self,
        model_variant: str = 'efficientnet_b0',
        num_classes: int = 2,
        pretrained: bool = True,
        **kwargs
    ):
        super().__init__(
            model_name=model_variant,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )


class ResNetModel(TransferLearningModel):
    """ResNet model for bone fracture detection."""
    
    def __init__(
        self,
        model_variant: str = 'resnet50',
        num_classes: int = 2,
        pretrained: bool = True,
        **kwargs
    ):
        super().__init__(
            model_name=model_variant,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )


class MobileNetModel(TransferLearningModel):
    """MobileNet model for bone fracture detection."""
    
    def __init__(
        self,
        model_variant: str = 'mobilenetv3_small_100',
        num_classes: int = 2,
        pretrained: bool = True,
        **kwargs
    ):
        super().__init__(
            model_name=model_variant,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )


class VisionTransformerModel(TransferLearningModel):
    """Vision Transformer model for bone fracture detection."""
    
    def __init__(
        self,
        model_variant: str = 'vit_small_patch16_224',
        num_classes: int = 2,
        pretrained: bool = True,
        **kwargs
    ):
        super().__init__(
            model_name=model_variant,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    
    def _build_classifier(self, dropout_rate: float) -> nn.Module:
        """Build classifier specific for ViT."""
        return nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(self.feature_dim // 2, self.num_classes)
        )


def create_transfer_model(
    architecture: str,
    variant: str = None,
    num_classes: int = 2,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create transfer learning models.
    
    Args:
        architecture: Model architecture ('efficientnet', 'resnet', 'mobilenet', 'vit')
        variant: Specific model variant
        num_classes: Number of output classes
        pretrained: Whether to use pre-trained weights
        **kwargs: Additional arguments
        
    Returns:
        Transfer learning model instance
    """
    architecture = architecture.lower()
    
    if architecture == 'efficientnet':
        variant = variant or 'efficientnet_b0'
        return EfficientNetModel(variant, num_classes, pretrained, **kwargs)
    
    elif architecture == 'resnet':
        variant = variant or 'resnet50'
        return ResNetModel(variant, num_classes, pretrained, **kwargs)
    
    elif architecture == 'mobilenet':
        variant = variant or 'mobilenetv3_small_100'
        return MobileNetModel(variant, num_classes, pretrained, **kwargs)
    
    elif architecture == 'vit':
        variant = variant or 'vit_small_patch16_224'
        return VisionTransformerModel(variant, num_classes, pretrained, **kwargs)
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get comprehensive information about a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Model size estimation (MB)
    param_size = total_params * 4  # Assuming float32
    buffer_size = sum(b.numel() * 4 for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024 / 1024
    
    # FLOPs estimation (simplified)
    def count_conv_flops(module, input, output):
        if isinstance(module, nn.Conv2d):
            kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            output_elements = output.numel()
            return kernel_flops * output_elements
        return 0
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params,
        'model_size_mb': model_size_mb,
        'architecture': model.__class__.__name__
    }


def test_transfer_models():
    """Test transfer learning models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different architectures
    architectures = ['efficientnet', 'resnet', 'mobilenet']
    
    for arch in architectures:
        print(f"\nTesting {arch}...")
        
        try:
            model = create_transfer_model(
                arch, 
                num_classes=2, 
                pretrained=True,
                use_attention=True,
                attention_type='cbam'
            )
            model.to(device)
            
            # Test forward pass
            x = torch.randn(2, 3, 224, 224).to(device)
            with torch.no_grad():
                output = model(x)
            
            print(f"  Output shape: {output.shape}")
            
            # Get model info
            info = get_model_info(model)
            print(f"  Parameters: {info['total_parameters']:,}")
            print(f"  Model size: {info['model_size_mb']:.2f} MB")
            
            # Test feature extraction
            features = model.get_feature_maps(x)
            print(f"  Feature maps shape: {features.shape}")
            
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    test_transfer_models()