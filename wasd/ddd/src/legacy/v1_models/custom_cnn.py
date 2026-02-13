"""
Custom CNN architectures for bone fracture detection.
Includes lightweight and standard CNN models optimized for medical imaging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .attention import CBAM, SEBlock, ECA


class BoneFractureCNN(nn.Module):
    """
    Custom CNN architecture for bone fracture detection.
    Designed specifically for medical X-ray image analysis.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        use_attention: bool = True,
        attention_type: str = 'cbam',
        dropout_rate: float = 0.3
    ):
        super(BoneFractureCNN, self).__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fourth convolutional block
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fifth convolutional block
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Attention mechanism
        if use_attention:
            if attention_type.lower() == 'cbam':
                self.attention = CBAM(512)
            elif attention_type.lower() == 'se':
                self.attention = SEBlock(512)
            elif attention_type.lower() == 'eca':
                self.attention = ECA(512)
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate / 4),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional features
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x


class CompactBoneCNN(nn.Module):
    """
    Compact version of the bone fracture CNN for faster inference.
    Optimized for deployment scenarios with limited computational resources.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        use_attention: bool = True,
        dropout_rate: float = 0.2
    ):
        super(CompactBoneCNN, self).__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Compact convolutional blocks with depthwise separable convolutions
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Depthwise separable convolution block
        self.conv2 = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # Pointwise convolution
            nn.Conv2d(16, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Pointwise convolution
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv4 = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Pointwise convolution
            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Final feature extraction
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Lightweight attention
        if use_attention:
            self.attention = SEBlock(256, reduction_ratio=8)  # Lighter attention
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Compact classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional features
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x


def create_bone_cnn(
    model_type: str = 'standard',
    num_classes: int = 2,
    use_attention: bool = True,
    attention_type: str = 'cbam',
    dropout_rate: float = 0.3
) -> nn.Module:
    """
    Factory function to create bone fracture CNN models.
    
    Args:
        model_type: Type of model ('standard', 'compact')
        num_classes: Number of output classes
        use_attention: Whether to use attention mechanisms
        attention_type: Type of attention ('cbam', 'se', 'eca')
        dropout_rate: Dropout rate for regularization
        
    Returns:
        CNN model instance
    """
    if model_type.lower() == 'standard':
        return BoneFractureCNN(
            num_classes=num_classes,
            use_attention=use_attention,
            attention_type=attention_type,
            dropout_rate=dropout_rate
        )
    elif model_type.lower() == 'compact':
        return CompactBoneCNN(
            num_classes=num_classes,
            use_attention=use_attention,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def test_cnn_models():
    """Test CNN model creation and forward pass."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test standard model
    print("Testing standard CNN...")
    standard_model = create_bone_cnn('standard', num_classes=2, use_attention=True)
    standard_model.to(device)
    
    # Test compact model
    print("Testing compact CNN...")
    compact_model = create_bone_cnn('compact', num_classes=2, use_attention=True)
    compact_model.to(device)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    with torch.no_grad():
        standard_output = standard_model(dummy_input)
        compact_output = compact_model(dummy_input)
    
    print(f"Standard model output shape: {standard_output.shape}")
    print(f"Compact model output shape: {compact_output.shape}")
    
    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    standard_params = count_parameters(standard_model)
    compact_params = count_parameters(compact_model)
    
    print(f"Standard model parameters: {standard_params:,}")
    print(f"Compact model parameters: {compact_params:,}")
    print(f"Parameter reduction: {(1 - compact_params/standard_params)*100:.1f}%")
    
    print("CNN model tests completed successfully!")


if __name__ == "__main__":
    test_cnn_models()