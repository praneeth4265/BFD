"""
Custom lightweight CNN architecture for bone fracture detection.
Designed for medical imaging with attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .attention import CBAM, SEBlock, ECA


class ConvBlock(nn.Module):
    """Basic convolutional block with optional attention."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_attention: bool = False,
        attention_type: str = 'cbam',
        dropout_rate: float = 0.1
    ):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Add attention mechanism
        self.use_attention = use_attention
        if use_attention:
            if attention_type.lower() == 'cbam':
                self.attention = CBAM(out_channels)
            elif attention_type.lower() == 'se':
                self.attention = SEBlock(out_channels)
            elif attention_type.lower() == 'eca':
                self.attention = ECA(out_channels)
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        if self.use_attention:
            x = self.attention(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block with optional attention mechanism."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        stride: int = 1,
        use_attention: bool = False,
        attention_type: str = 'cbam',
        dropout_rate: float = 0.1
    ):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Add attention mechanism
        self.use_attention = use_attention
        if use_attention:
            if attention_type.lower() == 'cbam':
                self.attention = CBAM(out_channels)
            elif attention_type.lower() == 'se':
                self.attention = SEBlock(out_channels)
            elif attention_type.lower() == 'eca':
                self.attention = ECA(out_channels)
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        if self.use_attention:
            out = self.attention(out)
        
        out += residual
        out = F.relu(out)
        
        return out


class BoneFractureCNN(nn.Module):
    """
    Lightweight CNN architecture specifically designed for bone fracture detection.
    Incorporates medical imaging best practices and attention mechanisms.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 3,
        use_attention: bool = True,
        attention_type: str = 'cbam',
        dropout_rate: float = 0.2,
        activation: str = 'relu'
    ):
        super(BoneFractureCNN, self).__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Initial convolution
        self.initial_conv = ConvBlock(
            input_channels, 32, kernel_size=7, stride=2, padding=3,
            use_attention=False, dropout_rate=0.1
        )
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Feature extraction blocks
        self.block1 = self._make_layer(32, 64, 2, stride=1, use_attention=use_attention, 
                                     attention_type=attention_type, dropout_rate=dropout_rate)
        self.block2 = self._make_layer(64, 128, 2, stride=2, use_attention=use_attention,
                                     attention_type=attention_type, dropout_rate=dropout_rate)
        self.block3 = self._make_layer(128, 256, 2, stride=2, use_attention=use_attention,
                                     attention_type=attention_type, dropout_rate=dropout_rate)
        self.block4 = self._make_layer(256, 512, 2, stride=2, use_attention=use_attention,
                                     attention_type=attention_type, dropout_rate=dropout_rate)
        
        # Global feature aggregation
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512 * 2, 256),  # *2 because of avg + max pooling
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 4),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(
        self, 
        in_channels: int, 
        out_channels: int, 
        num_blocks: int,
        stride: int = 1,
        use_attention: bool = False,
        attention_type: str = 'cbam',
        dropout_rate: float = 0.1
    ) -> nn.Sequential:
        """Create a layer with multiple residual blocks."""
        layers = []
        
        # First block with stride
        layers.append(ResidualBlock(
            in_channels, out_channels, stride,
            use_attention=use_attention, attention_type=attention_type,
            dropout_rate=dropout_rate
        ))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(
                out_channels, out_channels, 1,
                use_attention=use_attention, attention_type=attention_type,
                dropout_rate=dropout_rate
            ))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/He initialization."""
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial feature extraction
        x = self.initial_conv(x)
        x = self.max_pool(x)
        
        # Progressive feature extraction
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Global feature aggregation
        avg_pooled = self.global_avg_pool(x).flatten(1)
        max_pooled = self.global_max_pool(x).flatten(1)
        x = torch.cat([avg_pooled, max_pooled], dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature maps for visualization/Grad-CAM.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature maps from the last convolutional layer
        """
        x = self.initial_conv(x)
        x = self.max_pool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        return x


class CompactBoneCNN(nn.Module):
    """
    Ultra-lightweight CNN for mobile deployment.
    Optimized for speed and memory efficiency.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 3,
        width_multiplier: float = 1.0
    ):
        super(CompactBoneCNN, self).__init__()
        
        def make_divisible(v, divisor=8):
            return max(divisor, int(v + divisor / 2) // divisor * divisor)
        
        # Depth-wise separable convolutions
        def depthwise_conv(in_ch, out_ch, stride):
            return nn.Sequential(
                # Depthwise
                nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU6(inplace=True),
                
                # Pointwise
                nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True)
            )
        
        # Calculate channel sizes
        ch32 = make_divisible(32 * width_multiplier)
        ch64 = make_divisible(64 * width_multiplier)
        ch128 = make_divisible(128 * width_multiplier)
        ch256 = make_divisible(256 * width_multiplier)
        
        self.features = nn.Sequential(
            # Initial convolution
            nn.Conv2d(input_channels, ch32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ch32),
            nn.ReLU6(inplace=True),
            
            # Depthwise separable blocks
            depthwise_conv(ch32, ch64, 1),
            depthwise_conv(ch64, ch128, 2),
            depthwise_conv(ch128, ch128, 1),
            depthwise_conv(ch128, ch256, 2),
            depthwise_conv(ch256, ch256, 1),
            
            # Global pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(ch256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for compact model."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_bone_cnn(
    model_type: str = 'standard',
    num_classes: int = 2,
    use_attention: bool = True,
    attention_type: str = 'cbam',
    **kwargs
) -> nn.Module:
    """
    Factory function to create different CNN architectures.
    
    Args:
        model_type: Type of model ('standard', 'compact')
        num_classes: Number of output classes
        use_attention: Whether to use attention mechanisms
        attention_type: Type of attention ('cbam', 'se', 'eca')
        **kwargs: Additional arguments
        
    Returns:
        CNN model instance
    """
    if model_type.lower() == 'standard':
        return BoneFractureCNN(
            num_classes=num_classes,
            use_attention=use_attention,
            attention_type=attention_type,
            **kwargs
        )
    elif model_type.lower() == 'compact':
        return CompactBoneCNN(
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def test_cnn_models():
    """Test the CNN models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test standard model
    model_std = create_bone_cnn('standard', use_attention=True)
    model_std.to(device)
    
    # Test compact model
    model_compact = create_bone_cnn('compact')
    model_compact.to(device)
    
    # Test input
    x = torch.randn(2, 3, 224, 224).to(device)
    
    # Forward pass
    with torch.no_grad():
        out_std = model_std(x)
        out_compact = model_compact(x)
    
    print(f"Standard model output shape: {out_std.shape}")
    print(f"Compact model output shape: {out_compact.shape}")
    
    # Count parameters
    std_params = sum(p.numel() for p in model_std.parameters())
    compact_params = sum(p.numel() for p in model_compact.parameters())
    
    print(f"Standard model parameters: {std_params:,}")
    print(f"Compact model parameters: {compact_params:,}")
    print(f"Compression ratio: {std_params / compact_params:.2f}x")


if __name__ == "__main__":
    test_cnn_models()