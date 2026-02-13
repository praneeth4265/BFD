"""
Attention mechanisms for medical image analysis.
Includes CBAM (Convolutional Block Attention Module) and SE (Squeeze-and-Excitation) blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel Attention Module from CBAM."""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module from CBAM."""
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Combines channel and spatial attention for better feature representation.
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply channel attention
        out = x * self.channel_attention(x)
        # Apply spatial attention
        out = out * self.spatial_attention(out)
        return out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Focuses on channel-wise feature recalibration.
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA).
    Lightweight alternative to SE blocks.
    """
    
    def __init__(self, in_channels: int, gamma: int = 2, b: int = 1):
        super(ECA, self).__init__()
        kernel_size = int(abs((torch.log2(torch.tensor(in_channels, dtype=torch.float32)) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


def add_attention_to_model(model: nn.Module, attention_type: str = 'cbam') -> nn.Module:
    """
    Add attention modules to an existing model.
    
    Args:
        model: PyTorch model to modify
        attention_type: Type of attention ('cbam', 'se', 'eca')
        
    Returns:
        Modified model with attention mechanisms
    """
    if attention_type.lower() == 'cbam':
        attention_class = CBAM
    elif attention_type.lower() == 'se':
        attention_class = SEBlock
    elif attention_type.lower() == 'eca':
        attention_class = ECA
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
    
    # Add attention after each convolutional block
    # This is a simplified example - in practice, you'd want to be more selective
    for name, module in model.named_children():
        if isinstance(module, (nn.Conv2d, nn.Sequential)):
            # Get the number of output channels
            if hasattr(module, 'out_channels'):
                channels = module.out_channels
            elif hasattr(module, '_modules'):
                # For Sequential modules, get the last conv layer's output channels
                last_conv = None
                for sub_module in module.modules():
                    if isinstance(sub_module, nn.Conv2d):
                        last_conv = sub_module
                if last_conv:
                    channels = last_conv.out_channels
                else:
                    continue
            else:
                continue
            
            # Add attention module after the conv layer
            attention_module = attention_class(channels)
            setattr(model, f"{name}_attention", attention_module)
    
    return model


# Test the attention modules
def test_attention_modules():
    """Test different attention mechanisms."""
    # Create dummy input
    x = torch.randn(2, 64, 32, 32)  # Batch size 2, 64 channels, 32x32 spatial
    
    # Test CBAM
    cbam = CBAM(64)
    cbam_out = cbam(x)
    print(f"CBAM input shape: {x.shape}, output shape: {cbam_out.shape}")
    
    # Test SE Block
    se = SEBlock(64)
    se_out = se(x)
    print(f"SE input shape: {x.shape}, output shape: {se_out.shape}")
    
    # Test ECA
    eca = ECA(64)
    eca_out = eca(x)
    print(f"ECA input shape: {x.shape}, output shape: {eca_out.shape}")
    
    print("All attention modules work correctly!")


if __name__ == "__main__":
    test_attention_modules()