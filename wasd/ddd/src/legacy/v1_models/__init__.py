"""
Module initialization for models package.
"""

from .attention import CBAM, SEBlock, ECA
from .cnn_model import BoneFractureCNN, CompactBoneCNN, create_bone_cnn
from .transfer_models import (
    TransferLearningModel, 
    EfficientNetModel, 
    ResNetModel, 
    MobileNetModel,
    VisionTransformerModel,
    create_transfer_model,
    get_model_info
)
from .ensemble_model import (
    EnsembleModel,
    AdaptiveEnsemble, 
    create_ensemble_from_config,
    save_ensemble,
    load_ensemble
)

__all__ = [
    # Attention mechanisms
    'CBAM', 'SEBlock', 'ECA',
    
    # Custom CNN models
    'BoneFractureCNN', 'CompactBoneCNN', 'create_bone_cnn',
    
    # Transfer learning models
    'TransferLearningModel', 'EfficientNetModel', 'ResNetModel', 
    'MobileNetModel', 'VisionTransformerModel', 'create_transfer_model',
    'get_model_info',
    
    # Ensemble models
    'EnsembleModel', 'AdaptiveEnsemble', 'create_ensemble_from_config',
    'save_ensemble', 'load_ensemble'
]