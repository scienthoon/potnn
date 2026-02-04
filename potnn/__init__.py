"""potnn: Power-of-Two Neural Network Compiler for Ultra-Low-Cost MCUs

A PyTorch-based library for training and deploying neural networks
on MCUs without multiplication instructions, using only shifts and adds.
"""

__version__ = "0.4.8"

# Import core modules
from .modules.conv import PoTConv2d
from .modules.conv1d import PoTConv1d
from .modules.depthwise import PoTDepthwiseConv2d
from .modules.linear import PoTLinear
from .modules.add import PoTAdd
from .modules.avgpool import PoTGlobalAvgPool
from .config import Config
from .export import export
from .quantize.calibration import calibrate_model
from .quantize.qat import prepare_qat, enable_integer_sim, disable_integer_sim
from .wrapper import train
from .fuse import fuse_batchnorm, check_bn_fused


def calibrate(model, data_loader, config=None, num_batches=10, mean=None, std=None):
    """Calibrate model activation scales.
    
    Two calling conventions supported:
    
    1. With config (recommended):
       calibrate(model, loader, config, num_batches=10)
       
    2. Direct parameters:
       calibrate(model, loader, num_batches=10, mean=[0.1307], std=[0.3081])
    
    Args:
        model: Model with PoT layers
        data_loader: Calibration data loader
        config: potnn.Config object (optional, extracts mean/std from it)
        num_batches: Number of batches for calibration (default: 10)
        mean: Dataset mean (list or float), used if config not provided
        std: Dataset std (list or float), used if config not provided
    
    Returns:
        Dictionary of activation max values per layer
    """
    # Handle config object
    if config is not None and isinstance(config, Config):
        mean = config.mean if config.mean is not None else [0.0]
        std = config.std if config.std is not None else [1.0]
    
    # Handle num_batches passed as config (common mistake)
    if isinstance(config, int):
        num_batches = config
        config = None
    
    # Default values
    if mean is None:
        mean = 0.0
    if std is None:
        std = 1.0
    
    return calibrate_model(model, data_loader, num_batches=num_batches, mean=mean, std=std)


__all__ = [
    'PoTConv2d',
    'PoTConv1d',
    'PoTDepthwiseConv2d',
    'PoTLinear',
    'PoTAdd',
    'PoTGlobalAvgPool',
    'Config',
    'export',
    'calibrate_model',
    'calibrate',
    'prepare_qat',
    'enable_integer_sim',
    'disable_integer_sim',
    'train',
    'fuse_batchnorm',
    'check_bn_fused',
]

# Package metadata
__author__ = "potnn developers"
__license__ = "MIT"