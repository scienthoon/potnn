"""BatchNorm fusion into Conv/Linear layers."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


def fuse_batchnorm(model: nn.Module) -> nn.Module:
    """Fuse BatchNorm layers into preceding Conv/Linear layers.
    
    This absorbs BatchNorm parameters (γ, β, μ, σ) into the weight and bias
    of the preceding convolution or linear layer, eliminating the need for
    separate BatchNorm computation at inference time.
    
    Formula:
        y = γ * (x - μ) / √(σ² + ε) + β
        
    For Conv/Linear followed by BatchNorm:
        out = BN(W*x + b)
            = scale * (W*x + b) + bias'
            = (scale * W) * x + (scale * b + bias')
        
        where:
            scale = γ / √(σ² + ε)
            bias' = β - γ * μ / √(σ² + ε)
        
        Therefore:
            W_fused = W * scale
            b_fused = b * scale + bias' = (b - μ) * scale + β
    
    Args:
        model: Model with Conv/Linear + BatchNorm sequences
        
    Returns:
        Model with BatchNorm fused (BatchNorm layers become identity)
    """
    print("Fusing BatchNorm layers...")
    
    # Find Conv/Linear -> BatchNorm pairs
    pairs = _find_bn_pairs(model)
    
    if not pairs:
        print("  No BatchNorm layers found to fuse.")
        return model
    
    # Fuse each pair
    for conv_name, bn_name, conv_module, bn_module in pairs:
        _fuse_single_bn(conv_module, bn_module)
        print(f"  Fused: {conv_name} <- {bn_name}")
    
    # Replace BatchNorm layers with Identity
    _replace_bn_with_identity(model, [bn_name for _, bn_name, _, _ in pairs])
    
    print(f"  Total {len(pairs)} BatchNorm layers fused.")
    
    return model


def _find_bn_pairs(model: nn.Module) -> List[Tuple[str, str, nn.Module, nn.Module]]:
    """Find Conv/Linear -> BatchNorm pairs in the model.
    
    Returns:
        List of (conv_name, bn_name, conv_module, bn_module) tuples
    """
    pairs = []
    prev_name = None
    prev_module = None
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            # Check if previous layer is Conv or Linear
            if prev_module is not None:
                if isinstance(prev_module, (nn.Conv2d, nn.Linear)):
                    pairs.append((prev_name, name, prev_module, module))
                elif hasattr(prev_module, 'weight'):
                    # PoTConv2d or PoTLinear
                    pairs.append((prev_name, name, prev_module, module))
        
        # Track previous layer (skip non-compute layers)
        if isinstance(module, (nn.Conv2d, nn.Linear)) or hasattr(module, 'weight'):
            if not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                prev_name = name
                prev_module = module
    
    return pairs


def _fuse_single_bn(conv: nn.Module, bn: nn.Module):
    """Fuse a single BatchNorm into its preceding Conv/Linear.
    
    Modifies conv.weight and conv.bias in-place.
    """
    with torch.no_grad():
        # Get BatchNorm parameters
        gamma = bn.weight  # γ (scale)
        beta = bn.bias     # β (shift)
        mean = bn.running_mean  # μ
        var = bn.running_var    # σ²
        eps = bn.eps
        
        # Compute scale factor: γ / √(σ² + ε)
        std = torch.sqrt(var + eps)
        scale = gamma / std
        
        # Compute bias adjustment: β - γ * μ / √(σ² + ε)
        bias_adjust = beta - gamma * mean / std
        
        # Get conv weight shape
        weight = conv.weight
        
        if isinstance(conv, nn.Conv2d) or (hasattr(conv, 'kernel_size')):
            # Conv2d: weight shape is [out_ch, in_ch, kH, kW]
            # Scale each output channel
            scale_shape = scale.view(-1, 1, 1, 1)
            conv.weight.data = weight * scale_shape
        else:
            # Linear: weight shape is [out_features, in_features]
            scale_shape = scale.view(-1, 1)
            conv.weight.data = weight * scale_shape
        
        # Handle bias
        if conv.bias is not None:
            # Existing bias: b_fused = b * scale + bias_adjust
            conv.bias.data = conv.bias * scale + bias_adjust
        else:
            # No existing bias: create one with just bias_adjust
            conv.bias = nn.Parameter(bias_adjust.clone())


def _replace_bn_with_identity(model: nn.Module, bn_names: List[str]):
    """Replace BatchNorm layers with Identity.
    
    This ensures the fused BatchNorm layers don't affect forward pass.
    """
    for bn_name in bn_names:
        # Navigate to parent and replace
        parts = bn_name.split('.')
        
        if len(parts) == 1:
            # Top-level module
            setattr(model, bn_name, nn.Identity())
        else:
            # Nested module
            parent = model
            for part in parts[:-1]:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            
            child_name = parts[-1]
            if child_name.isdigit():
                parent[int(child_name)] = nn.Identity()
            else:
                setattr(parent, child_name, nn.Identity())


def check_bn_fused(model: nn.Module) -> bool:
    """Check if all BatchNorm layers have been fused.
    
    Returns:
        True if no BatchNorm layers remain (or all are Identity)
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            return False
    return True
