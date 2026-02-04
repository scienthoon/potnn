"""Quantization-aware training utilities for potnn."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


def _compute_raw_alpha(target_alpha: float) -> float:
    """Compute raw_alpha value that produces target alpha after softplus.
    
    softplus(raw_alpha) = target_alpha
    raw_alpha = inverse_softplus(target_alpha) = log(exp(target_alpha) - 1)
    
    Args:
        target_alpha: Desired alpha value after softplus
        
    Returns:
        raw_alpha value to set
    """
    if target_alpha > 10:
        # For large values, softplus(x) ≈ x
        return target_alpha
    elif target_alpha > 0.1:
        # Normal case: inverse softplus
        return math.log(math.exp(target_alpha) - 1)
    elif target_alpha > 0.01:
        # Small values: approximation
        return math.log(target_alpha)
    else:
        # Very small: minimum
        return -4.0  # softplus(-4) ≈ 0.018


def prepare_qat(model: nn.Module, config=None, force_alpha_init: bool = True):
    """Prepare model for quantization-aware training.

    This function:
    1. Fuses BatchNorm layers into preceding Conv layers (CRITICAL!)
    2. Sets encoding for each layer based on config.layer_encodings
    3. Initializes alpha values based on weight statistics (if not calibrated)
    4. Calculates combined scale factors for each layer
    5. Enables quantization mode for all PoT layers

    Args:
        model: Model to prepare (should be wrapped with potnn)
        config: potnn Config with layer_encodings (optional)
        force_alpha_init: If True, always initialize alpha from weight stats.
                         If False, only initialize if alpha seems uncalibrated (≈0.974).
                         Default True for safety.
    """
    from ..modules import PoTLinear, PoTConv2d, PoTDepthwiseConv2d
    from ..modules.base import PoTLayerBase
    from ..fuse import fuse_batchnorm, check_bn_fused

    # ========================================
    # Step 0: Fuse BatchNorm layers BEFORE QAT
    # This is CRITICAL! BN must be fused before PoT weight training.
    # ========================================
    fuse_batchnorm(model)

    # Replace AdaptiveAvgPool2d with PoTGlobalAvgPool BEFORE collecting layers
    # This ensures it's included in any subsequent processing if needed
    from ..modules.avgpool import PoTGlobalAvgPool
    # Use list(model.named_modules()) to avoid modification during iteration issues
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.AdaptiveAvgPool2d):
            if module.output_size == 1 or module.output_size == (1, 1):
                replacements.append((name, module))
    
    for name, module in replacements:
        parts = name.split('.')
        if len(parts) > 1:
            parent = model.get_submodule('.'.join(parts[:-1]))
            child_name = parts[-1]
        else:
            parent = model
            child_name = name
        
        setattr(parent, child_name, PoTGlobalAvgPool())
        print(f"  {name}: Replaced AdaptiveAvgPool2d with PoTGlobalAvgPool")


    print("Preparing model for QAT...")

    # Collect all PoT layers
    pot_layers = []
    for name, module in model.named_modules():
        if isinstance(module, PoTLayerBase):
            pot_layers.append((name, module))

    # ========================================
    # Step 0.5: Set encoding for each layer
    # ========================================
    if config is not None:
        for name, module in pot_layers:
            encoding = config.get_encoding(name)
            module.encoding = encoding
            print(f"  {name}: encoding={encoding}")

    # Initialize alpha values based on weight statistics
    # This is a safety net if calibration was not called
    for name, module in pot_layers:
        if not hasattr(module, 'weight'):
            continue
            
        with torch.no_grad():
            current_alpha = module.alpha.item()
            w_std = module.weight.std().item()
            
            # Check if alpha needs initialization
            # Default alpha ≈ 0.974 (from softplus(0.5))
            needs_init = force_alpha_init or (0.9 < current_alpha < 1.1)
            
            if needs_init and w_std > 0.001:
                # Use weight std as target alpha
                target_alpha = w_std
                target_alpha = max(target_alpha, 0.01)  # Minimum
                
                # Compute raw_alpha for this target
                raw = _compute_raw_alpha(target_alpha)
                module.raw_alpha.data.fill_(raw)
                
                # Update alpha_init for regularization
                actual_alpha = module.alpha.item()
                module.alpha_init.data.fill_(actual_alpha)
                
                print(f"  {name}: alpha initialized {current_alpha:.4f} → {actual_alpha:.4f} (w_std={w_std:.4f})")
            else:
                print(f"  {name}: alpha kept at {current_alpha:.4f} (w_std={w_std:.4f})")

    # Calculate combined scale factors and set prev_act_scale
    # These are used for both Integer-Only QAT and C export
    prev_act_scale = 1.0
    
    # Iterate over all modules in order to propagate prev_act_scale correctly
    # We need to handle both PoTLayerBase and PoTGlobalAvgPool
    from ..modules.avgpool import PoTGlobalAvgPool
    
    # Collect relevant layers in order
    ordered_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (PoTLayerBase, PoTGlobalAvgPool)):
            ordered_layers.append((name, module))
            
    for i, (name, module) in enumerate(ordered_layers):
        is_first = (i == 0)
        is_last = (i == len(ordered_layers) - 1)
        
        if isinstance(module, PoTLayerBase):
            module.set_layer_position(is_first, is_last)
            
            # Set previous layer's act_scale (Critical for Integer QAT)
            if isinstance(prev_act_scale, torch.Tensor):
                 module.set_prev_act_scale(prev_act_scale.item())
            else:
                 module.set_prev_act_scale(prev_act_scale)
                 
            # Also set combined_scale_factor for export (legacy but useful)
            scale_val = prev_act_scale.item() if isinstance(prev_act_scale, torch.Tensor) else prev_act_scale
            module.combined_scale_factor = 1.0 / scale_val
            
            # Update prev_act_scale for next layer
            if module.act_scale is not None:
                prev_act_scale = module.act_scale
                
        elif isinstance(module, PoTGlobalAvgPool):
            # Set GAP's input scale
            scale_val = prev_act_scale.item() if isinstance(prev_act_scale, torch.Tensor) else prev_act_scale
            module.prepare_qat(act_scale=scale_val)
            
            # GAP doesn't change scale (it's averaging), so prev_act_scale remains valid for next layer
            # Unless GAP has its own act_scale? 
            # PoTGlobalAvgPool has act_scale (input scale).
            # Output scale is same as input scale for averaging?
            # Yes, average of scaled integers is scaled integer.
            # So prev_act_scale should NOT change.
            pass

            print(f"  {name}: GAP prepared for QAT (act_scale={scale_val:.4f})")

    # Enable quantization mode
    for name, module in pot_layers:
        module.quantize = True

    # Also enable quantization for PoTAdd layers
    from ..modules.add import PoTAdd
    add_count = 0
    for name, module in model.named_modules():
        if isinstance(module, PoTAdd):
            module.quantize = True
            add_count += 1

    print(f"QAT mode enabled for {len(pot_layers)} PoT layers and {add_count} Add layers.")


def alpha_reg_loss(model: nn.Module, lambda_reg: float = 0.01) -> torch.Tensor:
    """Calculate alpha regularization loss.

    This prevents alpha values from drifting too far from their initial values,
    which helps stabilize training and prevent weight collapse.

    Args:
        model: Model with PoT layers
        lambda_reg: Regularization strength

    Returns:
        Regularization loss term to add to the main loss
    """
    from ..modules import PoTLinear, PoTConv2d, PoTConv1d

    reg_loss = torch.tensor(0.0, device=next(model.parameters()).device)

    for module in model.modules():
        if isinstance(module, (PoTLinear, PoTConv2d, PoTConv1d)):
            # Apply regularization to keep alpha near its initial value
            alpha = F.softplus(module.alpha).clamp(min=0.01)
            reg_loss += (alpha - module.alpha_init) ** 2

    return lambda_reg * reg_loss


def enable_integer_sim(model: nn.Module, input_std=1.0, input_mean=0.0, verbose: bool = True):
    """Enable integer simulation mode for C-compatible inference.
    
    This function sets up all PoT layers for integer simulation that matches
    C inference bit-for-bit. Call this after calibration and before QAT training
    for best results.
    
    The integer simulation ensures:
    - Python forward pass uses same integer arithmetic as C
    - Eliminates QAT-to-C accuracy gap
    - Proper scale chain propagation between layers
    
    Args:
        model: Model with PoT layers (should already be calibrated)
        input_std: Input standard deviation for first layer. 
                   Can be float, List[float], or Config object.
        input_mean: Input mean for first layer.
                    Can be float, List[float], or Config object.
        verbose: Print debug info (default True)
    
    Example:
        # After calibration, before QAT training:
        prepare_qat(model)
        enable_integer_sim(model, input_std=0.3081, input_mean=0.1307)
        
        # Or after training, before export:
        enable_integer_sim(model, input_std=config.std, input_mean=config.mean)
    """
    from ..modules.base import PoTLayerBase
    from ..modules.add import PoTAdd
    from ..config import Config
    
    # Handle Config object passed as input_std (common mistake)
    if isinstance(input_std, Config):
        config = input_std
        input_std = config.std
        input_mean = config.mean if hasattr(config, 'mean') else input_mean
    
    # Convert list to average (for multi-channel, use avg_std like C code)
    if isinstance(input_std, (list, tuple)):
        input_std = sum(input_std) / len(input_std)
    if isinstance(input_mean, (list, tuple)):
        input_mean = sum(input_mean) / len(input_mean)
    
    if verbose:
        print("="*60)
        print("Enabling integer simulation mode...")
        print(f"  input_mean={input_mean}, input_std={input_std}")
        print("="*60)
    
    # Collect all PoT layers in order
    pot_layers = []
    for name, module in model.named_modules():
        if isinstance(module, PoTLayerBase) and not isinstance(module, PoTAdd):
            pot_layers.append((name, module))
    
    if len(pot_layers) == 0:
        print("  Warning: No PoT layers found!")
        return
    
    # Set up each layer
    prev_act_scale = 1.0
    
    for i, (name, layer) in enumerate(pot_layers):
        is_first = (i == 0)
        is_last = (i == len(pot_layers) - 1)
        
        # Set layer position
        layer.set_layer_position(is_first, is_last)
        
        # Set previous layer's act_scale
        layer.set_prev_act_scale(prev_act_scale)
        
        # Set input stats for first layer
        if is_first:
            layer.set_input_std(input_std, input_mean)
            if verbose:
                print(f"  [DEBUG] First layer: mean={input_mean}, std={input_std}")
        
        # Compute integer parameters
        scale_int, shift = layer.compute_integer_params()
        
        # Enable integer simulation
        layer.use_integer_sim = True
        
        if verbose:
            act_scale = layer.act_scale.item() if layer.act_scale is not None else None
            alpha = layer.alpha.item()
            print(f"  {name}: first={is_first}, last={is_last}, "
                  f"prev_scale={prev_act_scale:.4f}, act_scale={act_scale}, "
                  f"alpha={alpha:.4f}, scale_int={scale_int}, shift={shift}")
        
        # Update prev_act_scale for next layer
        if layer.act_scale is not None:
            prev_act_scale = layer.act_scale.item()
    
    # Enable integer sim for PoTGlobalAvgPool layers
    from ..modules.avgpool import PoTGlobalAvgPool
    for name, module in model.named_modules():
        if isinstance(module, PoTGlobalAvgPool):
            module.integer_sim_enabled = True
            if verbose:
                print(f"  {name}: GAP integer sim enabled")
    


    
    if verbose:
        print(f"Integer simulation enabled for {len(pot_layers)} layers.")
        print("="*60)


def disable_integer_sim(model: nn.Module):
    """Disable integer simulation mode, reverting to float QAT.
    
    Args:
        model: Model with PoT layers
    """
    from ..modules.base import PoTLayerBase
    from ..modules.avgpool import PoTGlobalAvgPool
    
    for module in model.modules():
        if isinstance(module, PoTLayerBase):
            module.use_integer_sim = False
        elif isinstance(module, PoTGlobalAvgPool):
            module.integer_sim_enabled = False
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            if hasattr(module, '_original_forward'):
                module.forward = module._original_forward
                del module._original_forward
    
    print("Integer simulation disabled.")