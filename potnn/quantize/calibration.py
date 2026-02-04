"""Activation calibration for PoT quantization."""

import torch
import math
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Union, List

from ..modules.base import PoTLayerBase


def calibrate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    num_batches: int = 10,
    mean: Union[float, List[float]] = 0.0,
    std: Union[float, List[float]] = 1.0
) -> Dict[str, float]:
    """Calibrate activation scales for each PoT layer.

    This function measures the maximum activation values for each layer
    during inference on calibration data. These values are used to set
    fixed activation scales for quantization-aware training.

    IMPORTANT: This function should be called ONCE before QAT training,
    after float warmup training.

    Args:
        model: Model containing PoT layers to calibrate
        data_loader: DataLoader for calibration data
        num_batches: Number of batches to use for calibration (default: 10)
        mean: Dataset mean for normalization (default: 0.0)
        std: Dataset std for normalization (default: 1.0)

    Returns:
        Dictionary mapping layer names to maximum activation values
    """
    # ========================================
    # Step 0: Fuse BatchNorm layers BEFORE calibration
    # This ensures calibration measures activations with final fused weights.
    # ========================================
    from ..fuse import fuse_batchnorm
    fuse_batchnorm(model)
    
    print("Starting calibration...")

    # Set model to evaluation mode
    model.eval()

    # Dictionary to store max activation values for each layer
    activation_max = {}
    hooks = []

    def make_hook(name):
        """Create a forward hook to capture activation values."""
        def hook(module, input, output):
            with torch.no_grad():
                max_val = output.abs().max().item()
                if name not in activation_max:
                    activation_max[name] = max_val
                else:
                    activation_max[name] = max(activation_max[name], max_val)
        return hook

    # Register hooks on all PoT layers
    for name, module in model.named_modules():
        if isinstance(module, PoTLayerBase):
            hook = module.register_forward_hook(make_hook(name))
            hooks.append(hook)
    
    # Also register hooks on PoTAdd layers to track their output range
    from ..modules.add import PoTAdd
    
    # PoTAdd 입력 scale 추적용 딕셔너리
    add_input_max = {}  # {name: {'x': max_x, 'y': max_y}}
    
    def make_add_hook(name):
        """Create a forward hook for PoTAdd to capture input/output values."""
        def hook(module, input, output):
            with torch.no_grad():
                # output max
                max_val = output.abs().max().item()
                if name not in activation_max:
                    activation_max[name] = max_val
                else:
                    activation_max[name] = max(activation_max[name], max_val)
                
                # input max (x=skip, y=conv)
                x, y = input[0], input[1]
                max_x = x.abs().max().item()
                max_y = y.abs().max().item()
                
                if name not in add_input_max:
                    add_input_max[name] = {'x': max_x, 'y': max_y}
                else:
                    add_input_max[name]['x'] = max(add_input_max[name]['x'], max_x)
                    add_input_max[name]['y'] = max(add_input_max[name]['y'], max_y)
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, PoTAdd):
            hook = module.register_forward_hook(make_add_hook(name))
            hooks.append(hook)

    # Run forward passes to collect statistics
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break

            # Handle different data formats (data, target) or just data
            if isinstance(batch, (list, tuple)):
                data = batch[0]
            else:
                data = batch

            # Move to same device as model
            device = next(model.parameters()).device
            data = data.to(device)
            
            # Normalize data (support both scalar and per-channel mean/std)
            # Dynamic reshape based on data dimensions (3D for Conv1d, 4D for Conv2d)
            if isinstance(mean, (list, tuple)):
                mean_t = torch.tensor(mean, dtype=data.dtype, device=device)
                if data.dim() == 3:  # Conv1d: (B, C, L)
                    mean_t = mean_t.view(1, -1, 1)
                else:  # Conv2d: (B, C, H, W)
                    mean_t = mean_t.view(1, -1, 1, 1)
            else:
                mean_t = mean
            if isinstance(std, (list, tuple)):
                std_t = torch.tensor(std, dtype=data.dtype, device=device)
                if data.dim() == 3:  # Conv1d: (B, C, L)
                    std_t = std_t.view(1, -1, 1)
                else:  # Conv2d: (B, C, H, W)
                    std_t = std_t.view(1, -1, 1, 1)
            else:
                std_t = std
            data = (data - mean_t) / std_t

            # Forward pass
            _ = model(data)

            # Progress indicator
            if (i + 1) % 5 == 0:
                print(f"  Calibration batch {i + 1}/{min(num_batches, len(data_loader))}")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Set activation scales based on calibration
    print("\nSetting activation scales:")
    for name, module in model.named_modules():
        if isinstance(module, PoTLayerBase):
            if name in activation_max and activation_max[name] > 0:
                # Set the activation scale for quantization
                module.calibrate(activation_max[name])
                print(f"  {name}: act_max={activation_max[name]:.2f}, "
                      f"act_scale={module.act_scale.item():.4f}")
            else:
                # Default if layer wasn't activated during calibration
                module.calibrate(1.0)
                print(f"  {name}: no activations detected, using default scale")
    
    # Set activation scales for PoTAdd layers
    print("\nSetting PoTAdd scales:")
    for name, module in model.named_modules():
        if isinstance(module, PoTAdd):
            if name in activation_max and activation_max[name] > 0:
                # act_scale = 127 / max_activation
                act_scale = 127.0 / activation_max[name]
                module.act_scale = torch.tensor(act_scale)
                module.quantize = True
                
                # 입력 scale 설정 (핵심!)
                if name in add_input_max:
                    max_x = add_input_max[name]['x']
                    max_y = add_input_max[name]['y']
                    scale_x = 127.0 / max_x if max_x > 0 else 1.0
                    scale_y = 127.0 / max_y if max_y > 0 else 1.0
                    module.set_scales(scale_x, scale_y)
                    print(f"  {name}: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}, "
                          f"ratio={scale_y/scale_x:.4f} (PoTAdd)")
                else:
                    print(f"  {name}: act_max={activation_max[name]:.2f}, "
                          f"act_scale={act_scale:.4f} (PoTAdd, no input scales)")
            else:
                # Default scale
                module.act_scale = torch.tensor(1.0)
                module.quantize = True
                print(f"  {name}: no activations detected, using default scale (PoTAdd)")

    # Initialize alpha values based on weight distribution
    print("\nInitializing alpha values based on weight distribution:")
    for name, module in model.named_modules():
        if isinstance(module, PoTLayerBase):
            with torch.no_grad():
                # Calculate weight standard deviation
                w_std = module.weight.std().item()

                # Calculate appropriate raw_alpha value
                # We want: softplus(raw_alpha) ≈ w_std
                # softplus(x) = log(1 + exp(x))
                # So we need: x = log(exp(w_std) - 1) when w_std > log(2)
                if w_std > 0.01:
                    # Inverse softplus: raw = log(exp(target) - 1)
                    # But ensure numerical stability
                    target = w_std
                    if target > 10:  # Avoid overflow
                        raw = target  # For large values, softplus(x) ≈ x
                    elif target > 0.1:
                        raw = math.log(math.exp(target) - 1)
                    else:
                        # For small values, use approximation
                        raw = math.log(target) if target > 0 else -2.0
                else:
                    # Very small weights, use small alpha
                    raw = -2.0

                # Update raw_alpha
                module.raw_alpha.data.fill_(raw)

                # Update alpha_init to match the newly initialized alpha
                # This ensures regularization pulls towards the calibrated value
                module.alpha_init.data.fill_(module.alpha.item())

                # Verify the result
                actual_alpha = module.alpha.item()
                print(f"  {name}: weight_std={w_std:.4f}, "
                      f"raw_alpha={raw:.4f}, alpha={actual_alpha:.4f}")

    print(f"\nCalibration complete. Processed {len(activation_max)} layers.")
    return activation_max