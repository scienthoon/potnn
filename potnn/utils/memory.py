"""Memory estimation and validation utilities."""

import torch
import torch.nn as nn
from typing import Dict, Tuple


def estimate_layer_size(module: nn.Module) -> int:
    """Estimate the size of a layer in bytes.

    Args:
        module: Neural network module

    Returns:
        Estimated size in bytes
    """
    param_count = 0

    # Count weight parameters
    if hasattr(module, 'weight') and module.weight is not None:
        param_count += module.weight.numel()

    # Count bias parameters
    if hasattr(module, 'bias') and module.bias is not None:
        param_count += module.bias.numel()

    return param_count


def estimate_activation_size(model: nn.Module, input_shape: Tuple) -> int:
    """Estimate maximum activation buffer size needed.

    Args:
        model: Neural network model
        input_shape: Shape of input tensor (without batch dimension)

    Returns:
        Maximum activation size in bytes
    """
    # Create dummy input
    dummy_input = torch.zeros(1, *input_shape)
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)

    max_size = 0
    hooks = []

    def hook_fn(module, input, output):
        nonlocal max_size
        if isinstance(output, torch.Tensor):
            size = output.numel()  # int8 = 1 byte per element
            max_size = max(max_size, size)

    # Register hooks
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d)):
            hooks.append(module.register_forward_hook(hook_fn))

    # Run forward pass
    model.eval()
    with torch.no_grad():
        model(dummy_input)

    # Clean up hooks
    for hook in hooks:
        hook.remove()

    return max_size


def estimate_memory_usage(model: nn.Module, input_shape: Tuple, mode: str = 'all') -> Dict[str, int]:
    """Estimate memory usage of the model.

    Args:
        model: Neural network model
        input_shape: Shape of input tensor (without batch dimension)
        mode: 'all', 'weights', or 'activations'

    Returns:
        Dictionary with memory estimates in bytes
    """
    result = {}

    if mode in ['all', 'weights']:
        # Estimate weight memory
        total_weights = 0
        layer_weights = {}

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                size = estimate_layer_size(module)
                layer_weights[name] = size
                total_weights += size

        result['total_weights'] = total_weights
        result['layer_weights'] = layer_weights

    if mode in ['all', 'activations']:
        # Estimate activation memory
        result['max_activation'] = estimate_activation_size(model, input_shape)

    if mode == 'all':
        # Input buffer size
        input_size = torch.zeros(1, *input_shape).numel()
        result['input_buffer'] = input_size

        # Total RAM needed (input + activations + some weights in loop mode)
        result['estimated_ram'] = result['input_buffer'] + result['max_activation']

        # Total Flash needed (mainly for unrolled weights as code)
        # This is a rough estimate - actual size depends on unroll/loop decisions
        result['estimated_flash'] = total_weights * 4  # Rough estimate for unrolled code

    return result


def validate_memory(model: nn.Module, flash_budget: int, ram_budget: int,
                   input_shape: Tuple = (1, 16, 16)) -> bool:
    """Validate if model fits within memory budgets.

    Args:
        model: Neural network model
        flash_budget: Flash memory budget in bytes
        ram_budget: RAM budget in bytes
        input_shape: Input tensor shape (default for 16x16 grayscale)

    Returns:
        True if model fits, False otherwise

    Raises:
        ValueError: If model doesn't fit with error details
    """
    estimates = estimate_memory_usage(model, input_shape)

    # Check RAM budget
    min_ram_needed = estimates['input_buffer'] + estimates['max_activation']
    if min_ram_needed > ram_budget:
        raise ValueError(
            f"Model requires at least {min_ram_needed} bytes of RAM "
            f"(input: {estimates['input_buffer']}, activation: {estimates['max_activation']}), "
            f"but only {ram_budget} bytes available."
        )

    # Check if we can fit weights either in Flash (unrolled) or RAM (loop)
    # This is a simplified check - actual allocation is done by allocate_hybrid
    # For unrolled code, estimate ~4 bytes per weight
    # For loop with packing, estimate ~0.25 bytes per weight (2-bit packing)
    unrolled_size = estimates['total_weights'] * 4
    packed_size = estimates['total_weights'] // 4  # 2-bit packing

    if unrolled_size > flash_budget and packed_size > (ram_budget - min_ram_needed):
        raise ValueError(
            f"Model weights ({estimates['total_weights']} parameters) too large. "
            f"Unrolled: {unrolled_size} bytes > Flash {flash_budget} bytes. "
            f"Packed: {packed_size} bytes > Available RAM {ram_budget - min_ram_needed} bytes."
        )

    return True