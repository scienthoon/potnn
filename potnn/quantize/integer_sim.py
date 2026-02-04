"""Integer simulation functions for QAT.

These functions simulate C integer operations in PyTorch while allowing
gradient flow through Straight-Through Estimator (STE).

C operations:
    - round: (x + 0.5) truncation
    - clamp: min/max saturation  
    - requantize: (acc * scale_int + round) >> shift

Python simulation must match C bit-for-bit for QAT to be accurate.

Usage:
    from potnn.quantize.integer_sim import (
        round_ste, floor_ste, clamp_ste,
        quantize_to_int8_ste, quantize_to_uint8_ste,
        requantize_ste, compute_scale_params
    )
"""

import torch
import torch.nn as nn


class RoundSTE(torch.autograd.Function):
    """Round with Straight-Through Estimator.
    
    Forward: torch.round(x)
    Backward: gradient passes through unchanged
    """
    
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class RoundHalfUpSTE(torch.autograd.Function):
    """Half-up rounding with STE (C style).
    
    Forward: floor(x + 0.5) - matches C's (x + 0.5) truncation
    Backward: gradient passes through unchanged
    
    This matches C integer rounding:
        (int)(x + 0.5)  for positive x
        (x * scale + (1 << (shift-1))) >> shift
    """
    
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x + 0.5)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class FloorSTE(torch.autograd.Function):
    """Floor with Straight-Through Estimator.
    
    Forward: torch.floor(x)
    Backward: gradient passes through unchanged
    
    Used for integer division: a // b = floor(a / b)
    """
    
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ClampSTE(torch.autograd.Function):
    """Clamp with Straight-Through Estimator.
    
    Forward: torch.clamp(x, min_val, max_val)
    Backward: gradient passes through unchanged
    
    Note: Standard clamp has zero gradient outside [min, max].
    STE version allows gradient to flow for training stability.
    """
    
    @staticmethod
    def forward(ctx, x, min_val, max_val):
        return torch.clamp(x, min_val, max_val)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round with STE for gradient flow (C style half-up)."""
    return RoundHalfUpSTE.apply(x)  # floor(x + 0.5) - matches C


def round_half_up_ste(x: torch.Tensor) -> torch.Tensor:
    """Half-up rounding with STE (C style).
    
    This matches C integer rounding behavior.
    Example: 2.5 -> 3, -2.5 -> -2
    """
    return RoundHalfUpSTE.apply(x)


def floor_ste(x: torch.Tensor) -> torch.Tensor:
    """Floor with STE for gradient flow."""
    return FloorSTE.apply(x)


def clamp_ste(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """Clamp with STE for gradient flow."""
    return ClampSTE.apply(x, min_val, max_val)


def quantize_to_int8_ste(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to int8 range with STE.
    
    Forward:
        x_int = round(x * scale)
        x_int = clamp(x_int, -128, 127)
    
    Backward: gradient passes through unchanged
    
    Args:
        x: Input tensor (float)
        scale: Quantization scale (127.0 / max_activation)
    
    Returns:
        Tensor with int8 values (but float dtype for gradient)
    """
    x_scaled = x * scale
    x_rounded = round_ste(x_scaled)
    x_clamped = clamp_ste(x_rounded, -128.0, 127.0)
    return x_clamped


def quantize_to_uint8_ste(x: torch.Tensor, scale: float = 256.0) -> torch.Tensor:
    """Quantize tensor to uint8 range with STE.
    
    For first layer: input [0, 1] -> [0, 255]
    
    Args:
        x: Input tensor (float, assumed [0, 1] normalized)
        scale: Quantization scale (default 256 for /256 normalization)
    
    Returns:
        Tensor with uint8 values (but float dtype for gradient)
    """
    x_scaled = x * scale
    x_rounded = round_ste(x_scaled)
    x_clamped = clamp_ste(x_rounded, 0.0, 255.0)
    return x_clamped


def requantize_ste(acc: torch.Tensor, scale_int: int, shift: int) -> torch.Tensor:
    """Simulate C requantization with STE.
    
    C code:
        out = ((int64_t)acc * scale_int + (1 << (shift-1))) >> shift
    
    This is equivalent to:
        out = floor((acc * scale_int + round_const) / divisor)
    
    where round_const = 1 << (shift-1), divisor = 1 << shift
    
    Args:
        acc: Accumulator tensor (int32 range values in float tensor)
        scale_int: Integer scale factor
        shift: Right shift amount
    
    Returns:
        Requantized tensor (int32 range values in float tensor)
    """
    if shift > 0:
        round_const = 1 << (shift - 1)
    else:
        round_const = 0
    
    divisor = float(1 << shift)
    
    numerator = acc * float(scale_int) + float(round_const)
    result = floor_ste(numerator / divisor)
    
    return result


def compute_scale_params(combined_scale: float, target_range: tuple = (64, 512)) -> tuple:
    """Compute integer scale and shift from float scale.
    
    Find (scale_int, shift) such that:
        scale_int / (1 << shift) ≈ combined_scale
        target_range[0] <= scale_int <= target_range[1]
    
    Args:
        combined_scale: Float scale value (alpha * act_scale / prev_act_scale)
        target_range: Target range for scale_int (default 64-512 to match export.py)
    
    Returns:
        (scale_int, shift) tuple
    """
    if combined_scale == 0:
        return 0, 0
    
    min_scale, max_scale = target_range
    shift = 0
    scale_magnitude = abs(combined_scale)
    
    while scale_magnitude < min_scale and shift < 24:
        scale_magnitude *= 2
        shift += 1
    
    while scale_magnitude > max_scale and shift > 0:
        scale_magnitude /= 2
        shift -= 1
    
    scale_int = round(combined_scale * (1 << shift))
    
    return scale_int, shift
