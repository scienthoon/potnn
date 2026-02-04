"""Scale decomposition and combined scale calculation for C code generation."""

import torch
import math
from typing import List, Optional, Tuple


def decompose_scale_to_shifts(scale: int) -> List[int]:
    """Decompose an integer scale into shift positions.

    This finds which bits are set in the scale value,
    allowing multiplication to be replaced with shifts and adds.

    Example:
        scale = 21 = 0b10101 = (1<<0) + (1<<2) + (1<<4)
        Returns: [0, 2, 4]
        Meaning: x * 21 = x + (x<<2) + (x<<4)

    Args:
        scale: Integer scale value

    Returns:
        List of bit positions where scale has 1s
    """
    shifts = []
    for i in range(20):  # Check up to 20 bits (supports scales up to ~1M)
        if scale & (1 << i):
            shifts.append(i)
    return shifts


def generate_scale_func(layer_name: str, scale: float, shift: int) -> str:
    """Generate C function for scale multiplication using shifts and adds.

    This function converts floating-point scale to fixed-point integer,
    then decomposes it into shift+add operations to avoid multiplication.

    Args:
        layer_name: Name of the layer (for function naming)
        scale: Floating-point scale value
        shift: Total right shift to apply after multiplication

    Returns:
        C function code as string
    """
    # Convert to fixed-point integer
    scale_int = int(scale * (1 << 16))  # 16-bit fixed point
    total_shift = shift + 16

    # Decompose scale into shifts
    shifts = decompose_scale_to_shifts(scale_int)

    if not shifts:
        # Scale is 0
        return f"""static inline int32_t scale_{layer_name}(int32_t x) {{
    return 0;
}}
"""

    # Generate C code
    code = f"static inline int32_t scale_{layer_name}(int32_t x) {{\n"

    if len(shifts) == 1 and shifts[0] == 0:
        # Scale is 1, only need shift
        if total_shift == 0:
            code += f"    return x;\n"
        elif total_shift == 1:
            code += f"    return (x + 1) >> 1;  // Rounding\n"
        else:
            code += f"    return (x + (1 << {total_shift - 1})) >> {total_shift};  // Rounding\n"
    else:
        # Multiple shifts - need to add them
        terms = []
        for s in shifts:
            if s == 0:
                terms.append("x")
            else:
                terms.append(f"(x << {s})")

        # Add rounding term for proper rounding (handle edge cases)
        if total_shift == 0:
            code += f"    return ({' + '.join(terms)});\n"
        elif total_shift == 1:
            code += f"    return (({' + '.join(terms)}) + 1) >> 1;\n"
        else:
            code += f"    return (({' + '.join(terms)}) + (1 << {total_shift - 1})) >> {total_shift};\n"

    code += "}\n"
    return code


def calculate_combined_scale(
    alpha: float,
    act_scale: Optional[float],
    prev_act_scale: float = 1.0,
    std: Optional[float] = None,
    is_first: bool = False
) -> Tuple[float, int]:
    """Calculate combined scale for a layer.

    Formula:
    - General layer: combined_scale = alpha * act_scale / prev_act_scale
    - First layer: combined_scale = alpha * act_scale / std (absorb /std)

    The /256 normalization is handled separately via shift adjustment.

    Args:
        alpha: Alpha scaling parameter from the layer
        act_scale: Activation scale from calibration (None for last layer)
        prev_act_scale: Previous layer's activation scale
        std: Standard deviation for input normalization (first layer only)
             Can be float or List[float] (uses average for multi-channel)
        is_first: Whether this is the first layer

    Returns:
        Tuple of (scale_float, base_shift)
    """
    # Start with alpha
    combined_scale = alpha

    # Apply activation scale (if not None)
    if act_scale is not None:
        combined_scale *= act_scale

    # Handle first layer standardization
    if is_first and std is not None:
        # Use average std for multi-channel
        if isinstance(std, (list, tuple)):
            avg_std = sum(std) / len(std)
        else:
            avg_std = std
        combined_scale /= avg_std
    else:
        # Compensate for previous layer's scale
        combined_scale /= prev_act_scale

    # Base shift (will be adjusted for /256 in first layer)
    base_shift = 0

    return combined_scale, base_shift


def absorb_standardization(first_layer, mean, std):
    """Absorb input standardization into the first layer.

    This modifies the first layer's bias to absorb the mean subtraction:
    b' = b - Σ_c (mean[c]/std[c]) × ΣW[:,c,:,:] × α

    The std division is handled in combined_scale calculation.

    Args:
        first_layer: The first PoT layer in the model
        mean: Input mean for standardization (float or List[float])
        std: Input standard deviation for standardization (float or List[float])
    """
    if first_layer.bias is None:
        return

    # Normalize to list
    if isinstance(mean, (int, float)):
        mean = [mean]
    if isinstance(std, (int, float)):
        std = [std]

    with torch.no_grad():
        weight = first_layer.weight
        in_channels = weight.shape[1] if len(weight.shape) > 1 else 1
        
        # Channel-wise bias adjustment
        for c in range(in_channels):
            if hasattr(first_layer, 'kernel_size'):
                # Conv2d: sum over kernel for this channel
                weight_sum_c = weight[:, c, :, :].sum(dim=(1, 2))
            else:
                # Linear: handle flattened input
                features_per_ch = weight.shape[1] // len(mean)
                start_idx = c * features_per_ch
                end_idx = start_idx + features_per_ch
                weight_sum_c = weight[:, start_idx:end_idx].sum(dim=1)
            
            first_layer.bias.data -= (mean[c] / std[c]) * weight_sum_c

        print(f"Absorbed standardization into first layer:")
        print(f"  mean={mean}, std={std}")