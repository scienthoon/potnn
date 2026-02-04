"""Quantization module for potnn."""

from .pot import (
    quantize_to_pot, quantize_to_pot_ste, quantize_activation_ste, 
    quantize_activation_c_aligned_ste, get_pot_values
)
from .calibration import calibrate_model
from .qat import prepare_qat, alpha_reg_loss, enable_integer_sim, disable_integer_sim
from .integer_sim import (
    round_ste, round_half_up_ste, floor_ste, clamp_ste,
    quantize_to_int8_ste, quantize_to_uint8_ste,
    requantize_ste, compute_scale_params
)

__all__ = [
    'quantize_to_pot',
    'quantize_to_pot_ste',
    'quantize_activation_ste',
    'quantize_activation_c_aligned_ste',
    'get_pot_values',
    'calibrate_model',
    'prepare_qat',
    'alpha_reg_loss',
    'enable_integer_sim',
    'disable_integer_sim',
    # Integer simulation
    'round_ste',
    'round_half_up_ste',
    'floor_ste', 
    'clamp_ste',
    'quantize_to_int8_ste',
    'quantize_to_uint8_ste',
    'requantize_ste',
    'compute_scale_params',
]