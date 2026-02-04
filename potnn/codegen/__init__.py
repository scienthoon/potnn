"""C code generation module for potnn."""

from .header import generate_c_header
from .unroll import generate_unrolled_layer
from .scale import decompose_scale_to_shifts, generate_scale_func
from .fp130 import generate_fp130_layer
from .bit2 import generate_2bit_layer
from .level5 import generate_5level_layer
from .ternary import generate_ternary_layer

__all__ = [
    'generate_c_header',
    'generate_unrolled_layer',
    'decompose_scale_to_shifts',
    'generate_scale_func',
    'generate_fp130_layer',
    'generate_2bit_layer',
    'generate_5level_layer',
    'generate_ternary_layer',
]