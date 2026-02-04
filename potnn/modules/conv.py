"""PoT-quantized Conv2d layer with Integer Simulation.

v2: Added integer simulation for C-compatible QAT
- Forward pass can simulate C integer operations exactly
- Matches C inference bit-for-bit when use_integer_sim=True
- Eliminates QAT-C accuracy gap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple

from .base import PoTLayerBase
from ..quantize.pot import quantize_to_pot_ste, quantize_to_pot, quantize_activation_ste, apply_5level_zero_constraint
from ..quantize.integer_ops import (
    round_half_up_ste, clamp_ste,
    fake_quantize_input, fake_quantize_input_uint8,
    fake_requantize
)


class PoTConv2d(PoTLayerBase):
    """Power-of-Two quantized Conv2d layer.

    This layer implements a Conv2d layer with PoT weight quantization.
    
    [Integer-Only QAT Mode]
    The forward pass simulates C integer arithmetic EXACTLY:
    1. Input Quantization: float -> int8 (or uint8 for first layer)
    2. Integer Conv: int8 * int8 -> int32
    3. Requantize: (int32 * scale_int + round) >> shift
    4. Bias Add: + round(bias_adjusted * act_scale)
    5. Clamp: [0, 127]
    
    This ensures that training accuracy matches C deployment accuracy.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        encoding: str = 'unroll'
    ):
        super().__init__(encoding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups

        # Initialize weight parameter
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, *self.kernel_size
        ))

        # Initialize bias parameter
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize weights using Kaiming normal
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with three modes:
        1. Float warmup (quantize=False): Standard conv
        2. Float QAT (use_integer_sim=False): PoT weight + float activation
        3. Integer sim (use_integer_sim=True): C-identical integer ops
        """
        if not self.quantize:
            # Float mode (warmup training)
            return F.conv2d(
                x, self.weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups
            )
            
        if not getattr(self, 'use_integer_sim', False):
            # Float QAT: PoT weight + float activation
            # ReLU는 모델에서 외부로 호출 (torch.relu(conv(x)))
            w_pot = quantize_to_pot_ste(self.weight, self.alpha, encoding=self.encoding)
            
            # 5level constraint
            if self.encoding == '5level' and self.enforce_5level_constraint:
                w_pot = apply_5level_zero_constraint(w_pot)
            
            out = F.conv2d(
                x, w_pot * self.alpha, self.bias,
                self.stride, self.padding, self.dilation, self.groups
            )
            return out
        
        # === Integer Simulation Mode (C-identical) ===
        
        # === 1. Prepare Integer Parameters ===
        # Always compute dynamically to ensure consistency with export
        scale_int, shift, _ = self._compute_scale_and_shift()

        is_first = self.is_first_layer.item() if self.is_first_layer is not None else False
        is_last = self.is_last_layer.item() if self.is_last_layer is not None else False
        
        # === 2. Input Quantization ===
        if is_first:
            # First layer: Input is normalized float (x - mean) / std
            # We must simulate C behavior: raw uint8 input
            if self.input_mean is not None and self.input_std is not None:
                # Denormalize: x_raw = x * avg_std + mean
                avg_std = self.input_std.mean().item()
                mean = self.input_mean.view(1, -1, 1, 1).to(x.device)
                x_raw = x * avg_std + mean
                x_raw = clamp_ste(x_raw, 0.0, 1.0)
            else:
                x_raw = x
                
            # Quantize to uint8 [0, 255]
            # Match C test data generation (img * 255.0)
            x_int = fake_quantize_input_uint8(x_raw, 255.0)
        else:
            # Other layers: Input is already int8 from previous layer
            # No quantization needed
            x_int = x

        # === 3. Weight Quantization ===
        w_pot = quantize_to_pot_ste(self.weight, self.alpha, encoding=self.encoding)
        
        # 5level constraint (always apply for 5level encoding to match export)
        if self.encoding == '5level':
            w_pot = apply_5level_zero_constraint(w_pot)

        if is_first:
             # DEBUG: L0 weights
             pass


        # === 4. Integer Convolution ===
        # F.conv2d with integer-valued inputs/weights -> integer-valued output (float dtype)
        acc = F.conv2d(
            x_int, w_pot,
            None, # Bias added separately
            self.stride, self.padding, self.dilation, self.groups
        )

        # === 5. Requantize ===
        # (acc * scale_int + round) >> shift
        acc_scaled = fake_requantize(acc, scale_int, shift)

        # === 6. Bias Addition ===
        if self.bias is not None:
            act_scale = self.act_scale if self.act_scale is not None else torch.tensor(1.0)
            
            if is_first and self.input_mean is not None and self.input_std is not None:
                # Absorb mean/std into bias (Dynamic for training)
                # bias_adj = bias - (mean/std) * sum(W) * alpha
                avg_std = self.input_std.mean().item()
                alpha = self.alpha
                
                # Calculate weight sum per channel
                # w_pot shape: [out, in, k, k]
                w_sum = w_pot.sum(dim=(2, 3)) # [out, in]
                
                # We need to sum over input channels weighted by mean[c]
                # bias_correction = sum_c (mean[c]/avg_std * w_sum[:, c])
                mean_vec = self.input_mean.view(1, -1).to(x.device) # [1, in]
                bias_correction = (mean_vec / avg_std * w_sum).sum(dim=1) # [out]
                
                bias_adjusted = self.bias - bias_correction * alpha
            else:
                bias_adjusted = self.bias

            # Quantize bias: round(bias * act_scale)
            bias_int = round_half_up_ste(bias_adjusted * act_scale)
            
            # Add bias
            acc_scaled = acc_scaled + bias_int.view(1, -1, 1, 1)

        # === 7. Clamp (ReLU) ===
        if not is_last:
            out = clamp_ste(acc_scaled, 0.0, 127.0)
        else:
            out = acc_scaled

        # === 8. Output ===
        # Round to ensure exact integer (floating point precision)
        # Use STE to maintain gradient flow during training
        # int8 그대로 반환 (C와 동일)
        out = round_half_up_ste(out)

        return out

    def extra_repr(self) -> str:
        s = super().extra_repr()
        s += f', quantize={self.quantize}'
        return s
