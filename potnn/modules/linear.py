"""PoT-quantized Linear layer with Integer Simulation.

v2: Added integer simulation for C-compatible QAT
- Forward pass can simulate C integer operations exactly
- Matches C inference bit-for-bit when use_integer_sim=True
- Eliminates QAT-C accuracy gap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import PoTLayerBase
from ..quantize.pot import quantize_to_pot_ste, quantize_to_pot, quantize_activation_ste, apply_5level_zero_constraint
from ..quantize.integer_ops import (
    round_half_up_ste, clamp_ste,
    fake_quantize_input, fake_quantize_input_uint8,
    fake_requantize
)


class PoTLinear(PoTLayerBase):
    """Power-of-Two quantized Linear layer.

    This layer implements a Linear (fully connected) layer with PoT weight
    quantization.
    
    [Integer-Only QAT Mode]
    The forward pass simulates C integer arithmetic EXACTLY:
    1. Input Quantization: float -> int8 (or uint8 for first layer)
    2. Integer Linear: int8 * int8 -> int32
    3. Requantize: (int32 * scale_int + round) >> shift
    4. Bias Add: + round(bias_adjusted * act_scale)
    5. Clamp: [0, 127]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        encoding: str = 'unroll'
    ):
        """Initialize PoTLinear layer.

        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: If True, adds a learnable bias (default: True)
            encoding: Encoding type ('unroll', 'fp130', '5level', '2bit', 'ternary')
        """
        super().__init__(encoding)

        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight parameter
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # Initialize bias parameter
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize weights using Kaiming normal
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with three modes:
        1. Float warmup (quantize=False): Standard linear
        2. Float QAT (use_integer_sim=False): PoT weight + float activation
        3. Integer sim (use_integer_sim=True): C-identical integer ops
        """
        if not self.quantize:
            # Float mode (warmup training)
            return F.linear(x, self.weight, self.bias)
            
        if not getattr(self, 'use_integer_sim', False):
            # Float QAT: PoT weight + float activation
            # ReLU는 모델에서 외부로 호출
            w_pot = quantize_to_pot_ste(self.weight, self.alpha, encoding=self.encoding)
            
            # 5level constraint
            if self.encoding == '5level' and self.enforce_5level_constraint:
                w_pot = apply_5level_zero_constraint(w_pot)
            
            out = F.linear(x, w_pot * self.alpha, self.bias)
            return out
        
        # === Integer Simulation Mode (C-identical) ===
        
        # === 1. Prepare Integer Parameters ===
        # Always compute dynamically to ensure consistency with export
        scale_int, shift, _ = self._compute_scale_and_shift()

        is_first = self.is_first_layer.item() if self.is_first_layer is not None else False
        is_last = self.is_last_layer.item() if self.is_last_layer is not None else False
        
        # === 2. Input Quantization ===
        if is_first:
            # First layer: input is NORMALIZED (x - mean) / std
            # Simulate C behavior: raw uint8 input
            if self.input_mean is not None and self.input_std is not None:
                avg_std = self.input_std.mean().item()
                # Handle flattened input for Linear layer
                if x.dim() == 2:  # [batch, features]
                    num_ch = len(self.input_mean)
                    feat_per_ch = x.shape[1] // num_ch
                    if x.shape[1] == num_ch * feat_per_ch:
                        # Reshape to apply channel-wise mean
                        x_reshaped = x.view(x.shape[0], num_ch, feat_per_ch)
                        mean = self.input_mean.view(1, -1, 1).to(x.device)
                        x_raw = x_reshaped * avg_std + mean
                        x_raw = x_raw.view(x.shape[0], -1)
                    else:
                        # Fallback to average mean if dimensions don't match
                        avg_mean = self.input_mean.mean().item()
                        x_raw = x * avg_std + avg_mean
                else:
                    avg_mean = self.input_mean.mean().item()
                    x_raw = x * avg_std + avg_mean
                
                x_raw = clamp_ste(x_raw, 0.0, 1.0)
            else:
                x_raw = x
                
            # Quantize to uint8 [0, 255]
            x_int = fake_quantize_input_uint8(x_raw, 256.0)
        else:
            # Other layers: Input is already int8 from previous layer
            x_int = x

        # === 3. Weight Quantization ===
        w_pot = quantize_to_pot_ste(self.weight, self.alpha, encoding=self.encoding)
        
        # 5level constraint (always apply for 5level encoding to match export)
        if self.encoding == '5level':
            w_pot = apply_5level_zero_constraint(w_pot)

        # === 4. Integer Linear ===
        # F.linear with integer-valued inputs/weights -> integer-valued output
        acc = F.linear(x_int, w_pot, None)

        # === 5. Requantize ===
        acc_scaled = fake_requantize(acc, scale_int, shift)

        # === 6. Bias Addition ===
        if self.bias is not None:
            act_scale = self.act_scale if self.act_scale is not None else torch.tensor(1.0)
            
            if is_first and self.input_mean is not None and self.input_std is not None:
                # Absorb mean/std into bias
                avg_std = self.input_std.mean().item()
                alpha = self.alpha
                in_features = self.weight.shape[1]
                bias_adjusted = self.bias.clone()
                
                num_channels = len(self.input_mean)
                features_per_channel = in_features // num_channels if num_channels > 0 else in_features
                
                if num_channels > 0 and in_features == num_channels * features_per_channel:
                    for c in range(num_channels):
                        mean_c = self.input_mean[c].item()
                        start_idx = c * features_per_channel
                        end_idx = start_idx + features_per_channel
                        weight_sum_c = w_pot[:, start_idx:end_idx].sum(dim=1) * alpha
                        bias_adjusted = bias_adjusted - (mean_c / avg_std) * weight_sum_c
                else:
                    # Fallback if dimensions don't match
                    bias_adjusted = self.bias
            else:
                bias_adjusted = self.bias

            # Quantize bias: round(bias * act_scale)
            bias_int = round_half_up_ste(bias_adjusted * act_scale)
            
            # Add bias
            acc_scaled = acc_scaled + bias_int

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
