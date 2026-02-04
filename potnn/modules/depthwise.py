"""PoT-quantized Depthwise Conv2d layer with Integer Simulation.

v2: Added integer simulation for C-compatible QAT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple

from .base import PoTLayerBase
from ..quantize.pot import quantize_to_pot_ste, quantize_to_pot, quantize_activation_ste
from ..quantize.integer_sim import (
    round_ste, floor_ste, clamp_ste,
    quantize_to_int8_ste, quantize_to_uint8_ste,
    requantize_ste
)


class PoTDepthwiseConv2d(PoTLayerBase):
    """Power-of-Two quantized Depthwise Conv2d layer.

    Depthwise convolution applies a single filter per input channel.
    This is commonly used in MobileNet-style architectures as the first
    part of depthwise separable convolution.
    
    Key properties:
        - in_channels == out_channels == channels
        - groups = channels (each channel processed independently)
        - weight shape: [channels, 1, kH, kW]
    """

    def __init__(
        self,
        channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        encoding: str = 'unroll'
    ):
        """Initialize PoTDepthwiseConv2d layer.

        Args:
            channels: Number of input/output channels
            kernel_size: Size of the convolution kernel (default: 3)
            stride: Stride of the convolution (default: 1)
            padding: Zero-padding added to both sides (default: 1)
            dilation: Spacing between kernel elements (default: 1)
            bias: If True, adds a learnable bias (default: True)
            encoding: Encoding type ('unroll', 'fp130', '5level', '2bit', 'ternary')
        """
        super().__init__(encoding)

        self.channels = channels
        self.in_channels = channels  # alias for compatibility
        self.out_channels = channels  # alias for compatibility
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = channels  # depthwise: each channel is its own group

        # Initialize weight parameter: [channels, 1, kH, kW]
        self.weight = nn.Parameter(torch.empty(
            channels, 1, *self.kernel_size
        ))

        # Initialize bias parameter
        if bias:
            self.bias = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter('bias', None)

        # Initialize weights using Kaiming normal
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional PoT quantization.

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Output tensor of shape (N, C, H_out, W_out)
        """
        if not self.quantize:
            # Float mode (warmup training)
            return F.conv2d(
                x, self.weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups
            )
        
        if self.use_integer_sim and self.scale_int is not None:
            if self.training:
                # Training: use float QAT for gradient flow
                return self._forward_float_qat(x)
            else:
                # Eval: use integer sim for C-exact match
                return self._forward_integer_sim(x)
        else:
            # Standard Float QAT Mode
            return self._forward_float_qat(x)

    def _forward_float_qat(self, x: torch.Tensor) -> torch.Tensor:
        """Original float QAT forward."""
        w_q = quantize_to_pot_ste(self.weight, self.alpha, encoding=self.encoding)

        out = F.conv2d(
            x, w_q * self.alpha, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )

        if self.act_scale is not None:
            out = quantize_activation_ste(out, self.act_scale)

        return out

    def _forward_integer_sim(self, x: torch.Tensor) -> torch.Tensor:
        """Integer simulation forward - matches C inference exactly."""
        DEBUG = False  # True로 바꾸면 상세 디버그 출력
        
        is_first = self.is_first_layer.item() if self.is_first_layer is not None else False
        is_last = self.is_last_layer.item() if self.is_last_layer is not None else False
        
        if DEBUG:
            print(f"\n[DEBUG DW _forward_integer_sim] is_first={is_first}, is_last={is_last}")
            print(f"  input: shape={x.shape}, range=[{x.min():.4f}, {x.max():.4f}]")
        
        # Step 1: Quantize input
        if is_first:
            # First layer: input is NORMALIZED, denormalize with channel-wise mean
            if self.input_mean is not None and self.input_std is not None:
                avg_std = self.input_std.mean().item()
                mean = self.input_mean.view(1, -1, 1, 1).to(x.device)  # [1, C, 1, 1]
                x_raw = x * avg_std + mean  # channel-wise mean
                x_raw = torch.clamp(x_raw, 0.0, 1.0)
            else:
                x_raw = x
            # [0,1] → [0,255] (uint8), /256 absorbed in shift (+8)
            x_int = quantize_to_uint8_ste(x_raw, 256.0)
        else:
            prev_scale = self.prev_act_scale if self.prev_act_scale is not None else torch.tensor(1.0)
            x_int = quantize_to_int8_ste(x, prev_scale)
        
        # Step 2: PoT Depthwise Convolution with STE for gradient flow
        w_pot = quantize_to_pot_ste(self.weight, self.alpha, encoding=self.encoding)
        
        acc = F.conv2d(
            x_int, w_pot, None,
            self.stride, self.padding, self.dilation, self.groups
        )
        
        # Step 3: Requantize
        scale_int = self.scale_int.item() if self.scale_int is not None else 1
        shift = self.shift.item() if self.shift is not None else 0
        acc = requantize_ste(acc, scale_int, shift)
        
        if DEBUG:
            print(f"  acc after requantize: scale_int={scale_int}, shift={shift}, range=[{acc.min():.0f}, {acc.max():.0f}]")
        
        # Step 4: Add bias (with mean absorption for first layer)
        if self.bias is not None:
            act_scale = self.act_scale if self.act_scale is not None else torch.tensor(1.0)
            
            if is_first:
                # First layer: absorb mean into bias
                # Use avg_std to match QAT and C inference
                if self.input_mean is not None and self.input_std is not None:
                    avg_std = self.input_std.mean().item()
                    # Depthwise: weight is [channels, 1, kH, kW]
                    channels = w_pot.shape[0]
                    alpha = self.alpha
                    bias_adjusted = self.bias.clone()
                    for c in range(channels):
                        mean_c = self.input_mean[c].item() if c < len(self.input_mean) else 0.0
                        weight_sum_c = w_pot[c].sum() * alpha
                        bias_adjusted[c] = bias_adjusted[c] - (mean_c / avg_std) * weight_sum_c
                else:
                    bias_adjusted = self.bias
                bias_int = round_ste(bias_adjusted * act_scale)
                
                if DEBUG:
                    print(f"  [First layer DW bias absorption]")
            else:
                bias_int = round_ste(self.bias * act_scale)
            
            acc = acc + bias_int.view(1, -1, 1, 1)
        
        # Step 5: Clamp
        if not is_last:
            out = clamp_ste(acc, 0.0, 127.0)
        else:
            out = acc
        
        # Step 6: Convert back to float
        if self.act_scale is not None and not is_last:
            out = out / self.act_scale
        
        return out

    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        s = f'channels={self.channels}, kernel_size={self.kernel_size}, stride={self.stride}'
        if self.padding != (0, 0):
            s += f', padding={self.padding}'
        if self.dilation != (1, 1):
            s += f', dilation={self.dilation}'
        if self.bias is None:
            s += ', bias=False'
        if self.quantize:
            s += f', quantize=True, encoding={self.encoding}'
        if self.use_integer_sim:
            s += ', integer_sim=True'
        return s
