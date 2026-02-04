"""PoT-quantized Conv1d layer with Integer Simulation.

v1: 1D convolution support for time-series and audio processing
- Forward pass can simulate C integer operations exactly
- Matches C inference bit-for-bit when use_integer_sim=True
- Eliminates QAT-C accuracy gap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

from .base import PoTLayerBase
from ..quantize.pot import quantize_to_pot_ste, quantize_to_pot, quantize_activation_ste
from ..quantize.integer_sim import (
    round_ste, floor_ste, clamp_ste,
    quantize_to_int8_ste, quantize_to_uint8_ste,
    requantize_ste
)


class PoTConv1d(PoTLayerBase):
    """Power-of-Two quantized Conv1d layer.

    This layer implements a Conv1d layer with PoT weight quantization
    and alpha scaling. It can be used as a drop-in replacement for
    nn.Conv1d in QAT-aware models.
    
    Supports two modes:
    - Float QAT (default): Standard fake quantization with float operations
    - Integer Simulation: C-compatible integer operations for exact match
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        encoding: str = 'unroll'
    ):
        """Initialize PoTConv1d layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolution kernel
            stride: Stride of the convolution (default: 1)
            padding: Zero-padding added to both sides (default: 0)
            dilation: Spacing between kernel elements (default: 1)
            groups: Number of blocked connections (default: 1)
            bias: If True, adds a learnable bias (default: True)
            encoding: Encoding type ('unroll', 'fp130', '5level', '2bit', 'ternary')
        """
        super().__init__(encoding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Initialize weight parameter: (out_channels, in_channels/groups, kernel_size)
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, self.kernel_size
        ))

        # Initialize bias parameter
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize weights using Kaiming normal
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional PoT quantization.

        Args:
            x: Input tensor of shape (N, C_in, L)

        Returns:
            Output tensor of shape (N, C_out, L_out)
        """
        if not self.quantize:
            # Float mode (warmup training)
            return F.conv1d(
                x, self.weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups
            )
        
        if self.use_integer_sim and self.scale_int is not None:
            if self.training:
                # Training: use float QAT for gradient flow
                # Integer simulation doesn't support alpha gradients
                return self._forward_float_qat(x)
            else:
                # Eval: use integer sim for C-exact match
                return self._forward_integer_sim(x)
        else:
            # Standard Float QAT Mode
            return self._forward_float_qat(x)

    def _forward_float_qat(self, x: torch.Tensor) -> torch.Tensor:
        """Original float QAT forward.
        
        NOTE: Input is already normalized as (data * 256/255 - mean) / avg_std
        so mean is already subtracted. NO bias absorption needed here.
        Bias absorption is only for Integer Sim (raw uint8 input) and C export.
        """
        DEBUG_QAT = False  # True로 바꾸면 QAT 디버그 출력
        
        # PoT quantization
        w_q = quantize_to_pot_ste(self.weight, self.alpha, encoding=self.encoding)

        if DEBUG_QAT:
            print(f"\n[DEBUG QAT] input: range=[{x.min():.4f}, {x.max():.4f}]")
            print(f"  w_q: unique_vals={torch.unique(w_q).tolist()[:10]}...")
            print(f"  alpha={self.alpha.item():.4f}")
            print(f"  w_effective (w_q*alpha): range=[{(w_q*self.alpha).min():.4f}, {(w_q*self.alpha).max():.4f}]")

        # Convolution with scaled weights (NO bias adjustment - input already normalized)
        out = F.conv1d(
            x, w_q * self.alpha, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )

        if DEBUG_QAT:
            print(f"  conv output: range=[{out.min():.4f}, {out.max():.4f}]")

        # Activation quantization
        if self.act_scale is not None:
            out = quantize_activation_ste(out, self.act_scale)
            if DEBUG_QAT:
                print(f"  after act_quant (scale={self.act_scale.item():.4f}): range=[{out.min():.4f}, {out.max():.4f}]")

        return out

    def _forward_integer_sim(self, x: torch.Tensor) -> torch.Tensor:
        """Integer simulation forward - matches C inference exactly.
        
        C code equivalent:
            // Step 1: PoT convolution
            int32_t acc = 0;
            for (...) {
                acc += input[i] << k;  // or -= for negative weights
            }
            
            // Step 2: Requantize
            acc = ((int64_t)acc * scale_int + round) >> shift;
            
            // Step 3: Add bias (with mean absorption for first layer)
            acc += bias_int;
            
            // Step 4: Clamp (ReLU)
            output = clamp(acc, 0, 127);  // or -128,127 if no ReLU
        """
        DEBUG = False  # True로 바꾸면 상세 디버그 출력
        
        is_first = self.is_first_layer.item() if self.is_first_layer is not None else False
        is_last = self.is_last_layer.item() if self.is_last_layer is not None else False
        
        if DEBUG:
            print(f"\n[DEBUG _forward_integer_sim] is_first={is_first}, is_last={is_last}")
            print(f"  input: shape={x.shape}, range=[{x.min():.4f}, {x.max():.4f}]")
        
        # === Step 1: Quantize input to integer ===
        if is_first:
            # First layer: input is NORMALIZED (x - mean) / avg_std
            # C code receives raw uint8 [0,255], so we denormalize first
            if self.input_mean is not None and self.input_std is not None:
                # Denormalize: x_raw = x_norm * avg_std + mean (channel-wise mean!)
                # QAT normalized with channel-wise mean, so denorm with channel-wise mean
                avg_std = self.input_std.mean().item()
                mean = self.input_mean.view(1, -1, 1).to(x.device)  # [1, C, 1]
                x_raw = x * avg_std + mean  # channel-wise mean
                x_raw = torch.clamp(x_raw, 0.0, 1.0)
            else:
                x_raw = x
            # [0,1] → [0,255] (uint8), /256 absorbed in shift (+8)
            x_int = quantize_to_uint8_ste(x_raw, 256.0)
            if DEBUG:
                print(f"  x_int (uint8): range=[{x_int.min():.0f}, {x_int.max():.0f}]")
        else:
            # Other layers: convert float back to int8
            # Input was divided by prev_act_scale in previous layer
            prev_scale = self.prev_act_scale if self.prev_act_scale is not None else torch.tensor(1.0)
            x_int = quantize_to_int8_ste(x, prev_scale)
            if DEBUG:
                print(f"  x_int (int8): prev_scale={prev_scale.item():.4f}, range=[{x_int.min():.0f}, {x_int.max():.0f}]")
        
        # === Step 2: PoT Convolution (integer) ===
        # Get PoT weights with STE for gradient flow
        w_pot = quantize_to_pot_ste(self.weight, self.alpha, encoding=self.encoding)
        
        if DEBUG:
            print(f"  w_pot: shape={w_pot.shape}, unique_vals={torch.unique(w_pot).tolist()[:10]}...")
            print(f"  alpha={self.alpha.item():.4f}")
        
        # Integer convolution
        # In C: acc += input << k (shift operation)
        # In Python: float tensor but values are integers
        acc = F.conv1d(
            x_int, w_pot,
            None,  # bias added separately
            self.stride, self.padding, self.dilation, self.groups
        )
        
        if DEBUG:
            print(f"  acc after conv: range=[{acc.min():.0f}, {acc.max():.0f}]")
        
        # === Step 3: Requantize ===
        # C: ((int64_t)acc * scale_int + round) >> shift
        scale_int = self.scale_int.item() if self.scale_int is not None else 1
        shift = self.shift.item() if self.shift is not None else 0
        
        if DEBUG:
            print(f"  scale_int={scale_int}, shift={shift}")
        
        acc = requantize_ste(acc, scale_int, shift)
        
        if DEBUG:
            print(f"  acc after requantize: range=[{acc.min():.0f}, {acc.max():.0f}]")
        
        # === Step 4: Add bias (with mean absorption for first layer) ===
        if self.bias is not None:
            act_scale = self.act_scale if self.act_scale is not None else torch.tensor(1.0)
            
            if is_first:
                # First layer: absorb mean into bias
                # MUST match export.py absorb_standardization exactly
                # Use avg_std to match QAT and C export
                
                if self.input_mean is not None and self.input_std is not None:
                    avg_std = self.input_std.mean().item()
                    in_ch = w_pot.shape[1]
                    alpha = self.alpha
                    bias_adjusted = self.bias.clone()
                    
                    for c in range(in_ch):
                        mean_c = self.input_mean[c].item()
                        weight_sum_c = w_pot[:, c, :].sum(dim=1) * alpha  # [out_ch]
                        bias_adjusted = bias_adjusted - (mean_c / avg_std) * weight_sum_c
                    
                    if DEBUG:
                        print(f"  [First layer bias absorption - {in_ch} channels, avg_std={avg_std:.4f}]")
                        print(f"    input_mean={self.input_mean.tolist()}")
                        print(f"    original bias sample: [{self.bias[0].item():.4f}, {self.bias[1].item():.4f}, ...]")
                        print(f"    adjusted bias sample: [{bias_adjusted[0].item():.4f}, {bias_adjusted[1].item():.4f}, ...]")
                else:
                    # No standardization - use bias as-is
                    bias_adjusted = self.bias
                    if DEBUG:
                        print(f"  [First layer - no standardization]")
                
                bias_int = round_ste(bias_adjusted * act_scale)
                
                if DEBUG:
                    print(f"    bias_int sample: [{bias_int[0].item():.0f}, {bias_int[1].item():.0f}, ...]")
            else:
                # Other layers: simple bias scaling
                bias_int = round_ste(self.bias * act_scale)
                if DEBUG:
                    print(f"  bias_int: act_scale={act_scale.item():.4f}, range=[{bias_int.min():.0f}, {bias_int.max():.0f}]")
            
            # Add bias (broadcast over length dimension)
            acc = acc + bias_int.view(1, -1, 1)
            
            if DEBUG:
                print(f"  acc after bias: range=[{acc.min():.0f}, {acc.max():.0f}]")
        
        # === Step 5: Clamp (ReLU) ===
        if not is_last:
            # ReLU: clamp to [0, 127]
            out = clamp_ste(acc, 0.0, 127.0)
            if DEBUG:
                print(f"  out after ReLU clamp: range=[{out.min():.0f}, {out.max():.0f}]")
        else:
            # Last layer: no ReLU, output raw logits
            out = acc
            if DEBUG:
                print(f"  out (last layer, no clamp): range=[{out.min():.0f}, {out.max():.0f}]")
        
        # === Step 6: Convert back to float for next layer ===
        # Next layer expects: int_value / act_scale
        if self.act_scale is not None and not is_last:
            out = out / self.act_scale
            if DEBUG:
                print(f"  out after /act_scale: range=[{out.min():.4f}, {out.max():.4f}]")
        
        return out

    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        s = (f'{self.in_channels}, {self.out_channels}, '
             f'kernel_size={self.kernel_size}, stride={self.stride}')
        if self.padding != 0:
            s += f', padding={self.padding}'
        if self.dilation != 1:
            s += f', dilation={self.dilation}'
        if self.groups != 1:
            s += f', groups={self.groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.quantize:
            s += f', quantize=True, encoding={self.encoding}'
        if self.use_integer_sim:
            s += ', integer_sim=True'
        return s
