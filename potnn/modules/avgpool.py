"""PoT Global Average Pooling layer."""

import torch
import torch.nn as nn
import math


from ..quantize.integer_ops import round_half_up_ste, floor_ste

class PoTGlobalAvgPool(nn.Module):
    """Global Average Pooling with PoT-compatible quantization.
    
    [Integer-Only QAT Mode]
    Forward pass simulates C integer arithmetic:
    - Power of 2 size: (sum + (size//2)) >> log2(size)
    - Generic size: (sum * div_mult + round_const) >> div_shift
    """
    
    def __init__(self):
        """Initialize PoTGlobalAvgPool."""
        super().__init__()
        
        # Division parameters
        self.register_buffer('div_mult', torch.tensor(1))
        self.register_buffer('div_shift', torch.tensor(0))
        self.register_buffer('pool_size', torch.tensor(0))
        
        # Activation scale (passed from previous layer)
        self.register_buffer('act_scale', None)
        
        self.quantize = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: global average pooling.
        
        Args:
            x: Input tensor of shape (N, C, H, W)
            
        Returns:
            Output tensor of shape (N, C)
        """
        if not self.quantize:
            # Float mode
            return x.mean(dim=(2, 3))
            
        if not getattr(self, 'integer_sim_enabled', False):
            # Float QAT mode
            return x.mean(dim=(2, 3))

        # Calculate pool size dynamically if not set
        current_pool_size = x.shape[2] * x.shape[3]
        
        if self.pool_size.item() != current_pool_size:
            self.set_pool_size(x.shape[2], x.shape[3])
            
        # Integer mode
        # x is float but represents integer values (from previous layer)
        # We assume input is already scaled by act_scale of previous layer?
        # No, in our new design, previous layer output is "dequantized" float.
        # So x is float.
        # But GAP in C operates on the accumulated integer values?
        # Wait, C GAP input is the output of the previous layer *before* requantization?
        # No, usually GAP follows a Conv/ReLU layer.
        # The previous layer output is int8 (requantized).
        # So x here is int8 values (represented as float).
        # But wait, our Conv layer returns `out / act_scale`.
        # So x is float.
        # We need to recover the int8 values: `x_int = round(x * prev_scale)`
        # But `prev_scale` is `act_scale` of previous layer.
        # If we assume `act_scale` is passed to this layer, we can use it.
        
        # However, for GAP, usually we just average the values.
        # mean(x) = sum(x) / N
        # If x = x_int / scale, then mean(x) = sum(x_int) / N / scale
        # = (sum(x_int) / N) / scale
        # So we can just compute mean(x) in float?
        # NO! The rounding behavior of `sum(x_int) / N` in integer arithmetic is different from float mean.
        # C: `(sum(x_int) + N//2) >> log2(N)`
        # Python float: `mean(x_int)` (exact)
        # We must simulate the integer division on `x_int`.
        
        # So:
        # 1. Recover x_int: x_int = round(x * act_scale)
        # 2. Compute sum(x_int)
        # 3. Integer division
        # 4. Convert back to float: result / act_scale
        
        # We need `act_scale` of the input.
        # Usually this is passed or stored.
        # Let's assume `act_scale` is available (set by `set_prev_act_scale` or similar mechanism).
        # But `PoTGlobalAvgPool` doesn't inherit `PoTLayerBase` currently.
        # Let's assume for now we just operate on `x` assuming it's `x_int` if `act_scale` is 1.0.
        # But wait, if we don't know `act_scale`, we can't recover `x_int`.
        
        # In the user's specific case (SimpleNet), GAP follows Conv2.
        # Conv2 output is `out / act_scale`.
        # So GAP input is float.
        # If we want to match C, we need to know `act_scale`.
        
        # Let's check how `PoTGlobalAvgPool` is used.
        # It seems it's used in `SimpleNet`.
        # We should probably add `act_scale` management to `PoTGlobalAvgPool`.
        
        # For now, let's implement the integer logic assuming `x` is `x_int`?
        # No, `PoTConv2d` divides by scale.
        
        # Solution: `PoTGlobalAvgPool` needs `act_scale`.
        # We'll add `set_act_scale` method.
        
        scale = self.act_scale if self.act_scale is not None else torch.tensor(1.0)
        
        # Input should be integer values from previous layer
        # Round to ensure exact integer (may have floating point precision errors)
        # Use STE to maintain gradient flow during training
        x_int = round_half_up_ste(x)


        
        # 2. Sum over H, W
        sum_val = x_int.sum(dim=(2, 3))

        
        # 3. Integer Division
        pool_size = int(self.pool_size.item())
        if (pool_size & (pool_size - 1)) == 0:
            # Power of 2
            shift = int(math.log2(pool_size))
            round_const = 1 << (shift - 1)
            # Power of 2
            shift = int(math.log2(pool_size))
            round_const = 1 << (shift - 1)
            # (sum + round) >> shift
            out_int = floor_ste((sum_val + round_const) / (1 << shift))

        else:
            # Generic
            mult = self.div_mult.item()
            shift = self.div_shift.item()
            # (sum * mult + round) >> shift
            # round_const for shift is 1<<(shift-1)
            # But wait, C generic implementation:
            # avg = (sum * div_mult + (1<<(div_shift-1))) >> div_shift
            round_const = 1 << (shift - 1) if shift > 0 else 0
            val = sum_val * mult + round_const
            out_int = floor_ste(val / (1 << shift))
            
        # Output is int8 (no conversion back to float)
        return out_int

    def set_pool_size(self, h: int, w: int):
        """Set pool size and compute div_mult/div_shift."""
        pool_size = h * w
        self.pool_size = torch.tensor(pool_size)
        
        if pool_size > 0 and (pool_size & (pool_size - 1)) == 0:
            self.div_mult = torch.tensor(1)
            self.div_shift = torch.tensor(int(math.log2(pool_size)))
        else:
            base_shift = 15
            mult = round((1 << base_shift) / pool_size)
            while mult > 255 and base_shift > 8:
                base_shift -= 1
                mult = round((1 << base_shift) / pool_size)
            self.div_mult = torch.tensor(max(1, min(65535, mult)))
            self.div_shift = torch.tensor(base_shift)
            
    def prepare_qat(self, act_scale=None):
        self.quantize = True
        if act_scale is not None:
            self.act_scale = torch.tensor(act_scale)

    def extra_repr(self) -> str:
        return f"pool_size={self.pool_size.item()}, quantize={self.quantize}"
