"""PoT Add layer for skip/residual connections."""

import torch
import torch.nn as nn
import math


class PoTAdd(nn.Module):
    """Add layer for residual/skip connections with scale alignment.
    
    Skip connection에서 두 branch의 scale이 다를 수 있음:
        - x: 원래 입력 (scale_x)
        - y: conv 거친 출력 (scale_y)
    
    이 레이어는 scale 정합 후 더하기를 수행:
        output = rescale(x) + y
    
    rescale은 정수 MUL + shift로 구현:
        x_aligned = (x * rescale_mult) >> rescale_shift
    
    컴파일 타임에 rescale_mult, rescale_shift 계산.
    런타임에 float 연산 없음.
    
    사용 예:
        # ResNet block
        identity = x
        out = conv2(relu(conv1(x)))
        out = add_layer(identity, out)  # identity + out with scale alignment
        out = relu(out)
    """
    
    def __init__(self):
        """Initialize PoTAdd layer."""
        super().__init__()
        
        # Scale alignment: x_aligned = (x * rescale_mult) >> rescale_shift
        self.register_buffer('rescale_mult', torch.tensor(128))  # 기본값: 1.0 * 128
        self.register_buffer('rescale_shift', torch.tensor(7))   # 기본값: >>7
        
        # Activation scale for output (set during calibration)
        self.register_buffer('act_scale', None)
        
        # Scale info for the two inputs (set during calibration)
        self.register_buffer('scale_x', None)  # scale of first input (skip)
        self.register_buffer('scale_y', None)  # scale of second input (conv output)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass: aligned add.
        
        Args:
            x: First input (typically skip/identity branch)
            y: Second input (typically conv output)
            
        Returns:
            x + y with scale alignment applied to x
        """
        # QAT mode: simulate integer rescale
        if getattr(self, 'quantize', False) and self.scale_x is not None:
            # Simulate: x_aligned = (x * rescale_mult) >> rescale_shift
            # In float: x_aligned = x * (rescale_mult / 2^rescale_shift)
            ratio = self.rescale_mult.float() / (1 << self.rescale_shift.item())
            x = x * ratio
        
        return x + y
    
    def set_scales(self, scale_x: float, scale_y: float):
        """Set input scales and compute rescale_mult/rescale_shift.
        
        C 코드: skip_rescaled = (skip_int * mult) >> shift
        skip을 conv scale 기준으로 맞추려면:
            ratio = scale_y / scale_x  (conv/skip)
        
        Args:
            scale_x: Activation scale of first input (skip branch)
            scale_y: Activation scale of second input (conv branch)
        """
        self.scale_x = torch.tensor(scale_x)
        self.scale_y = torch.tensor(scale_y)
        
        # C 코드와 일치: skip을 conv scale 기준으로 변환
        ratio = scale_y / scale_x
        
        # 정수 양자화: ratio ≈ rescale_mult / 2^rescale_shift
        # mult = ratio * 2^shift, shift를 조정하여 mult를 1~255 범위로
        base_shift = 7
        mult = round(ratio * (1 << base_shift))
        
        # mult가 너무 크면 shift 감소 (mult = ratio * 2^shift)
        while mult > 255 and base_shift > 0:
            base_shift -= 1
            mult = round(ratio * (1 << base_shift))
        
        # mult가 너무 작으면 shift 증가
        while mult < 32 and base_shift < 15:
            base_shift += 1
            mult = round(ratio * (1 << base_shift))
        
        # clamp mult to safe range
        mult = max(1, min(255, mult))
        
        self.rescale_mult = torch.tensor(mult)
        self.rescale_shift = torch.tensor(base_shift)
        
        # Output scale is same as y's scale (after alignment)
        self.act_scale = torch.tensor(scale_y)
    
    def extra_repr(self) -> str:
        """String representation."""
        s = f"rescale_mult={self.rescale_mult.item()}, rescale_shift={self.rescale_shift.item()}"
        if self.scale_x is not None:
            ratio = self.scale_x.item() / self.scale_y.item()
            approx = self.rescale_mult.item() / (1 << self.rescale_shift.item())
            s += f", ratio={ratio:.3f}, approx={approx:.3f}"
        return s
