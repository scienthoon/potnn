"""Power-of-Two (PoT) quantization functions."""

import torch
import numpy as np
from typing import Union, List


# =============================================================================
# 인코딩별 레벨셋 정의
# =============================================================================

ENCODING_LEVELS = {
    # unroll: fp130 + Zero (17레벨) - 민감층용, 최고 정확도
    'unroll': [0, 1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64, 128, -128],
    
    # fp130: FP1.3.0 형식 (16레벨, Zero 없음) - Dense 레이어용
    'fp130': [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64, 128, -128],
    
    # 5level: 희소 레이어용 (5레벨, Zero 있음)
    '5level': [-8, -1, 0, 1, 8],
    
    # 2bit: 최소 메모리용 (4레벨, Zero 없음)
    '2bit': [-2, -1, 1, 2],
    
    # ternary: 최소 메모리 + Zero (3레벨)
    'ternary': [-1, 0, 1],
}

# 인코딩별 positive 값 (0 제외, 양자화용)
ENCODING_POS_VALUES = {
    'unroll': [0, 1, 2, 4, 8, 16, 32, 64, 128],
    'fp130': [1, 2, 4, 8, 16, 32, 64, 128],
    '5level': [0, 1, 8],
    '2bit': [1, 2],
    'ternary': [0, 1],
}

# 인코딩별 Zero 포함 여부
ENCODING_HAS_ZERO = {
    'unroll': True,
    'fp130': False,
    '5level': True,
    '2bit': False,
    'ternary': True,
}


def get_pot_values(encoding: str = 'unroll') -> torch.Tensor:
    """Get PoT values for given encoding.

    Args:
        encoding: Encoding type ('unroll', 'fp130', '5level', '2bit', 'ternary')

    Returns:
        Tensor of PoT values
    """
    if encoding not in ENCODING_LEVELS:
        raise ValueError(
            f"Unsupported encoding: {encoding}. "
            f"Must be one of {list(ENCODING_LEVELS.keys())}"
        )
    
    return torch.tensor(ENCODING_LEVELS[encoding], dtype=torch.float32)


def get_pot_pos_values(encoding: str = 'unroll') -> torch.Tensor:
    """Get positive PoT values (including 0 if applicable) for given encoding.

    Args:
        encoding: Encoding type

    Returns:
        Tensor of positive PoT values for quantization
    """
    if encoding not in ENCODING_POS_VALUES:
        raise ValueError(
            f"Unsupported encoding: {encoding}. "
            f"Must be one of {list(ENCODING_POS_VALUES.keys())}"
        )
    
    return torch.tensor(ENCODING_POS_VALUES[encoding], dtype=torch.float32)


# Backward compatibility: levels -> encoding 매핑
def _levels_to_encoding(levels: int) -> str:
    """Convert legacy levels parameter to encoding string."""
    mapping = {
        3: 'ternary',
        5: '5level',
        11: 'unroll',
        17: 'unroll',
        16: 'fp130',
    }
    if levels not in mapping:
        raise ValueError(f"Unsupported levels: {levels}")
    return mapping[levels]


def quantize_to_pot(
    weight: torch.Tensor, 
    alpha: float, 
    encoding: str = 'unroll',
    levels: int = None  # Backward compatibility
) -> torch.Tensor:
    """Quantize weight tensor to Power-of-Two values.

    This function quantizes float weights to the nearest PoT value.
    The quantization is done by finding the nearest PoT value for each weight.

    Args:
        weight: Weight tensor to quantize (float)
        alpha: Scaling factor to normalize weights before quantization
        encoding: Encoding type ('unroll', 'fp130', '5level', '2bit', 'ternary')
        levels: (Deprecated) Legacy parameter, use encoding instead

    Returns:
        Quantized weight tensor with PoT values
    """
    # Backward compatibility
    if levels is not None:
        encoding = _levels_to_encoding(levels)
    
    # Get positive PoT values for this encoding
    pot_pos = get_pot_pos_values(encoding).to(weight.device)
    has_zero = ENCODING_HAS_ZERO[encoding]

    # Separate sign and magnitude
    sign = torch.sign(weight)
    abs_weight = torch.abs(weight / alpha)

    # Find nearest PoT value for absolute weights
    abs_weight_flat = abs_weight.reshape(-1, 1)
    pot_pos_flat = pot_pos.reshape(1, -1)
    distances = torch.abs(abs_weight_flat - pot_pos_flat)
    indices = torch.argmin(distances, dim=1)

    # Get quantized absolute values
    quantized_abs = pot_pos[indices].reshape(weight.shape)

    # Restore sign
    w_q = sign * quantized_abs
    
    # Zero-free 인코딩의 경우, 0에 가장 가까운 값으로 대체
    if not has_zero:
        min_val = pot_pos[pot_pos > 0].min()
        zero_mask = (w_q == 0)
        
        if encoding == 'fp130':
            # fp130: Use alternating +1/-1 for zeros to reduce bias
            # This matches C export packing logic if zeros were preserved
            flat_indices = torch.arange(w_q.numel(), device=w_q.device)
            toggle_vals = torch.where(flat_indices % 2 == 0, min_val, -min_val)
            toggle_vals = toggle_vals.reshape(w_q.shape)
            w_q = torch.where(zero_mask, toggle_vals, w_q)
        else:
            # 기본: 최소 양수값으로 대체 (+1/-1은 sign 유지)
            # sign이 0인 경우 (원래 weight가 0) → +min_val로
            w_q = torch.where(zero_mask & (sign >= 0), min_val, w_q)
            w_q = torch.where(zero_mask & (sign < 0), -min_val, w_q)
    
    # 5level 인코딩 constraint는 forward에서 enforce_5level_constraint 플래그로 처리
    # (torch.export 호환성을 위해 여기서는 적용 안함)

    return w_q


def _apply_5level_constraint_numpy(w_q: torch.Tensor) -> torch.Tensor:
    """Apply 5-level constraint (non-differentiable, for export only)."""
    w_np = w_q.clone()
    
    if w_q.dim() == 2:  # Linear
        out_features, in_features = w_q.shape
        for o in range(out_features):
            zero_run = 0
            for i in range(in_features):
                if w_np[o, i] == 0:
                    zero_run += 1
                    if zero_run > 3:
                        w_np[o, i] = 1.0
                        zero_run = 0
                else:
                    zero_run = 0
    elif w_q.dim() == 4:  # Conv2d
        out_ch = w_q.shape[0]
        for oc in range(out_ch):
            w_flat = w_np[oc].flatten()
            zero_run = 0
            for i in range(len(w_flat)):
                if w_flat[i] == 0:
                    zero_run += 1
                    if zero_run > 3:
                        w_flat[i] = 1.0
                        zero_run = 0
                else:
                    zero_run = 0
            w_np[oc] = w_flat.view(w_q[oc].shape)
    
    return w_np


def apply_5level_zero_constraint(w_q: torch.Tensor) -> torch.Tensor:
    """Apply 5-level encoding constraint: max 3 consecutive zeros.
    
    5-level encoding uses 2-bit skip field (0~3), so 4+ consecutive zeros
    cannot be encoded. This function replaces the 4th+ consecutive zero
    with +1 (smallest non-zero value).
    
    Works with 2D (Linear) and 4D (Conv2d) tensors.
    For Conv2d, applies constraint per output filter.
    
    Args:
        w_q: Quantized weight tensor
        
    Returns:
        Constrained weight tensor (differentiable via STE)
    """
    # Detach for modification, but keep gradient flow via STE
    w_constrained = w_q.clone()
    
    if w_q.dim() == 2:  # Linear: (out_features, in_features)
        out_features, in_features = w_q.shape
        for o in range(out_features):
            zero_run = 0
            for i in range(in_features):
                if w_constrained[o, i] == 0:
                    zero_run += 1
                    if zero_run > 3:
                        w_constrained[o, i] = 1.0  # Replace with +1
                        zero_run = 0
                else:
                    zero_run = 0
                    
    elif w_q.dim() == 4:  # Conv2d: (out_ch, in_ch, kh, kw)
        out_ch = w_q.shape[0]
        flat_size = w_q[0].numel()
        for oc in range(out_ch):
            w_flat = w_constrained[oc].flatten()
            zero_run = 0
            for i in range(flat_size):
                if w_flat[i] == 0:
                    zero_run += 1
                    if zero_run > 3:
                        w_flat[i] = 1.0
                        zero_run = 0
                else:
                    zero_run = 0
            w_constrained[oc] = w_flat.view(w_q[oc].shape)
    
    # STE: use constrained in forward, but gradient flows to original
    return w_q + (w_constrained - w_q).detach()


class PoTQuantizeSTE(torch.autograd.Function):
    """Straight-Through Estimator for PoT quantization.

    Forward pass: quantize to PoT values
    Backward pass: pass gradient through unchanged
    """

    @staticmethod
    def forward(ctx, weight, alpha, encoding):
        """Forward pass: quantize to PoT values."""
        return quantize_to_pot(weight, alpha, encoding=encoding)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: gradient passes through unchanged (STE)."""
        return grad_output, None, None


def quantize_to_pot_ste(
    weight: torch.Tensor, 
    alpha: torch.Tensor, 
    encoding: str = 'unroll',
    levels: int = None  # Backward compatibility
) -> torch.Tensor:
    """Quantize weight tensor to PoT values with Straight-Through Estimator.

    This function applies PoT quantization in the forward pass while allowing
    gradients to flow through unchanged in the backward pass (STE).

    Args:
        weight: Weight tensor to quantize
        alpha: Scaling factor (learnable parameter)
        encoding: Encoding type ('unroll', 'fp130', '5level', '2bit', 'ternary')
        levels: (Deprecated) Legacy parameter, use encoding instead

    Returns:
        Quantized weight tensor with STE for gradients
    """
    # Backward compatibility
    if levels is not None:
        encoding = _levels_to_encoding(levels)
    
    return PoTQuantizeSTE.apply(weight, alpha, encoding)


class QuantizeActivationSTE(torch.autograd.Function):
    """Straight-Through Estimator for activation quantization.
    
    Forward pass: quantize to int8 range using half-up rounding (C style)
    Backward pass: pass gradient through unchanged (STE)
    
    CRITICAL: Without STE, rounding has zero gradient,
    which blocks gradient flow to earlier layers and causes QAT to fail.
    
    NOTE: Uses floor(x + 0.5) for half-up rounding to match C behavior:
        C: (int)(x + 0.5) or (x * scale + round_const) >> shift
    """
    
    @staticmethod
    def forward(ctx, x, scale):
        """Forward pass: quantize activation to int8 range with half-up rounding."""
        # Half-up rounding to match C: floor(x + 0.5)
        return torch.floor(x * scale + 0.5).clamp(-128, 127) / scale
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: gradient passes through unchanged (STE)."""
        return grad_output, None


def quantize_activation_ste(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize activation tensor with Straight-Through Estimator.
    
    This function applies int8 quantization in the forward pass while allowing
    gradients to flow through unchanged in the backward pass.
    
    Args:
        x: Activation tensor to quantize
        scale: Scale factor (127.0 / max_activation)
    
    Returns:
        Quantized activation tensor with STE for gradients
    """
    return QuantizeActivationSTE.apply(x, scale)


class QuantizeActivationCAlignedSTE(torch.autograd.Function):
    """C-Aligned activation quantization with STE.
    
    Uses integer scale_int and shift to match C code exactly:
        C: out = (acc * scale_int + (1 << (shift-1))) >> shift
    
    This eliminates the floating-point precision gap between QAT and C.
    """
    
    @staticmethod
    def forward(ctx, x, scale_int, shift, act_scale):
        """Forward: C-style requantize."""
        # C: (x * scale_int + round_const) >> shift
        if shift > 0:
            round_const = 1 << (shift - 1)
        else:
            round_const = 0
        
        divisor = float(1 << shift)
        
        # Scale x to integer range first (matching C's integer accumulator)
        x_scaled = x * act_scale
        
        # Apply C-style requantize
        numerator = x_scaled * float(scale_int) + float(round_const)
        result = torch.floor(numerator / divisor)
        result = result.clamp(-128, 127)
        
        # Convert back to float range for next layer
        return result / act_scale
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward: STE - gradient passes through."""
        return grad_output, None, None, None


def quantize_activation_c_aligned_ste(
    x: torch.Tensor, 
    scale_int: int, 
    shift: int, 
    act_scale: torch.Tensor
) -> torch.Tensor:
    """C-aligned activation quantization with STE.
    
    Matches C code: out = (acc * scale_int + round) >> shift
    
    Args:
        x: Activation tensor
        scale_int: Integer scale (from compute_scale_params)
        shift: Shift amount (from compute_scale_params)
        act_scale: Original float act_scale (for scaling back)
    
    Returns:
        Quantized activation matching C behavior
    """
    return QuantizeActivationCAlignedSTE.apply(x, scale_int, shift, act_scale)


def pack_2bit(weights: np.ndarray) -> np.ndarray:
    """Pack 4-level weights into 2-bit representation.

    Encoding:
        -1 -> 0b00
         0 -> 0b01
         1 -> 0b10
         2 -> 0b11

    Args:
        weights: Array of weights with values in {-1, 0, 1, 2}

    Returns:
        Packed uint8 array (4 weights per byte)
    """
    # Map weights to 2-bit codes
    mapping = {-1: 0b00, 0: 0b01, 1: 0b10, 2: 0b11}

    # Flatten weights
    w_flat = weights.flatten()

    # Pad to multiple of 4
    pad_len = (4 - len(w_flat) % 4) % 4
    if pad_len > 0:
        w_flat = np.concatenate([w_flat, np.zeros(pad_len)])

    # Pack 4 weights per byte
    packed = []
    for i in range(0, len(w_flat), 4):
        byte = 0
        for j in range(4):
            code = mapping.get(int(w_flat[i+j]), 0b01)  # Default to 0 if not found
            byte |= (code << (j * 2))
        packed.append(byte)

    return np.array(packed, dtype=np.uint8)


def unpack_2bit(packed: np.ndarray, num_weights: int) -> np.ndarray:
    """Unpack 2-bit representation to 4-level weights.

    Args:
        packed: Packed uint8 array
        num_weights: Number of weights to unpack

    Returns:
        Array of weights with values in {-1, 0, 1, 2}
    """
    # Inverse mapping
    mapping = {0b00: -1, 0b01: 0, 0b10: 1, 0b11: 2}

    weights = []
    for byte in packed:
        for j in range(4):
            code = (byte >> (j * 2)) & 0b11
            weights.append(mapping[code])

    return np.array(weights[:num_weights])