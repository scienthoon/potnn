"""Base class for all PoT (Power-of-Two) quantized layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoTLayerBase(nn.Module):
    """Base class for all PoT layers with alpha scaling and activation quantization.

    This class provides:
    - Alpha scaling parameter (learnable)
    - Activation scale (fixed after calibration)
    - QAT (Quantization-Aware Training) mode management
    - Alpha regularization loss
    - Integer simulation mode for C-compatible inference
    """

    def __init__(self, encoding='unroll'):
        """Initialize PoT layer base.

        Args:
            encoding: Encoding type for weight quantization
                   - 'unroll': 17레벨 {0, ±1, ±2, ..., ±128} (default)
                   - 'fp130':  16레벨 {±1, ±2, ..., ±128} (Zero 없음)
                   - '5level': 5레벨 {-8, -1, 0, 1, 8}
                   - '2bit':   4레벨 {-2, -1, 1, 2} (Zero 없음)
                   - 'ternary': 3레벨 {-1, 0, 1}
        """
        super().__init__()
        self.encoding = encoding

        # Alpha scaling parameter (learnable)
        # raw_alpha → softplus → clamp(0.01) → alpha
        self.raw_alpha = nn.Parameter(torch.tensor(0.5))

        # Alpha initial value for regularization
        # This will be updated during calibration to match the initialized alpha
        self.register_buffer('alpha_init', torch.tensor(0.5))

        # Activation scale (fixed after calibration)
        self.register_buffer('act_scale', None)

        # QAT mode flag
        self.quantize = False
        
        # === Integer Simulation Parameters ===
        # These enable C-compatible integer arithmetic simulation
        
        # Layer position flags
        self.register_buffer('is_first_layer', torch.tensor(False))
        self.register_buffer('is_last_layer', torch.tensor(False))
        
        # Previous layer's act_scale (for scale chain)
        self.register_buffer('prev_act_scale', None)
        
        # Input std (for first layer standardization absorption)
        # Per-channel tensor [in_ch] or None
        self.register_buffer('input_std', None)
        
        # Input mean (for first layer bias adjustment)
        # Per-channel tensor [in_ch] or None
        self.register_buffer('input_mean', None)
        
        # Pre-computed integer scale parameters
        self.register_buffer('scale_int', None)
        self.register_buffer('shift', None)
        
        # Integer simulation mode flag
        self.use_integer_sim = False
        
        # 5level encoding constraint flag
        # When True, enforces max 3 consecutive zeros (skip field is 2 bits)
        self.enforce_5level_constraint = False

    @property
    def alpha(self):
        """Get positive alpha value using softplus + clamp.

        Returns:
            Positive alpha value for scaling PoT weights.
        """
        return F.softplus(self.raw_alpha).clamp(min=0.01)

    def calibrate(self, act_max):
        """Set activation scale based on calibration.

        Args:
            act_max: Maximum activation value from calibration.
        """
        if act_max > 0:
            self.act_scale = torch.tensor(127.0 / act_max)
        else:
            self.act_scale = torch.tensor(1.0)

    def prepare_qat(self):
        """Enable QAT (Quantization-Aware Training) mode."""
        self.quantize = True

    def alpha_reg_loss(self, lambda_reg=0.01):
        """Calculate alpha regularization loss.

        This loss encourages alpha to stay close to its initial value,
        preventing it from drifting too far during training.

        Args:
            lambda_reg: Regularization strength (default: 0.01)

        Returns:
            Alpha regularization loss value.
        """
        # Use the stored alpha_init which is set during calibration
        return lambda_reg * (self.alpha - self.alpha_init) ** 2
    
    # === Integer Simulation Methods ===
    
    def set_layer_position(self, is_first: bool, is_last: bool):
        """Set layer position in the network.
        
        Args:
            is_first: True if this is the first PoT layer (input is uint8)
            is_last: True if this is the last PoT layer (no ReLU)
        """
        self.is_first_layer = torch.tensor(is_first)
        self.is_last_layer = torch.tensor(is_last)

    def set_prev_act_scale(self, prev_scale: float):
        """Set previous layer's activation scale.
        
        Args:
            prev_scale: Previous layer's act_scale value
        """
        if prev_scale is not None:
            self.prev_act_scale = torch.tensor(prev_scale)
        else:
            self.prev_act_scale = None

    def set_input_std(self, std, mean=None):
        """Set input statistics for first layer.
        
        Args:
            std: Standard deviation - float (single channel) or List[float] (multi-channel)
            mean: Mean values - float (single channel) or List[float] (multi-channel)
        """
        # Convert to per-channel tensor
        if isinstance(std, (int, float)):
            self.input_std = torch.tensor([float(std)])
        else:
            self.input_std = torch.tensor([float(s) for s in std])
        
        if mean is not None:
            if isinstance(mean, (int, float)):
                self.input_mean = torch.tensor([float(mean)])
            else:
                self.input_mean = torch.tensor([float(m) for m in mean])
        else:
            self.input_mean = None

    def compute_integer_params(self):
        """Compute integer scale parameters for C-compatible inference.
        
        MUST match export.py calculate_combined_scales() exactly!
        
        Returns:
            (scale_int, shift) tuple
        """
        scale_int, shift, _ = self._compute_scale_and_shift()
        
        self.scale_int = torch.tensor(scale_int, device=self.raw_alpha.device)
        self.shift = torch.tensor(shift, device=self.raw_alpha.device)
        
        return scale_int, shift

    def _compute_scale_and_shift(self):
        """Internal method to compute scale_int and shift dynamically.
        
        Returns:
            (scale_int, shift, combined_scale)
        """
        # self.alpha is already softplus(raw_alpha).clamp(0.01) via property
        alpha = self.alpha.item()
        act_scale = self.act_scale.item() if self.act_scale is not None else None
        
        is_first = self.is_first_layer.item()
        
        # Calculate combined_scale - EXACTLY like export.py
        if is_first:
            # Use average std for combined_scale (matches export.py)
            if self.input_std is not None:
                input_std = self.input_std.mean().item()
            else:
                input_std = 1.0
            if act_scale is not None:
                combined_scale = alpha * act_scale / input_std
            else:
                combined_scale = alpha / input_std
        else:
            prev_scale = self.prev_act_scale.item() if self.prev_act_scale is not None else 1.0
            if act_scale is not None:
                combined_scale = alpha * act_scale / prev_scale
            else:
                combined_scale = alpha / prev_scale
        
        # Determine shift - EXACTLY like export.py
        base_shift = 0
        scale_magnitude = abs(combined_scale)
        
        # Target: scale_int around 64-512 for precision (export.py uses 64-512)
        while scale_magnitude < 64 and base_shift < 24:
            scale_magnitude *= 2
            base_shift += 1
        while scale_magnitude > 512 and base_shift > 0:
            scale_magnitude /= 2
            base_shift -= 1
        
        # For first layer, add +8 for /256 absorption
        if is_first:
            combined_shift = base_shift + 8
        else:
            combined_shift = base_shift
        
        # Calculate integer scale
        scale_int = round(combined_scale * (1 << base_shift))
        
        return scale_int, combined_shift, combined_scale