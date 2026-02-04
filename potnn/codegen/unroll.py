"""Generate unrolled C code for PoT layers."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

from ..quantize.pot import quantize_to_pot


def generate_unrolled_layer(layer_info: Dict[str, Any]) -> str:
    """Generate unrolled C code for a layer.

    In unrolled mode, each weight becomes a direct instruction
    with the shift amount embedded as an immediate value.
    Zero weights are omitted entirely.

    For Conv2d: Uses C loops for spatial positions, unrolls channels.
    For Linear: Fully unrolls all weights.

    Args:
        layer_info: Dictionary with layer information

    Returns:
        C code for the unrolled layer
    """
    name = layer_info['name']
    layer_type = layer_info['type']
    weight = layer_info['weight']
    alpha = layer_info['alpha']
    bias = layer_info.get('bias', None)
    use_relu = layer_info.get('has_relu', False)
    is_last = layer_info.get('is_last', False)
    
    # Layer dimensions for Conv2d
    in_h = layer_info.get('in_h', 0)
    in_w = layer_info.get('in_w', 0)
    out_h = layer_info.get('out_h', 0)
    out_w = layer_info.get('out_w', 0)
    stride = layer_info.get('stride', 1)
    padding = layer_info.get('padding', 0)

    # Quantize weights
    with torch.no_grad():
        w_q = quantize_to_pot(weight, alpha, levels=11).numpy()

    # Get act_scale for bias scaling (None for last layer)
    act_scale = layer_info.get('act_scale')
    if act_scale is None:
        act_scale = 1.0  # Last layer: no scaling

    # Debug output for codegen
    print(f"\n    [DEBUG] Generating unrolled code for {name}:")
    print(f"      Weight shape: {weight.shape}")
    print(f"      PoT weight unique values: {sorted(set(w_q.flatten()))}")
    print(f"      Non-zero weights: {np.count_nonzero(w_q)} / {w_q.size}")
    if bias is not None:
        all_bias_scaled = [int(b.item() * act_scale + 0.5) for b in bias]
        print(f"      act_scale: {act_scale:.4f}")
        print(f"      All scaled bias values: {all_bias_scaled}")
    else:
        print(f"      No bias")

    code = f"// {name} - Unrolled (11 levels)\n"

    if 'Linear' in layer_type:
        code += _generate_linear_unrolled(name, w_q, bias, use_relu, act_scale)
    elif 'Depthwise' in layer_type:
        # DepthwiseConv2d: use baseline depthwise generator
        from ..export import generate_depthwise_conv_layer
        return generate_depthwise_conv_layer(layer_info, is_first_layer=False)
    elif 'Conv2d' in layer_type:
        groups = layer_info.get('groups', 1)
        code += _generate_conv2d_unrolled(
            name, w_q, bias, use_relu,
            in_h, in_w, out_h, out_w, stride, padding, act_scale, groups
        )

    return code


def _generate_linear_unrolled(name: str, w_q: np.ndarray, bias, use_relu: bool, act_scale: float) -> str:
    """Generate unrolled code for Linear layer."""
    out_features, in_features = w_q.shape
    
    code = f"static void {name}_forward(const int8_t* input, int8_t* output) {{\n"

    for out_idx in range(out_features):
        code += f"    {{ // output[{out_idx}]\n"
        code += f"        int32_t acc = 0;\n"

        # Generate unrolled operations for non-zero weights
        for in_idx in range(in_features):
            w = int(w_q[out_idx, in_idx])
            if w == 0:
                continue
            code += _pot_operation_direct(in_idx, w)

        # Apply scale first
        code += f"        acc = scale_{name}(acc);\n"

        # Add bias after scale (scaled by act_scale)
        if bias is not None:
            bias_val = int(bias[out_idx].item() * act_scale + 0.5)
            if bias_val != 0:
                code += f"        acc += {bias_val};\n"
        
        # Apply ReLU if needed
        if use_relu:
            code += f"        if (acc < 0) acc = 0;\n"
        
        # Clamp and store
        code += f"        output[{out_idx}] = (int8_t)(acc > 127 ? 127 : (acc < -128 ? -128 : acc));\n"
        code += f"    }}\n"

    code += "}\n\n"
    return code


def _generate_conv2d_unrolled(
    name: str, w_q: np.ndarray, bias, use_relu: bool,
    in_h: int, in_w: int, out_h: int, out_w: int,
    stride: int, padding: int, act_scale: float, groups: int
) -> str:
    """Generate Conv2d code with C loops for positions, unrolled channels.
    
    This produces compact code like v4 generator:
    - Outer loops (oy, ox) are C for-loops
    - Channel/kernel operations are unrolled inside
    """
    out_channels, in_channels, kh, kw = w_q.shape
    
    code = f"static void {name}_forward(const int8_t* input, int8_t* output) {{\n"
    code += f"    // Conv2d: {in_channels}x{in_h}x{in_w} -> {out_channels}x{out_h}x{out_w}\n"
    code += f"    // Kernel: {kh}x{kw}, Stride: {stride}, Padding: {padding}\n"
    code += f"    int32_t acc;\n\n"
    
    # C loops for output positions
    code += f"    for (int oy = 0; oy < {out_h}; oy++) {{\n"
    code += f"        for (int ox = 0; ox < {out_w}; ox++) {{\n"
    
    # Unroll each output channel
    for oc in range(out_channels):
        code += f"            // Output channel {oc}\n"
        code += f"            acc = 0;\n"
        
        # Unroll kernel operations
        for ic in range(in_channels):
            for ky in range(kh):
                for kx in range(kw):
                    w = int(w_q[oc, ic, ky, kx])
                    if w == 0:
                        continue
                    
                    # Calculate offset from (oy, ox)
                    # Handle padding as int or tuple
                    pad_h = padding[0] if isinstance(padding, tuple) else padding
                    pad_w = padding[1] if isinstance(padding, tuple) else padding
                    ky_off = ky - pad_h  # -1, 0, 1 for 3x3 with pad=1
                    kx_off = kx - pad_w
                    
                    # Build input index expression
                    # in_y = oy * stride + ky_off
                    # in_x = ox * stride + kx_off
                    # idx = ic * in_h * in_w + in_y * in_w + in_x
                    
                    
                    # Group offset calculation
                    channels_per_group = in_channels
                    if groups == out_channels:
                        # Depthwise
                        group_ch_offset = oc * channels_per_group
                    elif groups > 1:
                        out_per_group = out_channels // groups
                        group_ch_offset = (oc // out_per_group) * channels_per_group
                    else:
                        group_ch_offset = 0
                    
                    ic_base = (group_ch_offset + ic) * in_h * in_w
                    
                    if stride == 1:
                        if ky_off == 0:
                            y_expr = "oy"
                        elif ky_off > 0:
                            y_expr = f"oy + {ky_off}"
                        else:
                            y_expr = f"oy - {-ky_off}"
                        
                        if kx_off == 0:
                            x_expr = "ox"
                        elif kx_off > 0:
                            x_expr = f"ox + {kx_off}"
                        else:
                            x_expr = f"ox - {-kx_off}"
                    else:
                        if ky_off == 0:
                            y_expr = f"oy * {stride}"
                        elif ky_off > 0:
                            y_expr = f"oy * {stride} + {ky_off}"
                        else:
                            y_expr = f"oy * {stride} - {-ky_off}"
                        
                        if kx_off == 0:
                            x_expr = f"ox * {stride}"
                        elif kx_off > 0:
                            x_expr = f"ox * {stride} + {kx_off}"
                        else:
                            x_expr = f"ox * {stride} - {-kx_off}"
                    
                    idx_expr = f"{ic_base} + ({y_expr}) * {in_w} + ({x_expr})"
                    
                    # Build boundary conditions
                    conditions = []
                    if ky_off < 0:
                        conditions.append(f"oy >= {-ky_off}")
                    elif ky_off > 0:
                        conditions.append(f"oy < {out_h - ky_off}")
                    
                    if kx_off < 0:
                        conditions.append(f"ox >= {-kx_off}")
                    elif kx_off > 0:
                        conditions.append(f"ox < {out_w - kx_off}")
                    
                    # Generate code
                    op = _pot_operation_expr(idx_expr, w)
                    
                    if conditions:
                        cond = " && ".join(conditions)
                        code += f"            if ({cond}) {op}\n"
                    else:
                        code += f"            {op}\n"
        
        # Apply scale first
        code += f"            acc = scale_{name}(acc);\n"
        
        # Add bias after scale (scaled by act_scale)
        if bias is not None:
            bias_val = int(bias[oc].item() * act_scale + 0.5)
            if bias_val != 0:
                code += f"            acc += {bias_val};\n"
        
        # Apply ReLU if needed
        if use_relu:
            code += f"            if (acc < 0) acc = 0;\n"
        
        # Store output
        out_base = oc * out_h * out_w
        code += f"            output[{out_base} + oy * {out_w} + ox] = (int8_t)(acc > 127 ? 127 : (acc < -128 ? -128 : acc));\n"
        code += f"\n"
    
    code += f"        }}\n"
    code += f"    }}\n"
    code += "}\n\n"
    return code


def _pot_operation_direct(idx: int, w: int) -> str:
    """Generate a PoT operation with direct index."""
    if w == 1:
        return f"        acc += (int32_t)input[{idx}];\n"
    elif w == -1:
        return f"        acc -= (int32_t)input[{idx}];\n"
    elif w == 2:
        return f"        acc += (int32_t)input[{idx}] << 1;\n"
    elif w == -2:
        return f"        acc -= (int32_t)input[{idx}] << 1;\n"
    elif w == 4:
        return f"        acc += (int32_t)input[{idx}] << 2;\n"
    elif w == -4:
        return f"        acc -= (int32_t)input[{idx}] << 2;\n"
    elif w == 8:
        return f"        acc += (int32_t)input[{idx}] << 3;\n"
    elif w == -8:
        return f"        acc -= (int32_t)input[{idx}] << 3;\n"
    elif w == 16:
        return f"        acc += (int32_t)input[{idx}] << 4;\n"
    elif w == -16:
        return f"        acc -= (int32_t)input[{idx}] << 4;\n"
    elif w == 32:
        return f"        acc += (int32_t)input[{idx}] << 5;\n"
    elif w == -32:
        return f"        acc -= (int32_t)input[{idx}] << 5;\n"
    elif w == 64:
        return f"        acc += (int32_t)input[{idx}] << 6;\n"
    elif w == -64:
        return f"        acc -= (int32_t)input[{idx}] << 6;\n"
    elif w == 128:
        return f"        acc += (int32_t)input[{idx}] << 7;\n"
    elif w == -128:
        return f"        acc -= (int32_t)input[{idx}] << 7;\n"
    return ""


def _pot_operation_expr(idx_expr: str, w: int) -> str:
    """Generate a PoT operation with index expression."""
    shift = _get_shift(abs(w))
    sign = "+" if w > 0 else "-"
    
    if shift == 0:
        return f"acc {sign}= (int32_t)input[{idx_expr}];"
    else:
        return f"acc {sign}= (int32_t)input[{idx_expr}] << {shift};"


def _get_shift(abs_w: int) -> int:
    """Get shift amount for absolute PoT value."""
    if abs_w == 1:
        return 0
    elif abs_w == 2:
        return 1
    elif abs_w == 4:
        return 2
    elif abs_w == 8:
        return 3
    elif abs_w == 16:
        return 4
    elif abs_w == 32:
        return 5
    elif abs_w == 64:
        return 6
    elif abs_w == 128:
        return 7
    return 0


# =============================================================================
# OPTIMIZED UNROLL GENERATION (Zero-Padding only)
# Eliminates boundary if statements by using padded buffer
# =============================================================================

def generate_unrolled_layer_optimized(layer_info: Dict[str, Any], is_first_layer: bool = False) -> str:
    """Generate optimized unrolled C code with Zero-Padding.
    
    Optimization applied:
    - Zero-Padding: Eliminates all boundary check if statements
    
    Args:
        layer_info: Dictionary with layer information
        is_first_layer: If True, input type is uint8_t (image input)
    
    Returns:
        C code for the optimized layer
    """
    name = layer_info['name']
    layer_type = layer_info['type']
    weight = layer_info['weight']
    alpha = layer_info['alpha']
    bias = layer_info.get('bias', None)
    use_relu = layer_info.get('has_relu', False)
    is_last = layer_info.get('is_last', False)
    
    # Layer dimensions
    in_h = layer_info.get('in_h', 0)
    in_w = layer_info.get('in_w', 0)
    out_h = layer_info.get('out_h', 0)
    out_w = layer_info.get('out_w', 0)
    stride = layer_info.get('stride', 1)
    padding = layer_info.get('padding', 0)
    
    # Weight is already quantized to PoT values in collect_pot_layer_info()
    # DO NOT call quantize_to_pot again! Just convert to numpy.
    with torch.no_grad():
        w_q = weight.numpy() if isinstance(weight, torch.Tensor) else weight
    
    # Get act_scale for bias scaling
    act_scale = layer_info.get('act_scale')
    if act_scale is None:
        act_scale = 1.0
    
    # DEBUG: Print weight statistics
    unique_vals = sorted(set(w_q.flatten().astype(int)))
    print(f"  [OPTIMIZED UNROLL] {name}: first_layer={is_first_layer}, unique weights={unique_vals}")
    
    code = f"// {name} - Unrolled with Zero-Padding (11 levels)\n"
    
    if 'Linear' in layer_type:
        # Linear layers: no spatial padding needed, use original
        code += _generate_linear_unrolled(name, w_q, bias, use_relu, act_scale)
    elif 'Conv1d' in layer_type:
        # Conv1d with Zero-Padding
        in_L = layer_info.get('in_L', 0)
        out_L = layer_info.get('out_L', 0)
        code += _generate_conv1d_unrolled_optimized(
            name, w_q, bias, use_relu,
            in_L, out_L, stride, padding, act_scale,
            is_first_layer=is_first_layer
        )
    elif 'Conv2d' in layer_type and 'Depthwise' not in layer_type:
        # Standard Conv2d with Zero-Padding
        code += _generate_conv2d_unrolled_optimized(
            name, w_q, bias, use_relu,
            in_h, in_w, out_h, out_w, stride, padding, act_scale,
            is_first_layer=is_first_layer
        )
    else:
        # DepthwiseConv2d: use baseline's dedicated depthwise generator
        if 'Depthwise' in layer_type:
            print(f"  [INFO] {name}: {layer_type} using baseline depthwise generator")
            from ..export import generate_depthwise_conv_layer
            return generate_depthwise_conv_layer(layer_info, is_first_layer)
        else:
            # Other types: fall back to original unroll
            print(f"  [INFO] {name}: {layer_type} falling back to original unroll")
            return generate_unrolled_layer(layer_info)
    
    return code


def _generate_conv2d_unrolled_optimized(
    name: str, w_q: np.ndarray, bias, use_relu: bool,
    in_h: int, in_w: int, out_h: int, out_w: int,
    stride: int, padding: int, act_scale: float,
    is_first_layer: bool = False
) -> str:
    """Generate Conv2d unrolled code with Zero-Padding.
    
    Zero-Padding eliminates all boundary if statements.
    
    Args:
        is_first_layer: If True, input type is uint8_t (image input)
    """
    out_ch, in_ch, kh, kw = w_q.shape
    
    # Calculate padded dimensions
    padded_h = in_h + 2 * padding
    padded_w = in_w + 2 * padding
    padded_size = in_ch * padded_h * padded_w
    
    # Input type depends on whether this is the first layer
    input_type = "uint8_t" if is_first_layer else "int8_t"
    
    code = f"// Conv2d: {in_ch}x{in_h}x{in_w} -> {out_ch}x{out_h}x{out_w}\n"
    code += f"// Kernel: {kh}x{kw}, Stride: {stride}, Padding: {padding}\n"
    code += f"// Optimized: Zero-Padding (no boundary checks)\n\n"
    
    code += f"static void {name}_forward(const {input_type}* input, int8_t* output) {{\n"
    
    # Zero-padding buffer
    # First layer: use int16_t to preserve uint8 values (0-255) without overflow
    # Other layers: use int8_t since values are already in -128~127 range
    padded_type = "int16_t" if is_first_layer else "int8_t"
    padded_elem_size = 2 if is_first_layer else 1
    
    code += f"    // Zero-padding: {in_ch}x{in_h}x{in_w} -> {in_ch}x{padded_h}x{padded_w}\n"
    code += f"    static {padded_type} padded[{padded_size}];\n"
    code += f"    memset(padded, 0, {padded_size * padded_elem_size});\n\n"
    
    # Copy input to padded buffer (no casting needed now)
    code += f"    for (int c = 0; c < {in_ch}; c++)\n"
    code += f"        for (int y = 0; y < {in_h}; y++)\n"
    code += f"            for (int x = 0; x < {in_w}; x++)\n"
    code += f"                padded[c * {padded_h * padded_w} + (y + {padding}) * {padded_w} + (x + {padding})] = input[c * {in_h * in_w} + y * {in_w} + x];\n\n"
    
    code += f"    int32_t acc;\n\n"
    
    # C loops for output positions
    code += f"    for (int oy = 0; oy < {out_h}; oy++) {{\n"
    code += f"        for (int ox = 0; ox < {out_w}; ox++) {{\n"
    
    # Unroll each output channel
    for oc in range(out_ch):
        code += f"            // Output channel {oc}\n"
        code += f"            acc = 0;\n"
        
        # Unroll kernel operations - NO boundary checks needed!
        for ic in range(in_ch):
            for ky in range(kh):
                for kx in range(kw):
                    w = int(w_q[oc, ic, ky, kx])
                    if w == 0:
                        continue
                    
                    # Direct access to padded buffer
                    # padded index = ic * padded_h * padded_w + (oy * stride + ky) * padded_w + (ox * stride + kx)
                    ic_base = ic * padded_h * padded_w
                    
                    if stride == 1:
                        y_expr = f"oy + {ky}" if ky > 0 else "oy"
                        x_expr = f"ox + {kx}" if kx > 0 else "ox"
                    else:
                        y_expr = f"oy * {stride} + {ky}" if ky > 0 else f"oy * {stride}"
                        x_expr = f"ox * {stride} + {kx}" if kx > 0 else f"ox * {stride}"
                    
                    idx_expr = f"{ic_base} + ({y_expr}) * {padded_w} + ({x_expr})"
                    
                    # Generate operation without boundary check
                    op = _pot_operation_padded(idx_expr, w)
                    code += f"            {op}\n"
        
        # Apply scale
        code += f"            acc = scale_{name}(acc);\n"
        
        # Add bias
        if bias is not None:
            bias_val = int(bias[oc].item() * act_scale + 0.5)
            if bias_val != 0:
                code += f"            acc += {bias_val};\n"
        
        # Apply ReLU
        if use_relu:
            code += f"            if (acc < 0) acc = 0;\n"
        
        # Store output
        out_base = oc * out_h * out_w
        code += f"            output[{out_base} + oy * {out_w} + ox] = (int8_t)(acc > 127 ? 127 : (acc < -128 ? -128 : acc));\n"
        code += f"\n"
    
    code += f"        }}\n"
    code += f"    }}\n"
    code += "}\n\n"
    
    return code


def _pot_operation_padded(idx_expr: str, w: int) -> str:
    """Generate a PoT operation for padded buffer access."""
    shift = _get_shift(abs(w))
    sign = "+" if w > 0 else "-"
    
    if shift == 0:
        return f"acc {sign}= (int32_t)padded[{idx_expr}];"
    else:
        return f"acc {sign}= (int32_t)padded[{idx_expr}] << {shift};"


def _generate_conv1d_unrolled_optimized(
    name: str, w_q: np.ndarray, bias, use_relu: bool,
    in_L: int, out_L: int, stride: int, padding: int, act_scale: float,
    is_first_layer: bool = False
) -> str:
    """Generate Conv1d unrolled code with Zero-Padding.
    
    Zero-Padding eliminates all boundary if statements.
    
    Args:
        is_first_layer: If True, input type is uint8_t
    """
    out_ch, in_ch, kL = w_q.shape
    
    # Calculate padded dimensions
    padded_L = in_L + 2 * padding
    padded_size = in_ch * padded_L
    
    # Input type depends on whether this is the first layer
    input_type = "uint8_t" if is_first_layer else "int8_t"
    
    code = f"// Conv1d: {in_ch}x{in_L} -> {out_ch}x{out_L}\n"
    code += f"// Kernel: {kL}, Stride: {stride}, Padding: {padding}\n"
    code += f"// Optimized: Zero-Padding (no boundary checks)\n\n"
    
    code += f"static void {name}_forward(const {input_type}* input, int8_t* output) {{\n"
    
    # Zero-padding buffer
    padded_type = "int16_t" if is_first_layer else "int8_t"
    padded_elem_size = 2 if is_first_layer else 1
    
    code += f"    // Zero-padding: {in_ch}x{in_L} -> {in_ch}x{padded_L}\n"
    code += f"    static {padded_type} padded[{padded_size}];\n"
    code += f"    memset(padded, 0, {padded_size * padded_elem_size});\n\n"
    
    # Copy input to padded buffer
    code += f"    for (int c = 0; c < {in_ch}; c++)\n"
    code += f"        for (int i = 0; i < {in_L}; i++)\n"
    code += f"            padded[c * {padded_L} + (i + {padding})] = input[c * {in_L} + i];\n\n"
    
    code += f"    int32_t acc;\n\n"
    
    # Loop for output positions
    code += f"    for (int o = 0; o < {out_L}; o++) {{\n"
    
    # Unroll each output channel
    for oc in range(out_ch):
        code += f"        // Output channel {oc}\n"
        code += f"        acc = 0;\n"
        
        # Unroll kernel operations - NO boundary checks needed!
        for ic in range(in_ch):
            for k in range(kL):
                w = int(w_q[oc, ic, k])
                if w == 0:
                    continue
                
                # Direct access to padded buffer
                ic_base = ic * padded_L
                
                if stride == 1:
                    idx_expr = f"{ic_base} + o + {k}" if k > 0 else f"{ic_base} + o"
                else:
                    idx_expr = f"{ic_base} + o * {stride} + {k}" if k > 0 else f"{ic_base} + o * {stride}"
                
                # Generate operation without boundary check
                op = _pot_operation_padded(idx_expr, w)
                code += f"        {op}\n"
        
        # Apply scale
        code += f"        acc = scale_{name}(acc);\n"
        
        # Add bias
        if bias is not None:
            bias_val = int(bias[oc].item() * act_scale + 0.5)
            if bias_val != 0:
                code += f"        acc += {bias_val};\n"
        
        # Apply ReLU
        if use_relu:
            code += f"        if (acc < 0) acc = 0;\n"
        
        # Store output
        out_base = oc * out_L
        code += f"        output[{out_base} + o] = (int8_t)(acc > 127 ? 127 : (acc < -128 ? -128 : acc));\n"
        code += f"\n"
    
    code += f"    }}\n"
    code += "}\n\n"
    
    return code
