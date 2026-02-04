"""Generate 2-bit encoded C code for PoT layers.

2-bit encoding: [sign(1)][shift(1)] = 2bit
- 4 levels: ±1, ±2
- No zero (DenseShift style)
- Decoding: val = (shift ? 2 : 1) * (sign ? -1 : 1)
- Minimal memory footprint
"""

import numpy as np
from typing import Dict, Any, Tuple


def pack_weights_2bit(w_q: np.ndarray) -> Tuple[np.ndarray, int]:
    """Pack quantized weights to 2-bit format.
    
    Args:
        w_q: Quantized weights (values in ±1, ±2)
        
    Returns:
        packed: uint8 array (4 weights per byte)
        original_size: number of weights
    """
    # w_q shape: [out_ch, in_ch, kh, kw] or [out_features, in_features] or 3D
    if len(w_q.shape) == 4:
        out_ch, in_ch, kh, kw = w_q.shape
        kernel_size = in_ch * kh * kw
    elif len(w_q.shape) == 3:
        # Conv1d
        out_ch, in_ch, kw = w_q.shape
        kernel_size = in_ch * kw
    else:
        out_ch, in_features = w_q.shape
        kernel_size = in_features
        
    # Words per filter (16 weights per uint32)
    words_per_filter = (kernel_size + 15) // 16
    packed = np.zeros(out_ch * words_per_filter, dtype=np.uint32)
    
    for oc in range(out_ch):
        # Get filter weights and flatten
        filter_w = w_q[oc].flatten()
        n = len(filter_w)
        
        # Encode
        encoded = np.zeros(n, dtype=np.uint8)
        for i, w in enumerate(filter_w):
            if w == 0:
                w = 1 if i % 2 == 0 else -1
            sign = 1 if w < 0 else 0
            shift = 1 if abs(w) == 2 else 0
            encoded[i] = (sign << 1) | shift
            
        # Pack to uint32
        for i in range(0, n, 16):
            chunk = 0
            for j in range(16):
                if i + j < n:
                    chunk |= (int(encoded[i + j]) << (2 * j))
            packed[oc * words_per_filter + (i // 16)] = chunk
            
    return packed, out_ch * kernel_size


def generate_2bit_layer(layer_info: Dict[str, Any]) -> str:
    """Generate 2-bit encoded C code for a layer.
    
    Args:
        layer_info: Dictionary with layer information
        
    Returns:
        C code for the layer
    """
    name = layer_info['name']
    layer_type = layer_info['type']
    weight = layer_info['weight']
    bias = layer_info.get('bias', None)
    use_relu = layer_info.get('has_relu', False)
    act_scale = layer_info.get('act_scale', 1.0) or 1.0
    
    # Get weight as numpy
    if hasattr(weight, 'numpy'):
        w_q = weight.numpy()
    else:
        w_q = np.array(weight)
    
    # Pack weights
    packed, n_weights = pack_weights_2bit(w_q)
    
    code = f"// {name} - 2-bit encoding (4 levels: ±1, ±2)\n"
    code += f"// Packed weights: {len(packed)*4} bytes ({n_weights} weights, packed as uint32)\n\n"
    
    # Weight data
    code += f"static const uint32_t {name}_weights[] = {{\n    "
    for i, w in enumerate(packed):
        code += f"0x{w:08x}, "
        if (i + 1) % 8 == 0:
            code += "\n    "
    code += "\n};\n\n"
    
    # Bias data (scaled by act_scale)
    if bias is not None:
        code += f"static const int32_t {name}_bias[] = {{\n    "
        for i, b in enumerate(bias):
            bias_val = int(round(b.item() * act_scale))
            # No clipping for int32
            code += f"{bias_val}, "
            if (i + 1) % 16 == 0:
                code += "\n    "
        code += "\n};\n\n"
    
    if 'Linear' in layer_type:
        code += _generate_linear_2bit(name, w_q.shape, bias, use_relu, act_scale)
    elif 'Conv2d' in layer_type:
        code += _generate_conv2d_2bit(name, layer_info, bias, use_relu, act_scale)
    
    return code


def _generate_linear_2bit(name: str, shape: tuple, bias, use_relu: bool, act_scale: float) -> str:
    """Generate 2-bit Linear layer code."""
    out_features, in_features = shape
    
    code = f"static void {name}_forward(const int8_t* input, int8_t* output) {{\n"
    code += f"    const uint32_t* wp = {name}_weights;\n"
    code += f"    int32_t acc, shifted;\n"
    code += f"    uint32_t weight_chunk;\n"
    code += f"    uint8_t code, shift;\n\n"
    
    code += f"    for (int o = 0; o < {out_features}; o++) {{\n"
    code += f"        acc = 0;\n"
    code += f"        for (int i = 0; i < {in_features}; i += 16) {{\n"
    code += f"            weight_chunk = *wp++;\n"
    code += f"            \n"
    code += f"            // Process 16 weights from chunk (LSB first)\n"
    code += f"            for (int k = 0; k < 16 && (i + k) < {in_features}; k++) {{\n"
    code += f"                // 2 bits: [sign(1)][shift(1)]\n"
    code += f"                code = (weight_chunk >> (k << 1)) & 0x3;\n"
    code += f"                shift = code & 1;  // 0 or 1\n"
    code += f"                shifted = (int32_t)input[i + k] << shift;\n"
    code += f"                acc += (code & 2) ? -shifted : shifted;\n"
    code += f"            }}\n"
    code += f"        }}\n"
    
    # Scale
    code += f"        acc = scale_{name}(acc);\n"
    
    # Bias
    if bias is not None:
        code += f"        acc += {name}_bias[o];\n"
    
    # ReLU
    if use_relu:
        code += f"        if (acc < 0) acc = 0;\n"
    
    # Clamp and store
    code += f"        output[o] = (int8_t)(acc > 127 ? 127 : (acc < -128 ? -128 : acc));\n"
    code += f"    }}\n"
    code += f"}}\n\n"
    
    return code


def _generate_conv2d_2bit(name: str, layer_info: Dict, bias, use_relu: bool, act_scale: float) -> str:
    """Generate 2-bit Conv2d layer code."""
    weight = layer_info['weight']
    if hasattr(weight, 'shape'):
        w_shape = weight.shape
    else:
        w_shape = np.array(weight).shape
        
    if len(w_shape) == 4:
        out_ch, in_ch, kh, kw = w_shape
    elif len(w_shape) == 3:
        out_ch, in_ch, kw = w_shape
        kh = 1

    out_h = layer_info.get('out_h', 1)
    out_w = layer_info.get('out_w', 1)
    in_h = layer_info.get('in_h', 1)
    in_w = layer_info.get('in_w', 1)
    stride = layer_info.get('stride', 1)
    padding = layer_info.get('padding', 0)
    groups = layer_info.get('groups', 1)
    
    # Handle tuple parameters
    if isinstance(stride, tuple): stride_h, stride_w = stride
    else: stride_h = stride_w = stride
        
    if isinstance(padding, tuple): pad_h, pad_w = padding
    else: pad_h = pad_w = padding
    
    kernel_size = in_ch * kh * kw
    words_per_filter = (kernel_size + 15) // 16
    
    code = f"static void {name}_forward(const int8_t* input, int8_t* output) {{\n"
    code += f"    // Conv2d: {in_ch}x{in_h}x{in_w} -> {out_ch}x{out_h}x{out_w}\n"
    code += f"    const uint32_t* wp_base;\n"
    code += f"    int32_t acc, shifted;\n"
    code += f"    uint32_t weight_chunk;\n"
    code += f"    uint8_t code_bits, shift;\n\n"
    
    code += f"    for (int oc = 0; oc < {out_ch}; oc++) {{\n"
    code += f"        wp_base = {name}_weights + oc * {words_per_filter};\n"
    code += f"        for (int oy = 0; oy < {out_h}; oy++) {{\n"
    code += f"            for (int ox = 0; ox < {out_w}; ox++) {{\n"
    code += f"                acc = 0;\n"
    code += f"                const uint32_t* wp = wp_base;\n"
    code += f"                weight_chunk = *wp++;\n"
    code += f"                int w_idx = 0;\n"
    
    # Group offset calculation
    channels_per_group = in_ch
    if groups == out_ch:
        group_stride_str = f"oc * {channels_per_group}"
    elif groups > 1:
        out_per_group = out_ch // groups
        group_stride_str = f"(oc / {out_per_group}) * {channels_per_group}"
    else:
        group_stride_str = "0"

    code += f"                for (int ic = 0; ic < {in_ch}; ic++) {{\n"
    code += f"                    for (int ky = 0; ky < {kh}; ky++) {{\n"
    code += f"                        int iy = oy * {stride_h} + ky - {pad_h};\n"
    code += f"                        int skip = (iy < 0 || iy >= {in_h});\n"
    code += f"                        for (int kx = 0; kx < {kw}; kx++) {{\n"
    code += f"                            int ix = ox * {stride_w} + kx - {pad_w};\n"
    code += f"                            \n"
    code += f"                            // Extract 2 bits\n"
    code += f"                            if ((w_idx & 15) == 0 && w_idx > 0) weight_chunk = *wp++;\n"
    code += f"                            code_bits = (weight_chunk >> ((w_idx & 15) << 1)) & 3;\n"
    code += f"                            w_idx++;\n"
    code += f"                            \n"
    code += f"                            if (skip || ix < 0 || ix >= {in_w}) continue;\n"
    code += f"                            \n"
    code += f"                            // Consistent Zero Handling: Map 0 -> +1 (shift 0)\n"
    code += f"                            shift = code_bits & 1;\n"
    
    if group_stride_str == "0":
        input_idx = f"ic * {in_h * in_w} + iy * {in_w} + ix"
    else:
        input_idx = f"({group_stride_str} + ic) * {in_h * in_w} + iy * {in_w} + ix"
        
    code += f"                            shifted = (int32_t)input[{input_idx}] << shift;\n"
    code += f"                            acc += (code_bits & 2) ? -shifted : shifted;\n"
    code += f"                        }}\n"
    code += f"                    }}\n"
    code += f"                }}\n"
    code += f"                acc = scale_{name}(acc);\n"
    
    if bias is not None:
        code += f"                acc += {name}_bias[oc];\n"
    
    if use_relu:
        code += f"                if (acc < 0) acc = 0;\n"
    
    code += f"                output[oc * {out_h * out_w} + oy * {out_w} + ox] = (int8_t)(acc > 127 ? 127 : (acc < -128 ? -128 : acc));\n"
    code += f"            }}\n"
    code += f"        }}\n"
    code += f"    }}\n"
    code += f"}}\n\n"
    
    return code
