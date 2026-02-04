"""Generate 5-level encoded C code for PoT layers.

5-level encoding: [skip(2)][sign(1)][mag(1)] = 4bit
- 5 levels: -8, -1, 0, +1, +8
- Skip field handles consecutive zeros (0-3 positions)
- Decoding: skip positions, then val = (mag ? 8 : 1) * (sign ? -1 : 1)
- Balanced accuracy and memory efficiency
"""

import numpy as np
from typing import Dict, Any, Tuple, List


def pack_weights_5level(w_q: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Pack quantized weights to 5-level format with skip encoding.
    
    Args:
        w_q: Quantized weights (values in -8, -1, 0, +1, +8)
        
    Returns:
        packed: uint8 array (2 codes per byte)
        n_codes: number of 4-bit codes
        n_weights: original number of weights
    """
    flat = np.round(w_q.flatten()).astype(np.int16)
    n = len(flat)
    
    # Generate 4-bit codes with skip encoding
    codes: List[int] = []
    i = 0
    while i < n:
        # Count consecutive zeros (max 3)
        skip = 0
        while i + skip < n and flat[i + skip] == 0 and skip < 3:
            skip += 1
        
        i += skip
        
        if i >= n:
            # End of weights - emit dummy code if needed
            if skip > 0:
                codes.append((skip << 2) | 0b00)  # skip + val=+1
            break
        
        w = flat[i]
        sign = 1 if w < 0 else 0
        mag = 1 if abs(w) == 8 else 0  # 0=1, 1=8
        
        code = (skip << 2) | (sign << 1) | mag
        codes.append(code)
        i += 1
    
    # Pack 2 codes per byte
    n_codes = len(codes)
    packed_len = (n_codes + 1) // 2
    packed = np.zeros(packed_len, dtype=np.uint8)
    for i in range(0, n_codes, 2):
        high = codes[i]
        low = codes[i + 1] if i + 1 < n_codes else 0
        packed[i // 2] = (high << 4) | low
    
    return packed, n_codes, n


def pack_weights_5level_linear(w_q: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Pack Linear weights row by row (each output filter separately).
    
    Args:
        w_q: 2D weight array (out_features, in_features)
        
    Returns:
        packed: uint8 array (all rows concatenated, byte-aligned per row)
        row_bytes: list of byte counts per row
    
    Note:
        5-level encoding can only skip up to 3 consecutive zeros.
        If there are 4+ consecutive zeros, the 4th zero onwards are replaced
        with +1 (smallest non-zero value). This is a spec limitation.
    """
    out_features, in_features = w_q.shape
    all_packed = []
    row_bytes = []
    
    for o in range(out_features):
        row = np.round(w_q[o]).astype(np.int16).copy()
        
        # WORKAROUND: Replace 4th+ consecutive zeros with +1
        # 5-level skip field is 2 bits (max 3), so 4+ zeros can't be encoded
        zero_run = 0
        for j in range(len(row)):
            if row[j] == 0:
                zero_run += 1
                if zero_run > 3:
                    row[j] = 1  # Replace with smallest non-zero
                    zero_run = 0  # Reset counter
            else:
                zero_run = 0
        
        # Generate codes for this row
        codes: List[int] = []
        i = 0
        while i < in_features:
            skip = 0
            while i + skip < in_features and row[i + skip] == 0 and skip < 3:
                skip += 1
            
            i += skip
            
            if i >= in_features:
                # Trailing zeros: emit skip code so decoder advances i
                if skip > 0:
                    codes.append((skip << 2) | 0b00)  # dummy: will be skipped by decoder
                break
            
            w = row[i]
            sign = 1 if w < 0 else 0
            mag = 1 if abs(w) == 8 else 0
            
            code = (skip << 2) | (sign << 1) | mag
            codes.append(code)
            i += 1
        
        # Pack to bytes (byte-aligned per row)
        n_codes = len(codes)
        packed_len = (n_codes + 1) // 2
        for j in range(0, n_codes, 2):
            high = codes[j]
            low = codes[j + 1] if j + 1 < n_codes else 0
            all_packed.append((high << 4) | low)
        
        row_bytes.append(packed_len)
    
    return np.array(all_packed, dtype=np.uint8), row_bytes


def generate_5level_layer(layer_info: Dict[str, Any]) -> str:
    """Generate 5-level encoded C code for a layer.
    
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
    
    if 'Linear' in layer_type:
        # Linear: row-aligned packing
        packed, row_bytes = pack_weights_5level_linear(w_q)
        
        code = f"// {name} - 5-level encoding (-8, -1, 0, +1, +8)\n"
        code += f"// Packed: {len(packed)} bytes (row-aligned)\n\n"
        
        # Weight data
        code += f"static const uint8_t {name}_weights[] = {{\n    "
        for i, b in enumerate(packed):
            code += f"0x{b:02x}, "
            if (i + 1) % 16 == 0:
                code += "\n    "
        code += "\n};\n\n"
        
        # Bias data
        if bias is not None:
            code += f"static const int32_t {name}_bias[] = {{\n    "
            for i, b in enumerate(bias):
                bias_val = int(round(b.item() * act_scale))
                # No clipping for int32
                code += f"{bias_val}, "
            code += "\n};\n\n"
        
        code += _generate_linear_5level(name, w_q.shape, bias, use_relu, act_scale)
        return code
    
    elif 'Conv2d' in layer_type:
        # Conv2d: flat packing (pre-decode at runtime)
        packed, n_codes, n_weights = pack_weights_5level(w_q)
        
        code = f"// {name} - 5-level encoding (-8, -1, 0, +1, +8)\n"
        code += f"// Packed: {len(packed)} bytes ({n_codes} codes for {n_weights} weights)\n\n"
        
        # Weight data
        code += f"static const uint8_t {name}_weights[] = {{\n    "
        for i, b in enumerate(packed):
            code += f"0x{b:02x}, "
            if (i + 1) % 16 == 0:
                code += "\n    "
        code += "\n};\n\n"
        
        # Bias data
        if bias is not None:
            code += f"static const int32_t {name}_bias[] = {{\n    "
            for i, b in enumerate(bias):
                bias_val = int(round(b.item() * act_scale))
                # No clipping for int32
                code += f"{bias_val}, "
            code += "\n};\n\n"
        
        code += _generate_conv2d_5level(name, layer_info, bias, use_relu, act_scale)
        return code
    
    return ""


def _generate_linear_5level(name: str, shape: tuple, bias, use_relu: bool, act_scale: float) -> str:
    """Generate 5-level Linear layer code."""
    out_features, in_features = shape
    
    code = f"static void {name}_forward(const int8_t* input, int8_t* output) {{\n"
    code += f"    const uint8_t* wp = {name}_weights;\n"
    code += f"    int32_t acc;\n"
    code += f"    uint8_t packed, code, skip, sign, mag;\n"
    code += f"    int shift, shifted, mask;\n"
    code += f"    int i, nibble;\n\n"
    
    code += f"    for (int o = 0; o < {out_features}; o++) {{\n"
    code += f"        acc = 0;\n"
    code += f"        i = 0;\n"
    code += f"        nibble = 0;  // reset per row (row-aligned packing)\n"
    code += f"        while (i < {in_features}) {{\n"
    code += f"            if (nibble == 0) {{\n"
    code += f"                packed = *wp++;\n"
    code += f"                code = packed >> 4;\n"
    code += f"            }} else {{\n"
    code += f"                code = packed & 0xf;\n"
    code += f"            }}\n"
    code += f"            nibble = 1 - nibble;\n"
    code += f"            \n"
    code += f"            skip = (code >> 2) & 0x3;\n"
    code += f"            i += skip;  // skip zeros\n"
    code += f"            if (i >= {in_features}) break;\n"
    code += f"            \n"
    code += f"            sign = (code >> 1) & 1;\n"
    code += f"            mag = code & 1;\n"
    code += f"            shift = (mag << 1) + mag;  // 0 or 3\n"
    code += f"            shifted = (int)input[i] << shift;\n"
    code += f"            mask = -(int)sign;\n"
    code += f"            acc += (shifted ^ mask) - mask;\n"
    code += f"            i++;\n"
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


def _generate_conv2d_5level(name: str, layer_info: Dict, bias, use_relu: bool, act_scale: float) -> str:
    """Generate 5-level Conv2d layer code."""
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
    
    code = f"static void {name}_forward(const int8_t* input, int8_t* output) {{\n"
    code += f"    // Conv2d: {in_ch}x{in_h}x{in_w} -> {out_ch}x{out_h}x{out_w}\n"
    code += f"    // 5-level with skip encoding\n"
    code += f"    int32_t acc;\n"
    code += f"    uint8_t packed, code, skip, sign, mag;\n"
    code += f"    int base, mask, val;\n\n"
    
    # For conv2d, we need to decode per-filter
    # Pre-decode weights to array for random access during convolution
    code += f"    // Pre-decoded weights for random access\n"
    code += f"    static int8_t w_decoded[{out_ch * kernel_size}];\n"
    code += f"    static int decoded = 0;\n"
    code += f"    if (!decoded) {{\n"
    code += f"        const uint8_t* wp = {name}_weights;\n"
    code += f"        int idx = 0, nibble = 0;\n"
    code += f"        uint8_t p;\n"
    code += f"        while (idx < {out_ch * kernel_size}) {{\n"
    code += f"            if (nibble == 0) {{ p = *wp++; code = p >> 4; }}\n"
    code += f"            else {{ code = p & 0xf; }}\n"
    code += f"            nibble = 1 - nibble;\n"
    code += f"            skip = (code >> 2) & 0x3;\n"
    code += f"            for (int s = 0; s < skip && idx < {out_ch * kernel_size}; s++)\n"
    code += f"                w_decoded[idx++] = 0;\n"
    code += f"            if (idx >= {out_ch * kernel_size}) break;\n"
    code += f"            sign = (code >> 1) & 1;\n"
    code += f"            mag = code & 1;\n"
    code += f"            base = 1 << ((mag << 1) + mag);  // 1 or 8\n"
    code += f"            mask = -(int)sign;\n"
    code += f"            w_decoded[idx++] = (base ^ mask) - mask;\n"
    code += f"        }}\n"
    code += f"        decoded = 1;\n"
    code += f"    }}\n\n"
    
    code += f"    for (int oc = 0; oc < {out_ch}; oc++) {{\n"
    code += f"        const int8_t* wf = w_decoded + oc * {kernel_size};\n"
    code += f"        for (int oy = 0; oy < {out_h}; oy++) {{\n"
    code += f"            for (int ox = 0; ox < {out_w}; ox++) {{\n"
    code += f"                acc = 0;\n"
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
    code += f"                        if (iy < 0 || iy >= {in_h}) {{ w_idx += {kw}; continue; }}\n"
    code += f"                        for (int kx = 0; kx < {kw}; kx++) {{\n"
    code += f"                            int ix = ox * {stride_w} + kx - {pad_w};\n"
    code += f"                            if (ix >= 0 && ix < {in_w}) {{\n"
    code += f"                                val = wf[w_idx];\n"
    code += f"                                if (val) {{\n"
    
    if group_stride_str == "0":
        input_idx = f"ic * {in_h * in_w} + iy * {in_w} + ix"
    else:
        input_idx = f"({group_stride_str} + ic) * {in_h * in_w} + iy * {in_w} + ix"

    code += f"                                    int32_t inp = input[{input_idx}];\n"
    code += f"                                    // 5-level: val is -8, -1, +1, or +8\n"
    code += f"                                    if (val == 1) acc += inp;\n"
    code += f"                                    else if (val == -1) acc -= inp;\n"
    code += f"                                    else if (val == 8) acc += inp << 3;\n"
    code += f"                                    else acc -= inp << 3;  // val == -8\n"
    code += f"                                }}\n"
    code += f"                            }}\n"
    code += f"                            w_idx++;\n"
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
