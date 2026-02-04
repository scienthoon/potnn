"""Generate Ternary RLE encoded C code for PoT layers.

Ternary encoding: Run-Length Encoded
- 3 levels: -1, 0, +1
- RLE format: [type(2)][count/value]
  - 00: +1 single
  - 01: -1 single
  - 10: zero run (next 6 bits = count 1-64)
  - 11: non-zero run (next 2 bits = count 1-4, then values)
- Highest compression ratio for sparse ternary weights
"""

import numpy as np
from typing import Dict, Any, Tuple, List


def pack_weights_ternary(w_q: np.ndarray) -> Tuple[bytes, int]:
    """Pack quantized weights to Ternary RLE format.
    
    Args:
        w_q: Quantized weights (values in -1, 0, +1)
        
    Returns:
        packed: bytes of RLE encoded data
        n_weights: original number of weights
    """
    flat = np.round(w_q.flatten()).astype(np.int8)
    n = len(flat)
    
    # Simple RLE: encode runs of zeros and individual non-zeros
    # Format: [code(2 bits)][payload]
    # 00 = +1, 01 = -1, 10 = zero run (6 bits count), 11 = reserved
    
    bits: List[int] = []  # List of bits
    
    i = 0
    while i < n:
        if flat[i] == 0:
            # Count consecutive zeros
            count = 0
            while i + count < n and flat[i + count] == 0 and count < 63:
                count += 1
            
            # Encode zero run: 10 + 6-bit count (1-64 encoded as 0-63)
            bits.extend([1, 0])  # type = 10
            for b in range(5, -1, -1):
                bits.append((count - 1) >> b & 1)
            i += count
        else:
            # Single non-zero value
            if flat[i] == 1:
                bits.extend([0, 0])  # +1
            else:  # -1
                bits.extend([0, 1])  # -1
            i += 1
    
    # Pack bits into bytes
    packed = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte |= bits[i + j] << (7 - j)
        packed.append(byte)
    
    return bytes(packed), n


def pack_weights_triple_run(w_q: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Pack weights using 'Triple-Run' custom ternary encoding into uint32.
    
    Encoding:
    - 00: 0
    - 01: +1
    - 10: -1
    - 11: Repeat Previous x2 (Total 3 of same value)
    
    CRITICAL: Triple Runs MUST NOT cross block boundaries (rows for Linear, filters for Conv),
    because the C decoder processes blocks independently and resets state.
      
    Args:
        w_q: Quantized weights
        
    Returns:
        packed: uint32 array
        n_codes: number of 2-bit codes
        n: number of weights
    """
    shape = w_q.shape
    flat = np.round(w_q.flatten()).astype(np.int8)
    n = len(flat)
    
    # Determine Block Size
    if len(shape) == 2:
        # Linear: [Out, In] -> Block size is In
        block_size = shape[1]
    elif len(shape) == 4:
        # Conv2d: [Out, In, KH, KW] -> Block size is In*KH*KW
        block_size = shape[1] * shape[2] * shape[3]
    else:
        # Fallback or 1D
        block_size = n
        
    codes = []
    i = 0
    while i < n:
        val = flat[i]
        
        # Check boundary integrity
        # We can only do a Triple Run if i, i+1, i+2 are in the SAME block
        current_block = i // block_size
        next_block = (i + 2) // block_size
        
        can_run = (i + 2 < n) and \
                  (flat[i+1] == val and flat[i+2] == val) and \
                  (current_block == next_block)
        
        if can_run:
            # Emit code for val first
            if val == 0: codes.append(0b00)
            elif val == 1: codes.append(0b01)
            else: codes.append(0b10) # -1
            
            # Emit repeat code
            codes.append(0b11) # Repeat x2
            i += 3
        else:
            # Single emit
            if val == 0: codes.append(0b00)
            elif val == 1: codes.append(0b01)
            else: codes.append(0b10) # -1
            i += 1
            
    # Pack 16 codes per uint32
    n_codes = len(codes)
    packed_len = (n_codes + 15) // 16
    packed = np.zeros(packed_len, dtype=np.uint32)
    
    for i in range(0, n_codes, 16):
        b = 0
        for j in range(16):
            if i + j < n_codes:
                b |= (int(codes[i + j]) << (2 * j))
        packed[i // 16] = b
        
    return packed, n_codes, n


def generate_ternary_layer(layer_info: Dict[str, Any]) -> str:
    """Generate Triple-Run Ternary encoded C code."""
    name = layer_info['name']
    layer_type = layer_info['type']
    weight = layer_info['weight']
    bias = layer_info.get('bias', None)
    use_relu = layer_info.get('has_relu', False)
    act_scale = layer_info.get('act_scale', 1.0) or 1.0
    
    if hasattr(weight, 'numpy'):
        w_q = weight.numpy()
    else:
        w_q = np.array(weight)
    
    packed, n_codes, n_weights = pack_weights_triple_run(w_q)
    
    code = f"// {name} - Ternary Triple-Run (00:0, 01:+1, 10:-1, 11:Rep2)\n"
    code += f"// Packed: {len(packed)*4} bytes ({n_codes} codes for {n_weights} weights)\n\n"
    
    code += f"static const uint32_t {name}_weights[] = {{\n    "
    for i, w in enumerate(packed):
        code += f"0x{w:08x}, "
        if (i + 1) % 8 == 0:
            code += "\n    "
    code += "\n};\n\n"
    
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
        code += _generate_linear_ternary(name, w_q.shape, bias, use_relu, act_scale)
    elif 'Conv2d' in layer_type:
        code += _generate_conv2d_ternary(name, layer_info, bias, use_relu, act_scale)
    
    return code


def _generate_linear_ternary(name: str, shape: tuple, bias, use_relu: bool, act_scale: float) -> str:
    """Stream decoding for Linear layer."""
    out_features, in_features = shape
    
    code = f"static void {name}_forward(const int8_t* input, int8_t* output) {{\n"
    code += f"    const uint32_t* wp = {name}_weights;\n"
    code += f"    int32_t acc;\n"
    code += f"    uint32_t weight_chunk = *wp++;\n"
    code += f"    uint8_t code;\n"
    code += f"    int8_t prev_val = 0;\n"
    code += f"    int code_idx = 0;\n\n"
    
    code += f"    for (int o = 0; o < {out_features}; o++) {{\n"
    code += f"        acc = 0;\n"
    code += f"        int i = 0;\n"
    code += f"        while (i < {in_features}) {{\n"
    code += f"            code = (weight_chunk >> (code_idx << 1)) & 0x3;\n"
    code += f"            code_idx++;\n"
    code += f"            if (code_idx == 16) {{\n"
    code += f"                code_idx = 0;\n"
    code += f"                weight_chunk = *wp++;\n"
    code += f"            }}\n"
    code += f"            \n"
    code += f"            if (code == 3) {{ // Repeat x2 (Total 3)\n"
    code += f"                // prev_val applied 2 more times\n"
    code += f"                for (int k=0; k<2 && i < {in_features}; k++) {{\n"
    code += f"                    acc += (int32_t)input[i++] * prev_val;\n"
    code += f"                }}\n"
    code += f"            }} else {{\n"
    code += f"                // Decode new value\n"
    code += f"                if (code == 0) prev_val = 0;\n"
    code += f"                else if (code == 1) {{ prev_val = 1; acc += input[i]; }}\n"
    code += f"                else {{ prev_val = -1; acc -= input[i]; }}\n"
    code += f"                i++;\n"
    code += f"            }}\n"
    code += f"        }}\n"
    
    code += f"        acc = scale_{name}(acc);\n"
    if bias is not None: code += f"        acc += {name}_bias[o];\n"
    if use_relu: code += f"        if (acc < 0) acc = 0;\n"
    code += f"        output[o] = (int8_t)(acc > 127 ? 127 : (acc < -128 ? -128 : acc));\n"
    code += f"    }}\n"
    code += f"}}\n\n"
    
    return code


def _generate_conv2d_ternary(name: str, layer_info: Dict, bias, use_relu: bool, act_scale: float) -> str:
    """Block decoding for Conv2d layer."""
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
    code += f"    const uint32_t* wp = {name}_weights;\n"
    code += f"    int32_t acc;\n"
    code += f"    int8_t w_buf[{kernel_size}]; // Filter Weights Cache\n"
    code += f"    uint32_t weight_chunk;\n"
    code += f"    uint8_t code;\n"
    code += f"    int8_t prev_val;\n"
    code += f"    int code_idx;\n\n"
    
    code += f"    // wp is persistent across filters since we decode sequentially\n"
    code += f"    weight_chunk = *wp++;\n"
    code += f"    code_idx = 0;\n"
    code += f"    \n"
    
    code += f"    for (int oc = 0; oc < {out_ch}; oc++) {{\n"
    code += f"        // 1. Decode Filter Weights to Stack Buffer\n"
    code += f"        int w_ptr = 0;\n"
    code += f"        prev_val = 0;\n"
    code += f"        while (w_ptr < {kernel_size}) {{\n"
    code += f"            code = (weight_chunk >> (code_idx << 1)) & 0x3;\n"
    code += f"            code_idx++;\n"
    code += f"            if (code_idx == 16) {{\n"
    code += f"                code_idx = 0;\n"
    code += f"                weight_chunk = *wp++;\n"
    code += f"            }}\n"
    code += f"            \n"
    code += f"            if (code == 3) {{ // Repeat x2\n"
    code += f"                w_buf[w_ptr++] = prev_val;\n"
    code += f"                if (w_ptr < {kernel_size}) w_buf[w_ptr++] = prev_val;\n"
    code += f"            }} else {{\n"
    code += f"                if (code == 0) prev_val = 0;\n"
    code += f"                else if (code == 1) prev_val = 1;\n"
    code += f"                else prev_val = -1;\n"
    code += f"                w_buf[w_ptr++] = prev_val;\n"
    code += f"            }}\n"
    code += f"        }}\n"
    code += f"        \n"
    code += f"        // 2. Compute Convolution using Cache\n"
    code += f"        for (int oy = 0; oy < {out_h}; oy++) {{\n"
    code += f"            for (int ox = 0; ox < {out_w}; ox++) {{\n"
    code += f"                acc = 0;\n"
    code += f"                int buf_idx = 0;\n"
    
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
    code += f"                        if (iy < 0 || iy >= {in_h}) {{ buf_idx += {kw}; continue; }}\n"
    code += f"                        for (int kx = 0; kx < {kw}; kx++) {{\n"
    code += f"                            int ix = ox * {stride_w} + kx - {pad_w};\n"
    code += f"                            if (ix >= 0 && ix < {in_w}) {{\n"
    
    if group_stride_str == "0":
        input_idx = f"ic * {in_h * in_w} + iy * {in_w} + ix"
    else:
        input_idx = f"({group_stride_str} + ic) * {in_h * in_w} + iy * {in_w} + ix"
        
    code += f"                                int8_t w = w_buf[buf_idx];\n"
    code += f"                                if (w) acc += (w == 1) ? input[{input_idx}] : -input[{input_idx}];\n"
    code += f"                            }}\n"
    code += f"                            buf_idx++;\n"
    code += f"                        }}\n"
    code += f"                    }}\n"
    code += f"                }}\n"
    code += f"                \n"
    code += f"                acc = scale_{name}(acc);\n"
    if bias is not None: code += f"                acc += {name}_bias[oc];\n"
    if use_relu: code += f"                if (acc < 0) acc = 0;\n"
    code += f"                output[oc * {out_h * out_w} + oy * {out_w} + ox] = (int8_t)(acc > 127 ? 127 : (acc < -128 ? -128 : acc));\n"
    code += f"            }}\n"
    code += f"        }}\n"
    code += f"    }}\n"
    code += f"}}\n\n"
    
    return code
