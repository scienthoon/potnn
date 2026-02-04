"""Main C header generation for potnn."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple

from ..modules import PoTLinear, PoTConv2d
from .scale import generate_scale_func, calculate_combined_scale
from .unroll import generate_unrolled_layer



def generate_c_header(path: str, model: nn.Module):
    """Generate complete C header file for the model.

    Args:
        path: Output file path
        model: Trained potnn model
    """
    config = model._potnn_config
    allocations = model._potnn_allocations
    input_shape = model._potnn_input_shape

    # Analyze model structure and compute layer dimensions
    layer_dims = _compute_layer_dimensions(model, input_shape)
    
    # Debug: print computed dimensions
    print("\nComputed layer dimensions:")
    for name, dims in layer_dims.items():
        print(f"  {name}: {dims}")
    
    # Collect layer information
    layers = []
    other_layers = []  # MaxPool, Flatten 등
    prev_act_scale = 1.0
    pot_layer_idx = 0
    num_pot_layers = len([m for m in model.modules() if isinstance(m, (PoTLinear, PoTConv2d))])

    for name, module in model.named_modules():
        if isinstance(module, (PoTLinear, PoTConv2d)):
            alloc = allocations.get(name)
            if alloc is None:
                continue

            with torch.no_grad():
                alpha = F.softplus(module.alpha).clamp(min=0.01).item()

            is_first = (pot_layer_idx == 0)
            is_last = (pot_layer_idx == num_pot_layers - 1)

            print(f"\n{'='*60}")
            print(f"[DEBUG] Processing layer: {name}")
            print(f"  Type: {type(module).__name__}")
            print(f"  is_first: {is_first}, is_last: {is_last}")
            print(f"  alpha (raw): {module.alpha.item():.6f}")
            print(f"  alpha (softplus+clamp): {alpha:.6f}")
            print(f"  act_scale: {module.act_scale}")
            print(f"  prev_act_scale: {prev_act_scale:.6f}")

            # Calculate adjusted bias (absorb mean for first layer)
            adjusted_bias = calculate_adjusted_bias(
                bias=module.bias.detach().cpu() if module.bias is not None else None,
                weight=module.weight.detach().cpu(),
                mean=config.mean if is_first else None,
                is_first_layer=is_first
            )

            scale_int, shift = calculate_combined_scale(
                alpha=alpha,
                act_scale=module.act_scale if module.act_scale else 1.0,
                prev_act_scale=prev_act_scale,
                input_norm=config.input_norm if config.input_norm else 256,
                is_first_layer=is_first,
                is_last_layer=is_last,
                mean=config.mean if is_first else None,
                std=config.std if is_first else None
            )

            # Debug: Show bias scaling preview
            act_scale_for_bias = module.act_scale if module.act_scale else 1.0
            if adjusted_bias is not None:
                scaled_bias = [int(round(b.item() * act_scale_for_bias)) for b in adjusted_bias[:4]]
                print(f"    [DEBUG] Bias scaling preview:")
                print(f"      act_scale for bias: {act_scale_for_bias:.4f}")
                print(f"      adjusted_bias (first 4): {[round(b.item(), 4) for b in adjusted_bias[:4]]}")
                print(f"      scaled_bias (first 4): {scaled_bias}")

            # Make C-compatible name
            c_name = name.replace('.', '_')
            if c_name and c_name[0].isdigit():
                c_name = 'layer_' + c_name

            # Get dimensions from layer_dims
            dims = layer_dims.get(name, {})
            
            # Determine if ReLU follows this layer
            use_relu = _check_relu_follows(model, name)

            layer_info = {
                'name': c_name,
                'original_name': name,
                'type': type(module).__name__,
                'weight': module.weight.detach().cpu(),
                'bias': adjusted_bias,  # Use adjusted bias (mean absorbed for first layer)
                'alpha': alpha,
                'act_scale': module.act_scale if module.act_scale else 1.0,  # For bias scaling
                'levels': alloc.levels,
                'mode': alloc.mode,
                'scale_int': scale_int,
                'shift': shift,
                'use_relu': use_relu,
                # Dimensions
                'in_h': dims.get('in_h', 0),
                'in_w': dims.get('in_w', 0),
                'out_h': dims.get('out_h', 0),
                'out_w': dims.get('out_w', 0),
                'in_size': dims.get('in_size', 0),
                'out_size': dims.get('out_size', 0),
                'stride': dims.get('stride', 1),
                'padding': dims.get('padding', 0),
            }

            layers.append(layer_info)

            if module.act_scale is not None:
                prev_act_scale = module.act_scale

            pot_layer_idx += 1

        elif isinstance(module, nn.MaxPool2d):
            c_name = name.replace('.', '_')
            if c_name and c_name[0].isdigit():
                c_name = 'layer_' + c_name
            
            dims = layer_dims.get(name, {})
            other_layers.append({
                'name': c_name,
                'original_name': name,
                'type': 'MaxPool2d',
                'kernel_size': module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0],
                'stride': module.stride if isinstance(module.stride, int) else module.stride[0],
                'in_h': dims.get('in_h', 0),
                'in_w': dims.get('in_w', 0),
                'in_ch': dims.get('in_ch', 0),
                'out_h': dims.get('out_h', 0),
                'out_w': dims.get('out_w', 0),
                'in_size': dims.get('in_size', 0),
                'out_size': dims.get('out_size', 0),
            })

        elif isinstance(module, nn.Flatten):
            c_name = name.replace('.', '_')
            if c_name and c_name[0].isdigit():
                c_name = 'layer_' + c_name
            
            dims = layer_dims.get(name, {})
            other_layers.append({
                'name': c_name,
                'original_name': name,
                'type': 'Flatten',
                'in_size': dims.get('in_size', 0),
                'out_size': dims.get('out_size', 0),
            })

    # Compute buffer sizes
    max_buffer_size = _compute_max_buffer_size(layer_dims)
    
    # Get output classes from last layer
    last_layer = layers[-1] if layers else None
    num_classes = last_layer['weight'].shape[0] if last_layer else 10

    # Build ordered layer sequence
    layer_sequence = _build_layer_sequence(model, layers, other_layers)

    # Generate header file
    with open(path, 'w') as f:
        _write_header_preamble(f, config)
        
        # Add input scale info if available
        if hasattr(model, 'input_scale'):
            f.write(f"/* Input scale: {model.input_scale:.3f} (for test data: int8 = normalized_float * input_scale) */\n")
            f.write(f"/* Input max: {model.input_max:.3f} */\n\n")
        
        # Scale functions
        f.write("/* Scale functions using only shifts and adds */\n")
        for layer in layers:
            f.write(generate_scale_func(layer['name'], layer['scale_int'], layer['shift']))

        # MaxPool functions
        for other in other_layers:
            if other['type'] == 'MaxPool2d':
                f.write(_generate_maxpool_func(other))

        # Layer forward functions
        f.write("/* Layer forward functions */\n")
        for layer in layers:
            if layer['mode'] == 'unroll':
                f.write(generate_unrolled_layer(layer))
            else:
                # Fallback purely to unroll or error
                f.write(generate_unrolled_layer(layer))

        # Main prediction function
        f.write(_generate_main_predict(layer_sequence, max_buffer_size, num_classes))

        f.write("\n#endif /* POTNN_MODEL_H */\n")

    print(f"C header generated: {path}")
    print(f"  - {len(layers)} PoT layers")
    print(f"  - {len(other_layers)} other layers (MaxPool, Flatten)")
    print(f"  - Buffer size: {max_buffer_size} bytes")
    print(f"  - Output classes: {num_classes}")


def _write_header_preamble(f, config):
    """Write header file preamble."""
    f.write("/* POTNN Generated Model - MUL-FREE Neural Network */\n")
    f.write("/* Target: Ultra-low-cost MCUs without multiplication */\n")
    f.write(f"/* Flash budget: {config.flash} bytes */\n")
    f.write(f"/* RAM budget: {config.ram} bytes */\n\n")
    f.write("#ifndef POTNN_MODEL_H\n")
    f.write("#define POTNN_MODEL_H\n\n")
    f.write("#include <stdint.h>\n\n")


def _generate_maxpool_func(info: Dict) -> str:
    """Generate MaxPool2d function."""
    name = info['name']
    kernel = info['kernel_size']
    stride = info['stride']
    in_h = info['in_h']
    in_w = info['in_w']
    in_ch = info['in_ch']
    out_h = info['out_h']
    out_w = info['out_w']
    
    code = f"// {name} - MaxPool2d {kernel}x{kernel}, stride {stride}\n"
    code += f"static void {name}_forward(const int8_t* input, int8_t* output) {{\n"
    code += f"    const int IN_H = {in_h}, IN_W = {in_w}, IN_CH = {in_ch};\n"
    code += f"    const int OUT_H = {out_h}, OUT_W = {out_w};\n"
    code += f"    const int K = {kernel}, S = {stride};\n"
    code += f"    \n"
    code += f"    for (int c = 0; c < IN_CH; c++) {{\n"
    code += f"        for (int oy = 0; oy < OUT_H; oy++) {{\n"
    code += f"            for (int ox = 0; ox < OUT_W; ox++) {{\n"
    code += f"                int8_t max_val = -128;\n"
    code += f"                for (int ky = 0; ky < K; ky++) {{\n"
    code += f"                    for (int kx = 0; kx < K; kx++) {{\n"
    code += f"                        int in_y = oy * S + ky;\n"
    code += f"                        int in_x = ox * S + kx;\n"
    code += f"                        if (in_y < IN_H && in_x < IN_W) {{\n"
    code += f"                            int idx = c * IN_H * IN_W + in_y * IN_W + in_x;\n"
    code += f"                            if (input[idx] > max_val) max_val = input[idx];\n"
    code += f"                        }}\n"
    code += f"                    }}\n"
    code += f"                }}\n"
    code += f"                int out_idx = c * OUT_H * OUT_W + oy * OUT_W + ox;\n"
    code += f"                output[out_idx] = max_val;\n"
    code += f"            }}\n"
    code += f"        }}\n"
    code += f"    }}\n"
    code += "}\n\n"
    return code


def _generate_main_predict(layer_sequence: List[Dict], buffer_size: int, num_classes: int) -> str:
    """Generate the main prediction function."""
    code = "/* Main prediction function */\n"
    code += "int8_t potnn_predict(const int8_t* input) {\n"
    code += f"    static int8_t buffer1[{buffer_size}];\n"
    code += f"    static int8_t buffer2[{buffer_size}];\n"
    code += "    const int8_t* current_input = input;\n"
    code += "    int8_t* current_output = buffer1;\n\n"

    for i, layer in enumerate(layer_sequence):
        name = layer['name']
        layer_type = layer['type']
        is_last = (i == len(layer_sequence) - 1)

        if layer_type == 'Flatten':
            code += f"    // {name}: Flatten (no-op, just continue)\n"
            code += f"    // Input and output share same memory layout\n\n"
            continue

        code += f"    // {name}: {layer_type}\n"
        code += f"    {name}_forward(current_input, current_output);\n"

        if not is_last:
            code += "    current_input = current_output;\n"
            code += "    current_output = (current_output == buffer1) ? buffer2 : buffer1;\n\n"

    code += f"\n    // Find argmax for classification ({num_classes} classes)\n"
    code += "    int8_t max_val = current_output[0];\n"
    code += "    int8_t max_idx = 0;\n"
    code += f"    for (int i = 1; i < {num_classes}; i++) {{\n"
    code += "        if (current_output[i] > max_val) {\n"
    code += "            max_val = current_output[i];\n"
    code += "            max_idx = i;\n"
    code += "        }\n"
    code += "    }\n"
    code += "    return max_idx;\n"
    code += "}\n"

    return code


def _compute_layer_dimensions(model: nn.Module, input_shape: Tuple) -> Dict[str, Dict]:
    """Compute input/output dimensions for each layer.
    
    Handles both original nn.Conv2d/nn.Linear and PoTConv2d/PoTLinear.
    """
    dims = {}
    
    # Current shape tracking
    if len(input_shape) == 3:
        c, h, w = input_shape
    elif len(input_shape) == 1:
        c, h, w = 1, 1, input_shape[0]
    else:
        c, h, w = 1, input_shape[0], input_shape[1]

    print(f"\nTracking dimensions from input shape: c={c}, h={h}, w={w}")

    for name, module in model.named_modules():
        # Skip container modules
        if name == '':
            continue
            
        # Handle PoTConv2d and nn.Conv2d
        if isinstance(module, PoTConv2d) or isinstance(module, nn.Conv2d):
            in_h, in_w = h, w
            in_ch = c
            
            # Get kernel size, stride, padding
            if isinstance(module, PoTConv2d):
                kh = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                kw = module.kernel_size[1] if isinstance(module.kernel_size, tuple) else module.kernel_size
                sh = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                sw = module.stride[1] if isinstance(module.stride, tuple) else module.stride
                ph = module.padding[0] if isinstance(module.padding, tuple) else module.padding
                pw = module.padding[1] if isinstance(module.padding, tuple) else module.padding
                out_ch = module.out_channels
            else:
                kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
                sh, sw = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
                ph, pw = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
                out_ch = module.out_channels
            
            out_h = (in_h + 2*ph - kh) // sh + 1
            out_w = (in_w + 2*pw - kw) // sw + 1
            
            dims[name] = {
                'in_h': in_h, 'in_w': in_w, 'in_ch': in_ch,
                'out_h': out_h, 'out_w': out_w, 'out_ch': out_ch,
                'in_size': in_ch * in_h * in_w,
                'out_size': out_ch * out_h * out_w,
                'stride': sh,
                'padding': ph if ph == pw else (ph, pw),
            }
            
            # Update current shape
            c, h, w = out_ch, out_h, out_w
            print(f"  {name} (Conv): {in_ch}x{in_h}x{in_w} -> {out_ch}x{out_h}x{out_w}")
            
        elif isinstance(module, nn.MaxPool2d):
            in_h, in_w = h, w
            in_ch = c
            
            k = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
            s = module.stride if isinstance(module.stride, int) else module.stride[0]
            
            out_h = in_h // s
            out_w = in_w // s
            
            dims[name] = {
                'in_h': in_h, 'in_w': in_w, 'in_ch': in_ch,
                'out_h': out_h, 'out_w': out_w,
                'in_size': in_ch * in_h * in_w,
                'out_size': in_ch * out_h * out_w,
            }
            
            # Update current shape
            h, w = out_h, out_w
            print(f"  {name} (MaxPool): {in_ch}x{in_h}x{in_w} -> {in_ch}x{out_h}x{out_w}")
            
        elif isinstance(module, nn.Flatten):
            flat_size = c * h * w
            dims[name] = {
                'in_size': flat_size,
                'out_size': flat_size,
            }
            print(f"  {name} (Flatten): {c}x{h}x{w} -> {flat_size}")
            
        elif isinstance(module, PoTLinear) or isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            
            dims[name] = {
                'in_size': in_features,
                'out_size': out_features,
            }
            
            # Update current shape for potential next linear
            c, h, w = 1, 1, out_features
            print(f"  {name} (Linear): {in_features} -> {out_features}")
            
        # Skip BatchNorm, ReLU, Identity, Dropout - they don't change dimensions

    return dims


def _compute_max_buffer_size(layer_dims: Dict) -> int:
    """Compute maximum buffer size needed."""
    max_size = 256  # Minimum
    for name, dims in layer_dims.items():
        in_size = dims.get('in_size', 0)
        out_size = dims.get('out_size', 0)
        max_size = max(max_size, in_size, out_size)
    return max_size


def _check_relu_follows(model: nn.Module, layer_name: str) -> bool:
    """Check if ReLU follows the given layer.
    
    Skips Identity layers (fused BatchNorm) and BatchNorm layers.
    """
    found_layer = False
    for name, module in model.named_modules():
        if name == layer_name:
            found_layer = True
            continue
        if found_layer:
            # Skip Identity (fused BatchNorm) and BatchNorm
            if isinstance(module, (nn.Identity, nn.BatchNorm1d, nn.BatchNorm2d)):
                continue
            if isinstance(module, (nn.ReLU, nn.ReLU6)):
                return True
            elif isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d)):
                return False  # Another compute layer before ReLU
            # Also check for PoT modules
            if isinstance(module, (PoTLinear, PoTConv2d)):
                return False  # PoTLinear or PoTConv2d
    return False


def _build_layer_sequence(model: nn.Module, pot_layers: List[Dict], other_layers: List[Dict]) -> List[Dict]:
    """Build ordered sequence of all layers."""
    # Create lookup by original name
    pot_lookup = {l['original_name']: l for l in pot_layers}
    other_lookup = {l['original_name']: l for l in other_layers}
    
    sequence = []
    for name, module in model.named_modules():
        if name in pot_lookup:
            sequence.append(pot_lookup[name])
        elif name in other_lookup:
            sequence.append(other_lookup[name])
    
    return sequence
