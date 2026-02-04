"""Export functionality for PoT quantized models via ONNX.

Follows the technical specification exactly:
- Input: uint8 [0,255], no runtime normalization
- /256: absorbed into combined_shift (+8)
- mean: absorbed into bias (b' = b - mean × ΣW)  
- /std: absorbed into combined_scale (scale × 1/std)
- bias: scaled by act_scale (bias_int = round(bias × act_scale))
- Output: MUL-free C code (shift+add only, except combined_scale MUL per layer)

v8: Fixed all bugs from systematic verification
- Fixed: bias scaling with act_scale
- Fixed: standardization absorption with act_scale  
- Fixed: last layer act_scale=None handling
- Fixed: weight to shift conversion (w is already PoT*alpha)
"""

import os
import tempfile
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

try:
    import onnx
    from onnx import numpy_helper
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from .modules.base import PoTLayerBase
from .modules.conv import PoTConv2d
from .modules.conv1d import PoTConv1d
from .modules.depthwise import PoTDepthwiseConv2d
from .modules.linear import PoTLinear
from .modules.add import PoTAdd
from .config import Config
from .quantize.pot import quantize_to_pot


def export(model: nn.Module, output_path: str, config: Config, dummy_input: torch.Tensor = None, optimized: bool = True):
    """
    Export PyTorch model to C code.
    
    Args:
        model: PyTorch model (PoT layers)
        output_path: Path to save .c file
        config: Hardware configuration (Config class)
        dummy_input: Dummy input for ONNX export (optional, auto-generated if None)
        optimized: Enable optimized C kernels (default: True)
                   - Loop layers: Full Pipeline (Zero-Padding + im2col + Column-wise + 
                                  Position Blocking + Shift Grouping)
                   - Unroll layers: Zero-Padding (eliminates boundary checks)
    """
    if not ONNX_AVAILABLE:
        raise RuntimeError("onnx package required. Install with: pip install onnx")

    print(f"\nStarting export to {output_path}...")
    print(f"Optimized mode: {'ENABLED' if optimized else 'DISABLED'} (Full Pipeline for loop, Zero-Padding for unroll)")
    print("Using ONNX for graph extraction (v10 - hybrid unroll/loop)...")

    model.eval()
    
    # Step 0a: Disable Integer Simulation for clean ONNX export
    # Integer Sim adds round_ste/clamp_ste which become spurious Add nodes
    from .quantize.qat import disable_integer_sim
    disable_integer_sim(model)
    
    # Step 0a-2: Disable 5level constraint for torch.export compatibility
    # The constraint uses Python for-loops which torch.export doesn't support
    from .modules.base import PoTLayerBase
    for module in model.modules():
        if isinstance(module, PoTLayerBase):
            module.enforce_5level_constraint = False
    
    # Step 0b: Fuse BatchNorm layers into Conv/Linear (CRITICAL!)
    from .fuse import fuse_batchnorm, check_bn_fused
    model = fuse_batchnorm(model)
    if not check_bn_fused(model):
        print("Warning: Some BatchNorm layers could not be fused!")

    if dummy_input is None:
        if config.input_w == 1:
            # Conv1d: (B, C, L) - input_h is sequence length
            dummy_input = torch.randn(1, config.input_channels, config.input_h)
        else:
            # Conv2d: (B, C, H, W)
            dummy_input = torch.randn(1, config.input_channels, config.input_h, config.input_w)
    
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)
    
    # Auto-detect input shape from dummy_input
    if dummy_input is not None:
        if dummy_input.dim() == 4:
            # (B, C, H, W)
            print(f"[Export] Auto-detected input shape: {dummy_input.shape}")
            config.input_channels = dummy_input.shape[1]
            config.input_h = dummy_input.shape[2]
            config.input_w = dummy_input.shape[3]
        elif dummy_input.dim() == 3:
            # (B, C, L)
            print(f"[Export] Auto-detected input shape (1D): {dummy_input.shape}")
            config.input_channels = dummy_input.shape[1]
            config.input_h = dummy_input.shape[2] # Length
            config.input_w = 1

    # Step 1: Collect PoT layer info from PyTorch model
    pot_layer_infos = collect_pot_layer_info(model)
    print(f"Found {len(pot_layer_infos)} PoT layers in PyTorch model")
    
    # Collect PoTAdd layer info (for rescale calculation)
    add_layer_infos = collect_add_layer_info(model)
    
    # Debug: print layer info
    for info in pot_layer_infos:
        print(f"  {info['name']}: alpha={info['alpha']:.4f}, act_scale={info['act_scale']}")

    # Step 2: Export to ONNX
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        onnx_path = f.name

    try:
        # Use legacy ONNX exporter (dynamo=False) to avoid torch.export compatibility issues
        torch.onnx.export(
            model, dummy_input, onnx_path,
            input_names=['input'], output_names=['output'],
            opset_version=11, do_constant_folding=True,
            dynamo=False,  # Legacy exporter
        )
        print(f"ONNX export successful")

        # Step 3: Load ONNX graph
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        layer_infos = parse_onnx_graph(onnx_model, pot_layer_infos, config, add_layer_infos)
        print(f"Parsed {len(layer_infos)} layers from ONNX graph")

    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

    # Step 4: Absorb standardization into first layer (per specification 6.4-6.6)
    if config.mean is not None and config.std is not None:
        absorb_standardization(layer_infos, config.mean, config.std)

    # Step 5: Calculate combined scales (per specification 9.x)
    calculate_combined_scales(layer_infos, config)

    # Step 6: Scale biases by act_scale (CRITICAL FIX)
    scale_biases(layer_infos)
    
    # Step 6.5: Decide loop/unroll mode (All unroll now)
    allocations = {}
    
    # Add allocation info to layer_infos
    for info in layer_infos:
        if info['layer_type'] == 'pot':
            name = info['name']
            if name in allocations:
                alloc = allocations[name]
                info['code_mode'] = alloc.mode  # 'unroll' or 'loop'
                info['levels'] = alloc.levels   # 11 or 4
            else:
                # Default to unroll
                info['code_mode'] = 'unroll'
                info['levels'] = 11

    # Step 8: Generate C header
    code = generate_header(layer_infos, config, optimized=optimized)

    # Step 9: Write to file
    with open(output_path, 'w') as f:
        f.write(code)

    # Print summary
    pot_count = sum(1 for info in layer_infos if info['layer_type'] == 'pot')
    maxpool_count = sum(1 for info in layer_infos if info['layer_type'] == 'maxpool')
    unroll_count = sum(1 for info in layer_infos 
                       if info['layer_type'] == 'pot')
    
    print(f"\nExport complete!")
    print(f"  Output file: {output_path}")
    print(f"  Target MCU: {config.ram}B RAM, {config.flash}B Flash")
    print(f"  PoT layers: {pot_count}")
    print(f"  MaxPool layers: {maxpool_count}")
    print(f"  Total layers: {len(layer_infos)}")


def collect_pot_layer_info(model: nn.Module) -> List[Dict]:
    """Collect alpha, act_scale, weights from PoT layers in order."""
    infos = []
    
    for name, module in model.named_modules():
        if isinstance(module, PoTLayerBase):
            with torch.no_grad():
                alpha = module.alpha.item()
                # act_scale can be None for last layer
                act_scale = module.act_scale.item() if module.act_scale is not None else None
                weight = module.weight.cpu().clone()
                bias = module.bias.cpu().clone() if module.bias is not None else None
                
                # Get encoding from module (default: 'unroll')
                encoding = getattr(module, 'encoding', 'unroll')
                
                # Get PoT quantized weight with encoding-specific levels
                weight_q = quantize_to_pot(weight, alpha, encoding=encoding)
                
                # Apply 5level constraint (max 3 consecutive zeros)
                if encoding == '5level':
                    from .quantize.pot import apply_5level_zero_constraint
                    weight_q = apply_5level_zero_constraint(weight_q)
                
            layer_info = {
                'name': name,
                'pot_index': len(infos),
                'alpha': alpha,
                'act_scale': act_scale,  # Keep None as None!
                'weight': weight_q,
                'bias': bias,  # Original float bias
                'encoding': encoding,
            }
            
            if isinstance(module, PoTConv2d):
                layer_info['type'] = 'conv'
                layer_info['in_channels'] = module.in_channels
                layer_info['out_channels'] = module.out_channels
                layer_info['kernel_size'] = module.kernel_size
                layer_info['stride'] = module.stride
                layer_info['padding'] = module.padding
                layer_info['groups'] = module.groups
            elif isinstance(module, PoTConv1d):
                layer_info['type'] = 'conv1d'
                layer_info['in_channels'] = module.in_channels
                layer_info['out_channels'] = module.out_channels
                layer_info['kernel_size'] = module.kernel_size
                layer_info['stride'] = module.stride
                layer_info['padding'] = module.padding
                layer_info['groups'] = getattr(module, 'groups', 1)
            elif isinstance(module, PoTDepthwiseConv2d):
                layer_info['type'] = 'depthwise'
                layer_info['channels'] = module.channels
                layer_info['in_channels'] = module.channels
                layer_info['out_channels'] = module.channels
                layer_info['kernel_size'] = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                layer_info['stride'] = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                layer_info['padding'] = module.padding[0] if isinstance(module.padding, tuple) else module.padding
                layer_info['groups'] = module.channels
            elif isinstance(module, PoTLinear):
                layer_info['type'] = 'linear'
                layer_info['in_features'] = module.in_features
                layer_info['out_features'] = module.out_features
            
            infos.append(layer_info)
    
    return infos


def collect_add_layer_info(model: nn.Module) -> List[Dict]:
    """Collect PoTAdd layers info in order."""
    infos = []
    
    for name, module in model.named_modules():
        if isinstance(module, PoTAdd):
            layer_info = {
                'name': name,
                'add_index': len(infos),
                'type': 'add',
                'rescale_mult': module.rescale_mult.item() if module.rescale_mult is not None else 128,
                'rescale_shift': module.rescale_shift.item() if module.rescale_shift is not None else 7,
                'scale_x': module.scale_x.item() if module.scale_x is not None else None,
                'scale_y': module.scale_y.item() if module.scale_y is not None else None,
                'act_scale': module.act_scale.item() if module.act_scale is not None else None,
            }
            infos.append(layer_info)
            print(f"  Found PoTAdd: {name}, mult={layer_info['rescale_mult']}, shift={layer_info['rescale_shift']}")
    
    return infos


def parse_onnx_graph(onnx_model, pot_layer_infos: List[Dict], config: Config, add_layer_infos: List[Dict] = None) -> List[Dict]:
    """Parse ONNX graph and build layer_infos list with graph analysis for skip connections."""
    if add_layer_infos is None:
        add_layer_infos = []
    
    graph = onnx_model.graph
    
    initializers = {}
    for init in graph.initializer:
        initializers[init.name] = numpy_helper.to_array(init)
    
    # ========================================
    # Step 1: Build tensor→node mapping for graph analysis
    # ========================================
    tensor_to_producer = {}  # tensor_name → (node_idx, node)
    tensor_to_shape = {}     # tensor_name → (channels, h, w) or (channels, L) or (features,)
    tensor_to_scale = {}     # tensor_name → act_scale (CRITICAL for Add rescale!)
    
    # Detect Conv1d mode
    is_conv1d_model = (config.input_w == 1)
    
    # Input tensor
    input_name = graph.input[0].name
    if is_conv1d_model:
        # Conv1d: (C, L)
        tensor_to_shape[input_name] = (config.input_channels, config.input_h)
    else:
        # Conv2d: (C, H, W)
        tensor_to_shape[input_name] = (config.input_channels, config.input_h, config.input_w)
    tensor_to_producer[input_name] = (-1, None)  # -1 = input
    tensor_to_scale[input_name] = 1.0  # Input scale
    
    # First pass: collect all node outputs and their producers
    for node_idx, node in enumerate(graph.node):
        for out_name in node.output:
            tensor_to_producer[out_name] = (node_idx, node)
    
    # Build tensor→consumers mapping for forward tracing
    tensor_to_consumers = {}  # tensor_name → [(node_idx, node), ...]
    for node_idx, node in enumerate(graph.node):
        for in_name in node.input:
            if in_name not in tensor_to_consumers:
                tensor_to_consumers[in_name] = []
            tensor_to_consumers[in_name].append((node_idx, node))
    
    def trace_forward_to_relu(tensor_name: str) -> bool:
        """순방향 추적: 텐서에서 시작해서 Relu가 나오는지 확인.
        
        Conv → Mul → Round → Clip → Div → Relu 체인에서
        Conv 출력 텐서에서 시작해서 Relu까지 따라감.
        
        중간 노드(Mul, Round, Clip, Div)는 pass through.
        """
        visited = set()
        current = tensor_name
        
        while current and current not in visited:
            visited.add(current)
            
            if current not in tensor_to_consumers:
                return False
            
            consumers = tensor_to_consumers[current]
            if not consumers:
                return False
            
            # 첫 번째 consumer만 따라감 (linear chain 가정)
            _, node = consumers[0]
            op_type = node.op_type
            
            if op_type == 'Relu':
                return True
            
            # Pass through 노드: 출력을 따라감
            if op_type in ('Mul', 'Round', 'Clip', 'Div', 'Cast', 'Add', 'Floor'):
                if node.output:
                    current = node.output[0]
                else:
                    return False
            else:
                # 다른 노드 만나면 중단
                return False
        
        return False
    
    def trace_to_layer(tensor_name: str, tensor_to_layer_idx: dict) -> int:
        """역추적: 텐서에서 시작해서 원래 PoT 레이어까지 추적.
        
        ONNX 그래프에서 Conv → Mul → Round → Clip → Div → Relu 같은
        체인이 있을 때, 어떤 텐서에서든 원래 Conv 레이어를 찾아냄.
        
        범용성: 어떤 중간 노드가 있든 첫 번째 입력을 따라감.
        """
        visited = set()
        current = tensor_name
        
        while current and current not in visited:
            # 이미 매핑된 텐서면 바로 반환
            if current in tensor_to_layer_idx:
                return tensor_to_layer_idx[current]
            
            visited.add(current)
            
            # 이 텐서를 생성한 노드 찾기
            if current in tensor_to_producer:
                node_idx, node = tensor_to_producer[current]
                if node is None:
                    # 입력 텐서
                    return -1
                if len(node.input) > 0:
                    # 첫 번째 입력으로 역추적
                    current = node.input[0]
                else:
                    break
            else:
                break
        
        return -1  # 못 찾음
    
    def trace_to_scale(tensor_name: str) -> float:
        """텐서의 activation scale 추적.
        
        텐서 체인을 따라가며 가장 최근 scale 값을 찾음.
        MaxPool, Flatten 등은 입력 scale을 그대로 전파.
        """
        visited = set()
        current = tensor_name
        
        while current and current not in visited:
            if current in tensor_to_scale:
                return tensor_to_scale[current]
            
            visited.add(current)
            
            if current in tensor_to_producer:
                node_idx, node = tensor_to_producer[current]
                if node is None:
                    return 1.0  # Input
                if len(node.input) > 0:
                    current = node.input[0]
                else:
                    break
            else:
                break
        
        return 1.0  # Default
    
    # ========================================
    # Step 2: Process nodes in order
    # ========================================
    current_h = config.input_h
    current_w = config.input_w
    current_ch = config.input_channels
    
    layer_infos = []
    pot_index = 0
    num_pot_layers = len(pot_layer_infos)
    
    # Track which layer produces which tensor (for skip connections)
    tensor_to_layer_idx = {input_name: -1}  # -1 = input
    
    for node_idx, node in enumerate(graph.node):
        op_type = node.op_type
        output_name = node.output[0] if node.output else None
        
        if op_type == 'Conv':
            pot_info = pot_layer_infos[pot_index] if pot_index < num_pot_layers else None
            
            if pot_info and pot_info['type'] == 'conv1d':
                # ===== Conv1D Processing =====
                weight = pot_info['weight']
                bias = pot_info['bias']
                out_channels = pot_info['out_channels']
                in_channels = pot_info['in_channels']
                alpha = pot_info['alpha']
                act_scale = pot_info['act_scale']
                
                kL = pot_info['kernel_size']
                stride = pot_info['stride']
                padding = pot_info['padding']
                
                # For 1D: current_h represents length, current_w is 1
                in_L = current_h  # Use current_h as length
                out_L = (in_L + 2 * padding - kL) // stride + 1
                
                layer_idx = len(layer_infos)
                
                has_relu = trace_forward_to_relu(output_name) if output_name else False
                
                info = {
                    'name': f'layer_{layer_idx}',
                    'type': 'PoTConv1d',
                    'layer_type': 'pot',
                    'weight': weight,
                    'bias': bias,
                    'alpha': alpha,
                    'act_scale': act_scale,
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                    'kernel_size': kL,
                    'stride': stride,
                    'padding': padding,
                    'in_L': in_L,
                    'out_L': out_L,
                    # Compatibility with 2D: for buffer size calculation
                    'in_h': in_L,
                    'in_w': 1,
                    'out_h': out_L,
                    'out_w': 1,
                    'is_first': (pot_index == 0),
                    'is_last': (pot_index == num_pot_layers - 1),
                    'has_relu': has_relu,
                    'encoding': pot_info.get('encoding', 'unroll'),
                }
                layer_infos.append(info)
                
                if output_name:
                    tensor_to_layer_idx[output_name] = layer_idx
                    tensor_to_shape[output_name] = (out_channels, out_L)
                    if act_scale is not None:
                        tensor_to_scale[output_name] = act_scale
                
                current_h, current_w, current_ch = out_L, 1, out_channels
                pot_index += 1
                
                print(f"  Conv1d: {in_channels}x{in_L} -> {out_channels}x{out_L}")
            
            elif pot_info and pot_info['type'] in ('conv', 'depthwise'):
                # ===== Conv2D Processing =====
                weight = pot_info['weight']
                bias = pot_info['bias']
                out_channels = pot_info['out_channels']
                in_channels = pot_info['in_channels']
                alpha = pot_info['alpha']
                act_scale = pot_info['act_scale']  # Can be None
                is_depthwise = (pot_info['type'] == 'depthwise')
                
                # Use kernel_size, stride, padding from PyTorch model
                kh_tuple = pot_info['kernel_size']
                stride_tuple = pot_info['stride']
                padding_tuple = pot_info['padding']

                # Normalize to (h, w) tuples
                if isinstance(kh_tuple, int): kh_tuple = (kh_tuple, kh_tuple)
                if isinstance(stride_tuple, int): stride_tuple = (stride_tuple, stride_tuple)
                if isinstance(padding_tuple, int): padding_tuple = (padding_tuple, padding_tuple)

                kh, kw = kh_tuple
                sh, sw = stride_tuple
                ph, pw = padding_tuple
                
                # [Support Conv1d]
                # If weight is 3D (Out, In, Length), reshape to 4D (Out, In, 1, Length)
                # to satisfy generic Conv2d generators.
                if weight.ndim == 3:
                    weight = weight.unsqueeze(2)  # (N, C, L) -> (N, C, 1, L)
                    # For Conv1d, kh is usually tuple (k,) or int k
                    # If it was 1D kernel, now it's effectively 1xW.
                    # We might need to ensure kh/kw are correct?
                    # The generator reads shape from w_q.shape usually.
                    pass
            else:
                raise RuntimeError(f"PoT layer mismatch at index {pot_index}")
            input_tensor_name = node.input[0]
            if input_tensor_name in tensor_to_shape:
                input_shape = tensor_to_shape[input_tensor_name]
                in_h = input_shape[1]
                in_w = input_shape[2] if len(input_shape) > 2 else 1
            else:
                in_h, in_w = current_h, current_w  # fallback
            # Continue only for Conv2D (Conv1D handled above and continues)
            if pot_info and pot_info['type'] in ('conv', 'depthwise'):
                out_h = (in_h + 2 * ph - kh) // sh + 1
                out_w = (in_w + 2 * pw - kw) // sw + 1
                
                # Determine layer type
                if is_depthwise:
                    layer_type_name = 'PoTDepthwiseConv2d'
                else:
                    layer_type_name = 'PoTConv2d'
                
                layer_idx = len(layer_infos)
                
                # Check if ReLU follows this Conv (via ONNX chain)
                has_relu = trace_forward_to_relu(output_name) if output_name else False
                
                info = {
                    'name': f'layer_{layer_idx}',
                    'type': layer_type_name,
                    'layer_type': 'pot',
                    'weight': weight,
                    'bias': bias,
                    'alpha': alpha,
                    'act_scale': act_scale,  # Keep None as None
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                    'kernel_size': kh,
                    'stride': sh if sh == sw else (sh, sw),
                    'padding': ph if ph == pw else (ph, pw),
                    'in_h': in_h,
                    'in_w': in_w,
                    'out_h': out_h,
                    'out_w': out_w,
                    'is_first': (pot_index == 0),
                    'is_last': (pot_index == num_pot_layers - 1),
                    'has_relu': has_relu,
                    'encoding': pot_info.get('encoding', 'unroll'),
                    'groups': out_channels if is_depthwise else pot_info.get('groups', 1),
                }
                layer_infos.append(info)
                
                # Track output tensor
                if output_name:
                    tensor_to_layer_idx[output_name] = layer_idx
                    tensor_to_shape[output_name] = (out_channels, out_h, out_w)
                    # Track scale for Add rescale calculation
                    if act_scale is not None:
                        tensor_to_scale[output_name] = act_scale
                
                current_h, current_w, current_ch = out_h, out_w, out_channels
                pot_index += 1
                
                layer_desc = 'DepthwiseConv' if is_depthwise else 'Conv'
                print(f"  {layer_desc}: {in_channels}x{info['in_h']}x{info['in_w']} -> {out_channels}x{out_h}x{out_w}")
        
        elif op_type == 'MaxPool':
            kernel_shape = get_attribute(node, 'kernel_shape', [2, 2])
            strides = get_attribute(node, 'strides', [2, 2])
            
            # Detect 1D vs 2D based on kernel_shape length or is_conv1d_model
            is_maxpool_1d = (len(kernel_shape) == 1) or is_conv1d_model
            
            k = kernel_shape[0]
            s = strides[0]
            
            if is_maxpool_1d:
                # MaxPool1d: only update h (length)
                out_h = current_h // s
                out_w = 1  # Keep as 1 for 1D
                pool_type = 'MaxPool1d'
            else:
                # MaxPool2d
                out_h = current_h // s
                out_w = current_w // s
                pool_type = 'MaxPool2d'
            
            layer_idx = len(layer_infos)
            info = {
                'name': f'layer_{layer_idx}',
                'type': pool_type,
                'layer_type': 'maxpool',
                'kernel_size': k,
                'stride': s,
                'in_h': current_h,
                'in_w': current_w,
                'in_channels': current_ch,
                'out_h': out_h,
                'out_w': out_w,
                'is_1d': is_maxpool_1d,
            }
            layer_infos.append(info)
            
            # Track output tensor
            if output_name:
                tensor_to_layer_idx[output_name] = layer_idx
                if is_maxpool_1d:
                    tensor_to_shape[output_name] = (current_ch, out_h)
                else:
                    tensor_to_shape[output_name] = (current_ch, out_h, out_w)
                # Propagate scale from input (MaxPool doesn't change scale)
                input_name = node.input[0] if len(node.input) > 0 else None
                if input_name:
                    input_scale = trace_to_scale(input_name)
                    tensor_to_scale[output_name] = input_scale
            
            current_h, current_w = out_h, out_w
            
            if is_maxpool_1d:
                print(f"  MaxPool1d: {current_ch}x{info['in_h']} -> {current_ch}x{out_h}")
            else:
                print(f"  MaxPool2d: {current_ch}x{info['in_h']}x{info['in_w']} -> {current_ch}x{out_h}x{out_w}")
        
        elif op_type in ('GlobalAveragePool', 'ReduceMean'):
            # GlobalAveragePool: C×H×W → C×1×1
            # ReduceMean with axes=[2,3]: same effect
            
            # Check if it's ReduceMean over spatial dims
            if op_type == 'ReduceMean':
                axes = get_attribute(node, 'axes', None)
                
                # axes might be in input[1] instead of attribute (newer ONNX)
                if axes is None and len(node.input) > 1:
                    axes_name = node.input[1]
                    if axes_name in initializers:
                        axes = initializers[axes_name].tolist()
                
                # axes=[2,3] or axes=[-2,-1] means spatial reduction
                if axes is None:
                    continue
                if set(axes) not in ({2, 3}, {-2, -1}):
                    continue
            
            layer_idx = len(layer_infos)
            pool_size = current_h * current_w
            
            # Compute div_mult and div_shift for integer division
            if pool_size > 0 and (pool_size & (pool_size - 1)) == 0:
                # Power of 2
                import math
                div_mult = 1
                div_shift = int(math.log2(pool_size))
            else:
                # Not power of 2: avg ≈ (sum * div_mult) >> div_shift
                import math
                base_shift = 15
                div_mult = round((1 << base_shift) / pool_size)
                while div_mult > 255 and base_shift > 8:
                    base_shift -= 1
                    div_mult = round((1 << base_shift) / pool_size)
                div_shift = base_shift
                div_mult = max(1, min(65535, div_mult))
            
            info = {
                'name': f'layer_{layer_idx}',
                'type': 'GlobalAvgPool',
                'layer_type': 'global_avg_pool',
                'in_h': current_h,
                'in_w': current_w,
                'in_channels': current_ch,
                'out_channels': current_ch,
                'pool_size': pool_size,
                'div_mult': div_mult,
                'div_shift': div_shift,
            }
            layer_infos.append(info)
            
            # Track output tensor
            if output_name:
                tensor_to_layer_idx[output_name] = layer_idx
                tensor_to_shape[output_name] = (current_ch, 1, 1)
            
            # After global avg pool: H=1, W=1
            print(f"  GlobalAvgPool: {current_ch}x{current_h}x{current_w} -> {current_ch} (pool_size={pool_size}, mult={div_mult}, shift={div_shift})")
            current_h, current_w = 1, 1
        
        elif op_type in ('Flatten', 'Reshape'):
            shape = None
            if op_type == 'Reshape' and len(node.input) > 1:
                shape_name = node.input[1]
                if shape_name in initializers:
                    shape = initializers[shape_name].tolist()
            
            if op_type == 'Flatten' or (shape and len(shape) == 2):
                out_features = current_h * current_w * current_ch
                
                layer_idx = len(layer_infos)
                info = {
                    'name': f'layer_{layer_idx}',
                    'type': 'Flatten',
                    'layer_type': 'flatten',
                    'in_h': current_h,
                    'in_w': current_w,
                    'in_channels': current_ch,
                    'out_features': out_features,
                }
                layer_infos.append(info)
                
                # Track output tensor
                if output_name:
                    tensor_to_layer_idx[output_name] = layer_idx
                
                print(f"  Flatten: {current_ch}x{current_h}x{current_w} -> {out_features}")
                current_h, current_w = 1, 1
        
        elif op_type == 'Gemm':
            pot_info = pot_layer_infos[pot_index] if pot_index < num_pot_layers else None
            
            if pot_info and pot_info['type'] == 'linear':
                weight = pot_info['weight']
                bias = pot_info['bias']
                in_features = pot_info['in_features']
                out_features = pot_info['out_features']
                alpha = pot_info['alpha']
                act_scale = pot_info['act_scale']  # Can be None
            else:
                raise RuntimeError(f"PoT layer mismatch at index {pot_index}")
            
            layer_idx = len(layer_infos)
            
            # Check if ReLU follows this Linear (via ONNX chain)
            has_relu = trace_forward_to_relu(output_name) if output_name else False
            
            info = {
                'name': f'layer_{layer_idx}',
                'type': 'PoTLinear',
                'layer_type': 'pot',
                'weight': weight,
                'bias': bias,
                'alpha': alpha,
                'act_scale': act_scale,  # Keep None as None
                'in_features': in_features,
                'out_features': out_features,
                'is_first': (pot_index == 0),
                'is_last': (pot_index == num_pot_layers - 1),
                'has_relu': has_relu,
                'encoding': pot_info.get('encoding', 'unroll'),
            }
            layer_infos.append(info)
            
            # Track output tensor
            if output_name:
                tensor_to_layer_idx[output_name] = layer_idx
                # Track scale (Linear is usually last, but just in case)
                if act_scale is not None:
                    tensor_to_scale[output_name] = act_scale
            
            current_ch = out_features
            pot_index += 1
            
            print(f"  Linear: {in_features} -> {out_features}")
        
        elif op_type == 'Relu':
            # ReLU: pass through tensor mapping and scale
            if len(node.input) > 0 and len(node.output) > 0:
                relu_input = node.input[0]
                relu_output = node.output[0]
                if relu_input in tensor_to_layer_idx:
                    tensor_to_layer_idx[relu_output] = tensor_to_layer_idx[relu_input]
                if relu_input in tensor_to_shape:
                    tensor_to_shape[relu_output] = tensor_to_shape[relu_input]
                # Propagate scale
                input_scale = trace_to_scale(relu_input)
                tensor_to_scale[relu_output] = input_scale
        
        elif op_type in ('Mul', 'Round', 'Clip', 'Div'):
            # 양자화 관련 중간 노드: 텐서 매핑 및 scale 전파
            if len(node.input) > 0 and len(node.output) > 0:
                node_input = node.input[0]
                node_output = node.output[0]
                if node_input in tensor_to_layer_idx:
                    tensor_to_layer_idx[node_output] = tensor_to_layer_idx[node_input]
                if node_input in tensor_to_shape:
                    tensor_to_shape[node_output] = tensor_to_shape[node_input]
                # Propagate scale
                input_scale = trace_to_scale(node_input)
                tensor_to_scale[node_output] = input_scale
        
        elif op_type == 'Add':
            # ========================================
            # ONNX Add 노드 분류 (robust version)
            # 
            # ONNX에서 Add 노드가 생기는 경우:
            # 1. Bias addition: Conv/Linear의 bias (한 입력이 initializer)
            # 2. Skip connection: x + conv(x) (두 입력 모두 레이어 출력)
            # 3. 기타 산술 연산
            #
            # 판단 기준:
            # - 한 입력이 initializer → bias add → 무시
            # - 두 입력의 소스 레이어가 같음 → 무시
            # - 두 입력의 소스 레이어가 다름 → skip connection
            # ========================================
            
            input_a = node.input[0] if len(node.input) > 0 else None
            input_b = node.input[1] if len(node.input) > 1 else None
            
            # ------------------------------------------
            # Case 1: 한 입력이 initializer (상수/bias)
            # ------------------------------------------
            a_is_initializer = input_a in initializers
            b_is_initializer = input_b in initializers
            
            if a_is_initializer or b_is_initializer:
                # Bias addition - NOT a skip connection
                # Just propagate scale, shape, and layer index
                print(f"  [DEBUG] Bias Add detected (input is initializer), skipping")
                if output_name:
                    non_const_input = input_b if a_is_initializer else input_a
                    
                    # Propagate scale
                    input_scale = trace_to_scale(non_const_input)
                    tensor_to_scale[output_name] = input_scale
                    
                    # Propagate shape by tracing back
                    traced_shape = None
                    trace_name = non_const_input
                    trace_visited = set()
                    while trace_name and trace_name not in trace_visited:
                        trace_visited.add(trace_name)
                        if trace_name in tensor_to_shape:
                            traced_shape = tensor_to_shape[trace_name]
                            break
                        if trace_name in tensor_to_producer:
                            _, prod_node = tensor_to_producer[trace_name]
                            if prod_node and prod_node.input:
                                trace_name = prod_node.input[0]
                            else:
                                break
                        else:
                            break
                    if traced_shape:
                        tensor_to_shape[output_name] = traced_shape
                    
                    # Propagate layer index (this Add is pass-through)
                    traced_layer = trace_to_layer(non_const_input, tensor_to_layer_idx)
                    if traced_layer >= 0:
                        tensor_to_layer_idx[output_name] = traced_layer
                continue
            
            # ------------------------------------------
            # Case 2: 둘 다 텐서 - 소스 레이어 분석
            # ------------------------------------------
            source_a = trace_to_layer(input_a, tensor_to_layer_idx)
            source_b = trace_to_layer(input_b, tensor_to_layer_idx)
            
            print(f"  [DEBUG] Add node: source_a={source_a}, source_b={source_b}")
            
            # 같은 소스에서 오면 skip connection 아님
            if source_a == source_b:
                print(f"  [DEBUG] Same source ({source_a}), not a skip connection - skipping")
                if output_name:
                    input_scale = trace_to_scale(input_a)
                    tensor_to_scale[output_name] = input_scale
                    # Propagate shape
                    if input_a in tensor_to_shape:
                        tensor_to_shape[output_name] = tensor_to_shape[input_a]
                    # Propagate layer index
                    if source_a >= 0:
                        tensor_to_layer_idx[output_name] = source_a
                continue
            
            # Case 2.5: 한쪽이 -1이면 (입력 텐서에서 직접) skip connection 아님
            if source_a < 0 or source_b < 0:
                print(f"  [DEBUG] One source is input tensor ({source_a}, {source_b}), not a skip connection - skipping")
                if output_name:
                    # 유효한 소스의 정보를 전파
                    valid_source = max(source_a, source_b)
                    valid_tensor = input_a if source_a >= 0 else input_b
                    input_scale = trace_to_scale(valid_tensor)
                    tensor_to_scale[output_name] = input_scale
                    if valid_tensor in tensor_to_shape:
                        tensor_to_shape[output_name] = tensor_to_shape[valid_tensor]
                    if valid_source >= 0:
                        tensor_to_layer_idx[output_name] = valid_source
                continue
            
            # ------------------------------------------
            # Case 3: 다른 소스 = 실제 skip connection!
            # ------------------------------------------
            print(f"  [DEBUG] Skip connection detected: {source_a} vs {source_b}")
            
            # skip은 더 오래된(더 작은 인덱스) 레이어에서 온 것
            if source_a < source_b:
                skip_source = source_a
                conv_source = source_b
                skip_tensor = input_a
                conv_tensor = input_b
            else:
                skip_source = source_b
                conv_source = source_a
                skip_tensor = input_b
                conv_tensor = input_a
            
            # Count how many PoTAdd layers we've already processed
            add_count = sum(1 for l in layer_infos if l.get('type') == 'PoTAdd')
            
            # ========================================
            # Rescale 파라미터 결정
            # ========================================
            
            # Option 1: PyTorch PoTAdd 모듈에서 calibration 값 사용
            if add_count < len(add_layer_infos) and add_layer_infos[add_count]['scale_x'] is not None:
                skip_act_scale = add_layer_infos[add_count]['scale_x']
                conv_act_scale = add_layer_infos[add_count]['scale_y']
                rescale_mult = add_layer_infos[add_count]['rescale_mult']
                rescale_shift = add_layer_infos[add_count]['rescale_shift']
                print(f"    Add rescale (from PoTAdd calibration): skip_scale={skip_act_scale:.4f}, "
                      f"conv_scale={conv_act_scale:.4f}, mult={rescale_mult}, shift={rescale_shift}")
            
            # Option 2: PoTAdd 없이 skip connection 구현된 경우 - ONNX scale 추적
            else:
                skip_act_scale = trace_to_scale(skip_tensor)
                conv_act_scale = trace_to_scale(conv_tensor)
                
                # Calculate rescale ratio: convert skip scale to conv scale
                # skip_aligned = skip * ratio 해서 conv와 같은 scale로 맞춤
                if skip_act_scale != 0 and skip_act_scale != conv_act_scale:
                    ratio = conv_act_scale / skip_act_scale
                else:
                    ratio = 1.0
                
                # Convert to integer arithmetic: mult * x >> shift ≈ ratio * x
                rescale_shift = 7
                rescale_mult = round(ratio * (1 << rescale_shift))
                rescale_mult = max(1, min(rescale_mult, 512))  # Clamp to valid range
                
                if add_count >= len(add_layer_infos):
                    print(f"    Add rescale (no PoTAdd module - direct skip): "
                          f"skip_scale={skip_act_scale:.4f}, conv_scale={conv_act_scale:.4f}, "
                          f"ratio={ratio:.4f}, mult={rescale_mult}, shift={rescale_shift}")
                else:
                    print(f"    Add rescale (fallback - PoTAdd uncalibrated): "
                          f"skip_scale={skip_act_scale:.4f}, conv_scale={conv_act_scale:.4f}, "
                          f"ratio={ratio:.4f}, mult={rescale_mult}, shift={rescale_shift}")
            
            layer_idx = len(layer_infos)
            
            # Check if ReLU follows this Add (via ONNX chain)
            has_relu = trace_forward_to_relu(output_name) if output_name else False
            
            info = {
                'name': f'layer_{layer_idx}',
                'type': 'PoTAdd',
                'layer_type': 'add',
                'in_h': current_h,
                'in_w': current_w,
                'in_channels': current_ch,
                'out_h': current_h,
                'out_w': current_w,
                'out_channels': current_ch,
                'rescale_mult': rescale_mult,
                'rescale_shift': rescale_shift,
                'skip_source_layer': skip_source,  # skip이 시작된 레이어
                'conv_source_layer': conv_source,  # conv 경로의 마지막 레이어
                'act_scale': conv_act_scale,  # Add output uses conv's scale
                'has_relu': has_relu,
            }
            layer_infos.append(info)
            
            # Track output tensor and its scale
            if output_name:
                tensor_to_layer_idx[output_name] = layer_idx
                tensor_to_shape[output_name] = (current_ch, current_h, current_w)
                tensor_to_scale[output_name] = conv_act_scale
            
            print(f"  Add: {current_ch}x{current_h}x{current_w} (skip from layer_{skip_source}, conv from layer_{conv_source})")
    
    return layer_infos


def get_attribute(node, name: str, default=None):
    """Get attribute value from ONNX node."""
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            elif attr.type == onnx.AttributeProto.INT:
                return attr.i
            elif attr.type == onnx.AttributeProto.FLOATS:
                return list(attr.floats)
            elif attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
    return default


def absorb_standardization(layer_infos: List[Dict], mean: List[float], std: List[float]):
    """Absorb input standardization into first PoT layer.
    
    CRITICAL: Uses avg_std everywhere to match QAT exactly.
    - mean → bias: b' = b - Σ_c (mean[c]/avg_std) × ΣW[:,c,:,:] × α
    - /std → combined_scale: uses average std
    
    Args:
        layer_infos: List of layer info dicts
        mean: Per-channel mean values as list (e.g., [0.4914, 0.4822, 0.4465] for CIFAR-10)
        std: Per-channel std values as list (e.g., [0.2470, 0.2435, 0.2616] for CIFAR-10)
    
    Note: bias scaling by act_scale is done separately in scale_biases()
    """
    # Calculate avg_std upfront
    avg_std = sum(std) / len(std) if std else 1.0
    
    for info in layer_infos:
        if info['layer_type'] == 'pot' and info.get('is_first', False):
            weight = info['weight']
            bias = info['bias']
            alpha = info['alpha']
            layer_type = info['type']
            
            if weight is None:
                return
            
            in_channels = weight.shape[1]
            
            # Validate channel count
            if len(mean) != in_channels or len(std) != in_channels:
                raise ValueError(
                    f"mean/std length ({len(mean)}/{len(std)}) must match "
                    f"first layer's in_channels ({in_channels})"
                )
            
            if bias is None:
                bias = torch.zeros(weight.shape[0])
            
            # Use quantized weights for bias absorption (info['weight'] is already quantized)
            w_q = weight
            
            # Channel-wise bias adjustment using avg_std
            # b' = b - Σ_c (mean[c]/avg_std) × ΣW_q[:,c,...] × α
            for c in range(in_channels):
                if layer_type == 'PoTConv1d':
                    # Conv1D: weight shape is [out_ch, in_ch, kernel_size]
                    weight_sum_c = w_q[:, c, :].sum(dim=1)  # [out_ch]
                else:
                    # Conv2D: weight shape is [out_ch, in_ch, kh, kw]
                    weight_sum_c = w_q[:, c, :, :].sum(dim=(1, 2))  # [out_ch]
                
                bias = bias - (mean[c] / avg_std) * weight_sum_c * alpha

            
            info['bias'] = bias
            info['input_std'] = avg_std
            
            print(f"  Standardization absorbed into {info['name']}: mean={mean}, avg_std={avg_std:.4f}")
            break


def calculate_combined_scales(layer_infos: List[Dict], config: Config):
    """Calculate combined scale for each PoT layer.
    
    Per specification section 6.3 and 9.x:
    - First layer: combined_scale = α × act_scale / std
                   combined_shift = base_shift + 8  (/256 absorbed)
    - Other layers: combined_scale = α × act_scale / prev_act_scale
                    combined_shift = base_shift
    - Last layer (act_scale=None): only α / prev_act_scale (no output quantization)
    """
    prev_act_scale = 1.0
    
    for info in layer_infos:
        # CRITICAL: Update prev_act_scale for Add layers!
        # Add layer changes the scale, so subsequent layers need to use Add's output scale
        if info['layer_type'] == 'add':
            add_act_scale = info.get('act_scale')
            if add_act_scale is not None:
                print(f"  {info['name']}: Add layer, updating prev_act_scale to {add_act_scale:.4f}")
                prev_act_scale = add_act_scale
            continue
        
        if info['layer_type'] != 'pot':
            continue
        
        is_first = info.get('is_first', False)
        is_last = info.get('is_last', False)
        alpha = info['alpha']
        act_scale = info.get('act_scale')  # Can be None
        
        # Calculate combined scale
        if is_first:
            input_std = info.get('input_std', 1.0)
            if act_scale is not None:
                # combined_scale = α × act_scale / std
                combined_scale = alpha * act_scale / input_std
            else:
                # Last layer is also first layer (single layer model)
                combined_scale = alpha / input_std
        else:
            if act_scale is not None:
                # combined_scale = α × act_scale / prev_act_scale
                combined_scale = alpha * act_scale / prev_act_scale
            else:
                # Last layer: no act_scale, output is raw logits
                combined_scale = alpha / prev_act_scale
        
        # Determine shift amount for integer approximation
        base_shift = 0
        scale_magnitude = abs(combined_scale)
        
        # Target: scale_int around 64-256 for precision
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
        
        info['combined_scale'] = combined_scale
        info['scale_int'] = scale_int
        info['combined_shift'] = combined_shift
        info['base_shift'] = base_shift
        
        print(f"  {info['name']}: combined_scale={combined_scale:.6f}, scale_int={scale_int}, shift={combined_shift}, act_scale={act_scale}")

        
        # Update prev_act_scale for next layer
        if act_scale is not None:
            prev_act_scale = act_scale


def scale_biases(layer_infos: List[Dict]):
    """Scale all biases by act_scale.
    
    CRITICAL: In PyTorch training:
        output = conv(x) + bias
        output_quantized = output * act_scale
    
    So bias is also multiplied by act_scale!
    
    For last layer (act_scale=None), bias is used as-is (no quantization).
    """
    for info in layer_infos:
        if info['layer_type'] != 'pot':
            continue
        
        bias = info.get('bias')
        if bias is None:
            info['bias_scaled'] = None
            continue
        
        act_scale = info.get('act_scale')
        
        if act_scale is not None:
            # Scale bias by act_scale
            # Use floor(x + 0.5) to match round_half_up_ste in PyTorch modules
            # (torch.round uses bankers rounding which causes ±1 mismatch)
            bias_scaled = torch.floor(bias * act_scale + 0.5).to(torch.int32)
            info['bias_scaled'] = bias_scaled
            print(f"  {info['name']}: bias scaled by act_scale={act_scale:.2f}, range=[{bias_scaled.min()}, {bias_scaled.max()}]")
        else:
            # Last layer: no act_scale
            # Use floor(x + 0.5) to match round_half_up_ste
            bias_scaled = torch.floor(bias + 0.5).to(torch.int32)
            info['bias_scaled'] = bias_scaled
            print(f"  {info['name']}: last layer, bias not scaled, range=[{bias_scaled.min()}, {bias_scaled.max()}]")


def generate_header(layer_infos: List[Dict[str, Any]], config: Config, optimized: bool = True) -> str:
    """Generate complete C header file.
    
    Args:
        layer_infos: List of layer information dictionaries
        config: Configuration with MCU specs
        optimized: If True, use optimized code generation
    """
    code = []

    code.append("#ifndef POTNN_MODEL_H")
    code.append("#define POTNN_MODEL_H")
    code.append("")
    code.append("#include <stdint.h>")
    code.append("#include <string.h>")  # for memset
    code.append("")

    code.append("/*")
    code.append(f" * PoT Quantized Neural Network")
    code.append(f" * Target: {config.ram}B RAM, {config.flash}B Flash")
    if config.mean is not None:
        # Format list values nicely
        mean_str = ', '.join(f'{m:.4f}' for m in config.mean)
        std_str = ', '.join(f'{s:.4f}' for s in config.std)
        code.append(f" * Input normalization: mean=[{mean_str}], std=[{std_str}]")
        code.append(f" *   (absorbed into first layer: mean→bias, /std→scale, /256→shift)")
    code.append(f" * Input: uint8 [0,255] - no runtime normalization needed")
    if config.input_w == 1:
        # Conv1d: CxL format
        code.append(f" * Input size: {config.input_channels}x{config.input_h}")
    else:
        # Conv2d: CxHxW format
        code.append(f" * Input size: {config.input_channels}x{config.input_h}x{config.input_w}")
    if optimized:
        code.append(f" * Optimized: Full Pipeline (loop), Zero-Padding (unroll)")
    code.append(f" * Generated by potnn v8 (all bugs fixed)")
    code.append(" */")
    code.append("")

    # Generate scale functions
    code.append("// Scale functions (combined_scale MUL, 1회/layer)")
    for info in layer_infos:
        if info['layer_type'] == 'pot':
            scale_func = generate_scale_function(info)
            code.append(scale_func)
    code.append("")

    # Generate layer functions
    code.append("// Layer forward functions")
    for i, info in enumerate(layer_infos):
        if info['layer_type'] == 'pot':
            is_first = info.get('is_first', False)
            # CRITICAL FIX: Disable optimization for first layer due to bias generation bug (missing bias in Ch0)
            use_opt = optimized and not is_first
            layer_code = generate_pot_layer(info, is_first_layer=is_first, optimized=use_opt)
            code.append(layer_code)
        elif info['layer_type'] == 'maxpool':
            layer_code = generate_maxpool_layer(info)
            code.append(layer_code)
        elif info['layer_type'] == 'add':
            layer_code = generate_add_layer(info)
            code.append(layer_code)
        elif info['layer_type'] == 'global_avg_pool':
            layer_code = generate_global_avg_pool_layer(info)
            code.append(layer_code)

    # Generate main predict function
    code.append("// Main prediction function")
    code.append(generate_predict_function(layer_infos, config))

    code.append("#endif // POTNN_MODEL_H")

    return '\n'.join(code)


def generate_scale_function(info: Dict) -> str:
    """Generate scale function using MUL + shift."""
    name = info['name']
    scale_int = info['scale_int']
    shift = info['combined_shift']
    
    code = []
    code.append(f"// {name}: combined_scale={info['combined_scale']:.6f}, scale_int={scale_int}, shift={shift}")
    code.append(f"static inline int32_t scale_{name}(int32_t x) {{")
    
    # Handle edge cases to avoid undefined behavior (1 << -1)
    if shift == 0:
        code.append(f"    return (int32_t)((int64_t)x * {scale_int});")
    elif shift == 1:
        code.append(f"    return (int32_t)(((int64_t)x * {scale_int} + 1) >> 1);")
    else:
        code.append(f"    return (int32_t)(((int64_t)x * {scale_int} + (1 << {shift-1})) >> {shift});")
    
    code.append("}")
    code.append("")
    
    return '\n'.join(code)


def generate_pot_layer(info: Dict, is_first_layer: bool = False, optimized: bool = True) -> str:
    """Generate PoT Conv2d, DepthwiseConv2d, Linear, Add, or GlobalAvgPool layer code.
    
    Supports multiple encoding modes:
    - 'unroll': weights embedded as shift-add operations (11 levels, fastest)
    - 'fp130': FP1.3.0 packed weights (16 levels, no zero)
    - '5level': skip-encoded weights (5 levels: -8,-1,0,+1,+8)
    - '2bit': minimal 2-bit encoding (4 levels: ±1,±2)
    - 'ternary': ternary encoding (3 levels: -1,0,+1)
    - 'loop': weights in packed table with loop (slower, less Flash) [legacy]
    
    When optimized=True:
    - Loop mode: Full Pipeline (Zero-Padding + im2col + Column-wise + Position Blocking + Shift Grouping)
    - Unroll mode: Zero-Padding (eliminates boundary checks)
    """
    layer_type = info['type']
    code_mode = info.get('code_mode', 'unroll')
    levels = info.get('levels', 11)
    encoding = info.get('encoding', 'unroll')
    
    # Debug: print encoding for each layer
    print(f"  [CODEGEN] {info['name']}: encoding={encoding}, code_mode={code_mode}")
    
    # Non-PoT layers always use fixed implementations
    if layer_type == 'PoTAdd':
        return generate_add_layer(info)
    elif layer_type == 'GlobalAvgPool':
        return generate_global_avg_pool_layer(info)
    
    # Encoding-specific code generation (overrides code_mode)
    if encoding == 'fp130':
        from .codegen.fp130 import generate_fp130_layer
        print(f"    → Using FP1.3.0 encoder")
        return generate_fp130_layer(info)
    elif encoding == '2bit':
        from .codegen.bit2 import generate_2bit_layer
        print(f"    → Using 2-bit encoder")
        return generate_2bit_layer(info)
    elif encoding == '5level':
        from .codegen.level5 import generate_5level_layer
        print(f"    → Using 5-level encoder")
        return generate_5level_layer(info)
    elif encoding == 'ternary':
        from .codegen.ternary import generate_ternary_layer
        print(f"    → Using Ternary encoder")
        return generate_ternary_layer(info)
    
    # Default: unroll or loop mode (legacy)
    
    # Default: unroll mode
    if code_mode == 'loop':
        print(f"    [WARNING] 'loop' code_mode is deprecated and loop.py is removed. Falling back to 'unroll' mode.")
        if optimized:
            from .codegen.unroll import generate_unrolled_layer_optimized
            return generate_unrolled_layer_optimized(info, is_first_layer=is_first_layer)
        else:
             # Legacy unroll logic dispatch
             pass # Fallthrough to below
             
    # Standard unroll dispatch logic continues below...
    if True: # Indent anchor
        # Unroll mode
        if optimized:
            from .codegen.unroll import generate_unrolled_layer_optimized
            return generate_unrolled_layer_optimized(info, is_first_layer=is_first_layer)
        else:
            if layer_type == 'PoTConv2d':
                return generate_conv_layer(info, is_first_layer)
            elif layer_type == 'PoTConv1d':
                # Conv1d: use optimized version even in non-optimized mode
                from .codegen.unroll import generate_unrolled_layer_optimized
                return generate_unrolled_layer_optimized(info, is_first_layer=is_first_layer)
            elif layer_type == 'PoTDepthwiseConv2d':
                return generate_depthwise_conv_layer(info, is_first_layer)
            elif layer_type == 'PoTLinear':
                return generate_linear_layer(info, is_first_layer)
    
    return ""


def generate_global_avg_pool_layer(info: Dict) -> str:
    """Generate Global Average Pooling layer.
    
    C×H×W → C (채널당 평균)
    나눗셈을 정수 연산으로: avg = (sum * div_mult) >> div_shift
    """
    name = info['name']
    channels = info['in_channels']
    h = info['in_h']
    w = info['in_w']
    pool_size = info.get('pool_size', h * w)
    div_mult = info.get('div_mult', 1)
    div_shift = info.get('div_shift', 0)
    
    code = []
    code.append(f"// {name} - Global Average Pooling: {channels}x{h}x{w} -> {channels}")
    
    if div_mult == 1:
        # Power of 2: shift only
        code.append(f"// pool_size={pool_size} (2^{div_shift}), using shift only")
        code.append(f"static void {name}_forward(const int8_t* input, int8_t* output) {{")
        code.append(f"    for (int c = 0; c < {channels}; c++) {{")
        code.append(f"        int32_t sum = 0;")
        code.append(f"        for (int i = 0; i < {pool_size}; i++) {{")
        code.append(f"            sum += input[c * {pool_size} + i];")
        code.append(f"        }}")
        code.append(f"        output[c] = (int8_t)((sum + {1 << (div_shift - 1)}) >> {div_shift});")
        code.append(f"    }}")
        code.append(f"}}")
    else:
        # Not power of 2: mult + shift
        code.append(f"// pool_size={pool_size}, div = (sum * {div_mult}) >> {div_shift}")
        code.append(f"static void {name}_forward(const int8_t* input, int8_t* output) {{")
        code.append(f"    for (int c = 0; c < {channels}; c++) {{")
        code.append(f"        int32_t sum = 0;")
        code.append(f"        for (int i = 0; i < {pool_size}; i++) {{")
        code.append(f"            sum += input[c * {pool_size} + i];")
        code.append(f"        }}")
        code.append(f"        int32_t avg = (sum * {div_mult}) >> {div_shift};")
        code.append(f"        output[c] = (int8_t)(avg > 127 ? 127 : (avg < -128 ? -128 : avg));")
        code.append(f"    }}")
        code.append(f"}}")
    
    code.append("")
    return '\n'.join(code)


def generate_add_layer(info: Dict) -> str:
    """Generate Add layer for skip/residual connections.
    
    두 입력의 scale 정합을 정수 MUL + shift로 처리:
        x_aligned = (x * rescale_mult) >> rescale_shift
        output = x_aligned + y
    
    컴파일 타임에 mult/shift 계산, 런타임에 float 없음.
    """
    name = info['name']
    channels = info['in_channels']
    h = info['in_h']
    w = info['in_w']
    rescale_mult = info.get('rescale_mult', 128)
    rescale_shift = info.get('rescale_shift', 7)
    has_relu = info.get('has_relu', False)
    
    size = channels * h * w
    
    code = []
    code.append(f"// {name} - Add (skip connection): {channels}x{h}x{w}")
    code.append(f"// rescale: x_aligned = (x * {rescale_mult}) >> {rescale_shift}")
    if has_relu:
        code.append(f"// ReLU applied after add")
    code.append(f"static void {name}_forward(const int8_t* input_skip, const int8_t* input_conv, int8_t* output) {{")
    code.append(f"    for (int i = 0; i < {size}; i++) {{")
    code.append(f"        int32_t x = (int32_t)input_skip[i];")
    code.append(f"        int32_t y = (int32_t)input_conv[i];")
    
    # Rescale x to match y's scale: x * mult >> shift
    if rescale_mult == 128 and rescale_shift == 7:
        # 기본값 (ratio ≈ 1.0): 최적화 가능
        code.append(f"        // ratio ≈ 1.0, skip rescale")
    else:
        code.append(f"        x = (x * {rescale_mult}) >> {rescale_shift};")
    
    code.append(f"        int32_t sum = x + y;")
    
    # ReLU if needed
    if has_relu:
        code.append(f"        if (sum < 0) sum = 0;")
    
    code.append(f"        // Clamp to int8 range")
    code.append(f"        output[i] = (int8_t)(sum > 127 ? 127 : (sum < -128 ? -128 : sum));")
    code.append(f"    }}")
    code.append(f"}}")
    code.append("")
    
    return '\n'.join(code)


def generate_depthwise_conv_layer(info: Dict, is_first_layer: bool = False) -> str:
    """Generate Depthwise Conv2d layer with PoT weights.
    
    Depthwise Conv: 각 채널이 독립적으로 처리됨.
    weight shape: [channels, 1, kH, kW]
    입출력 채널이 동일 (channels = in_channels = out_channels)
    """
    name = info['name']
    weight = info['weight']
    bias_scaled = info.get('bias_scaled')
    alpha = info['alpha']
    is_last = info.get('is_last', False)
    
    channels = info['in_channels']  # in_channels == out_channels for depthwise
    kh = info['kernel_size']
    kw = kh
    stride = info['stride']
    padding = info['padding']
    in_h = info['in_h']
    in_w = info['in_w']
    out_h = info['out_h']
    out_w = info['out_w']
    
    input_type = "uint8_t" if is_first_layer else "int8_t"
    
    code = []
    code.append(f"// {name} - DepthwiseConv2d: {channels}x{in_h}x{in_w} -> {channels}x{out_h}x{out_w}")
    code.append(f"// Kernel: {kh}x{kw}, Stride: {stride}, Padding: {padding}, alpha={alpha:.4f}")
    if is_first_layer:
        code.append(f"// First layer: input is uint8 [0,255], /256 absorbed in shift")
    if is_last:
        code.append(f"// Last layer: no ReLU")
    code.append(f"static void {name}_forward(const {input_type}* input, int8_t* output) {{")
    code.append(f"    int32_t acc;")
    code.append("")
    
    # Depthwise: 각 채널을 독립적으로 처리
    code.append(f"    for (int c = 0; c < {channels}; c++) {{")
    code.append(f"        for (int oy = 0; oy < {out_h}; oy++) {{")
    code.append(f"            for (int ox = 0; ox < {out_w}; ox++) {{")
    code.append(f"                acc = 0;")
    code.append("")
    
    # 커널 루프 (언롤링)
    for ky in range(kh):
        for kx in range(kw):
            # weight index: [c, 0, ky, kx] → c dimension handled by switch
            iy_base = ky - padding
            ix_base = kx - padding
            
            # Input coordinate: iy = oy * stride + ky - padding
            if stride == 1:
                if iy_base == 0:
                    iy_expr = "oy"
                elif iy_base > 0:
                    iy_expr = f"oy + {iy_base}"
                else:
                    iy_expr = f"oy - {-iy_base}"
            else:
                if iy_base == 0:
                    iy_expr = f"oy * {stride}"
                elif iy_base > 0:
                    iy_expr = f"oy * {stride} + {iy_base}"
                else:
                    iy_expr = f"oy * {stride} - {-iy_base}"
            
            if stride == 1:
                if ix_base == 0:
                    ix_expr = "ox"
                elif ix_base > 0:
                    ix_expr = f"ox + {ix_base}"
                else:
                    ix_expr = f"ox - {-ix_base}"
            else:
                if ix_base == 0:
                    ix_expr = f"ox * {stride}"
                elif ix_base > 0:
                    ix_expr = f"ox * {stride} + {ix_base}"
                else:
                    ix_expr = f"ox * {stride} - {-ix_base}"
            
            # Boundary check
            checks = []
            if stride == 1:
                if iy_base < 0:
                    checks.append(f"oy >= {-iy_base}")
                if iy_base > 0:
                    checks.append(f"oy + {iy_base} < {in_h}")
                if ix_base < 0:
                    checks.append(f"ox >= {-ix_base}")
                if ix_base > 0:
                    checks.append(f"ox + {ix_base} < {in_w}")
            else:
                if iy_base < 0:
                    checks.append(f"oy * {stride} >= {-iy_base}")
                if iy_base > 0:
                    checks.append(f"oy * {stride} + {iy_base} < {in_h}")
                if ix_base < 0:
                    checks.append(f"ox * {stride} >= {-ix_base}")
                if ix_base > 0:
                    checks.append(f"ox * {stride} + {ix_base} < {in_w}")
            
            # Input index: c * in_h * in_w + iy * in_w + ix
            idx_expr = f"c * {in_h} * {in_w} + ({iy_expr}) * {in_w} + ({ix_expr})"
            
            # Generate per-channel weight switch
            code.append(f"                // Kernel position ({ky}, {kx})")
            
            # Check which channels have non-zero weights at this kernel position
            non_zero_channels = []
            for c in range(channels):
                w = weight[c, 0, ky, kx].item()
                if abs(w) >= 0.5:  # Not zero
                    w_abs = abs(w)
                    k = round(np.log2(w_abs + 1e-10))
                    k = max(0, min(4, k))
                    sign = '+' if w > 0 else '-'
                    non_zero_channels.append((c, k, sign))
            
            if len(non_zero_channels) == 0:
                code.append(f"                // All weights zero at this position")
                continue
            
            # Check if all non-zero channels have same k and sign
            all_same = len(set((k, sign) for _, k, sign in non_zero_channels)) == 1
            
            if all_same and len(non_zero_channels) == channels:
                # All channels same: no switch needed
                k, sign = non_zero_channels[0][1], non_zero_channels[0][2]
                if k == 0:
                    shift_expr = f"(int32_t)input[{idx_expr}]"
                else:
                    shift_expr = f"(int32_t)input[{idx_expr}] << {k}"
                
                if checks:
                    cond = " && ".join(checks)
                    code.append(f"                if ({cond}) acc {sign}= {shift_expr};")
                else:
                    code.append(f"                acc {sign}= {shift_expr};")
            else:
                # Different weights per channel: use switch
                cond_prefix = ""
                if checks:
                    cond = " && ".join(checks)
                    cond_prefix = f"if ({cond}) "
                
                code.append(f"                {cond_prefix}switch (c) {{")
                for c, k, sign in non_zero_channels:
                    if k == 0:
                        shift_expr = f"(int32_t)input[{idx_expr}]"
                    else:
                        shift_expr = f"(int32_t)input[{idx_expr}] << {k}"
                    code.append(f"                    case {c}: acc {sign}= {shift_expr}; break;")
                code.append(f"                }}")
            
            code.append("")
    
    # Apply scale
    code.append(f"                acc = scale_{name}(acc);")
    
    # Add scaled bias (per-channel)
    if bias_scaled is not None:
        code.append(f"                // Add per-channel bias")
        # DEBUG: Print bias values during generation
        if name == 'layer_0':
            print(f"  [GEN DEBUG] {name} bias_scaled: {bias_scaled[:5].tolist()}...")
        code.append(f"                switch (c) {{")

        for c in range(channels):
            b = int(bias_scaled[c].item())
            if b != 0:
                code.append(f"                    case {c}: acc += {b}; break;")
        code.append(f"                }}")
    
    # ReLU (based on ONNX graph analysis)
    if info.get('has_relu', False):
        code.append(f"                if (acc < 0) acc = 0;")
    
    # Clamp and store
    code.append(f"                int out_idx = c * {out_h} * {out_w} + oy * {out_w} + ox;")
    code.append(f"                output[out_idx] = (int8_t)(acc > 127 ? 127 : (acc < -128 ? -128 : acc));")
    
    code.append(f"            }}")
    code.append(f"        }}")
    code.append(f"    }}")
    code.append(f"}}")
    code.append("")
    
    return '\n'.join(code)


def generate_conv1x1_layer(info: Dict, is_first_layer: bool = False) -> str:
    """Generate optimized 1x1 Conv2d layer with PoT weights.
    
    1x1 Conv는 spatial 연산 없이 채널 간 내적만 수행.
    패딩/커널 루프 불필요 → 단순화된 코드 생성.
    """
    name = info['name']
    weight = info['weight']
    bias_scaled = info.get('bias_scaled')
    alpha = info['alpha']
    is_last = info.get('is_last', False)
    
    in_ch = info['in_channels']
    out_ch = info['out_channels']
    stride = info['stride']
    in_h = info['in_h']
    in_w = info['in_w']
    out_h = info['out_h']
    out_w = info['out_w']
    
    input_type = "uint8_t" if is_first_layer else "int8_t"
    
    code = []
    code.append(f"// {name} - Conv2d 1x1: {in_ch}x{in_h}x{in_w} -> {out_ch}x{out_h}x{out_w}")
    code.append(f"// Stride: {stride}, alpha={alpha:.4f}")
    if is_first_layer:
        code.append(f"// First layer: input is uint8 [0,255], /256 absorbed in shift")
    if is_last:
        code.append(f"// Last layer: no ReLU")
    code.append(f"static void {name}_forward(const {input_type}* input, int8_t* output) {{")
    code.append(f"    int32_t acc;")
    code.append("")
    
    # 1x1 Conv: 각 출력 위치에서 채널 내적만 수행
    code.append(f"    for (int oy = 0; oy < {out_h}; oy++) {{")
    code.append(f"        for (int ox = 0; ox < {out_w}; ox++) {{")
    
    # stride 적용: 입력 좌표 계산
    if stride == 1:
        code.append(f"            int iy = oy;")
        code.append(f"            int ix = ox;")
    else:
        code.append(f"            int iy = oy * {stride};")
        code.append(f"            int ix = ox * {stride};")
    code.append("")
    
    for oc in range(out_ch):
        code.append(f"            // Output channel {oc}")
        code.append(f"            acc = 0;")
        
        # 채널 내적 (1x1이므로 ky, kx 루프 없음)
        for ic in range(in_ch):
            w = weight[oc, ic, 0, 0].item()
            if abs(w) < 1e-9:  # Skip zero weights
                continue
            
            w_abs = abs(w)
            if w_abs < 0.5:  # Too small, skip
                continue
            
            # Find k where 2^k ≈ pot_value
            k = round(np.log2(w_abs + 1e-10))
            k = max(0, min(4, k))  # k in [0, 4] for {1, 2, 4, 8, 16}
            
            sign = '+' if w > 0 else '-'
            
            # 입력 인덱스: [ic, iy, ix] in CHW layout
            if in_ch > 1:
                idx = f"{ic} * {in_h} * {in_w} + iy * {in_w} + ix"
            else:
                idx = f"iy * {in_w} + ix"
            
            if k == 0:
                shift_expr = f"(int32_t)input[{idx}]"
            else:
                shift_expr = f"(int32_t)input[{idx}] << {k}"
            
            code.append(f"            acc {sign}= {shift_expr};")
        
        # Apply scale
        code.append(f"            acc = scale_{name}(acc);")
        
        # Add scaled bias
        if bias_scaled is not None:
            b = int(bias_scaled[oc].item())
            if b != 0:
                code.append(f"            acc += {b};")
        
        # ReLU (based on ONNX graph analysis)
        if info.get('has_relu', False):
            code.append(f"            if (acc < 0) acc = 0;")
        
        # Clamp and store
        out_idx = f"{oc} * {out_h} * {out_w} + oy * {out_w} + ox"
        code.append(f"            output[{out_idx}] = (int8_t)(acc > 127 ? 127 : (acc < -128 ? -128 : acc));")
        code.append("")
    
    code.append(f"        }}")
    code.append(f"    }}")
    code.append(f"}}")
    code.append("")
    
    return '\n'.join(code)


def generate_conv_layer(info: Dict, is_first_layer: bool = False) -> str:
    """Generate Conv2d layer with PoT weights."""
    name = info['name']
    weight = info['weight']
    bias_scaled = info.get('bias_scaled')
    alpha = info['alpha']
    is_last = info.get('is_last', False)
    
    in_ch = info['in_channels']
    out_ch = info['out_channels']
    kh = info['kernel_size']
    kw = kh
    stride = info['stride']
    padding = info['padding']
    in_h = info['in_h']
    in_w = info['in_w']
    out_h = info['out_h']
    out_w = info['out_w']
    
    # 1x1 Conv 최적화: 별도 함수로 처리
    if kh == 1 and kw == 1:
        return generate_conv1x1_layer(info, is_first_layer)
    
    # Input type: uint8 for first layer (raw input), int8 for others
    input_type = "uint8_t" if is_first_layer else "int8_t"
    
    code = []
    code.append(f"// {name} - Conv2d: {in_ch}x{in_h}x{in_w} -> {out_ch}x{out_h}x{out_w}")
    code.append(f"// Kernel: {kh}x{kw}, Stride: {stride}, Padding: {padding}, alpha={alpha:.4f}")
    if is_first_layer:
        code.append(f"// First layer: input is uint8 [0,255], /256 absorbed in shift")
    if is_last:
        code.append(f"// Last layer: no ReLU")
    code.append(f"static void {name}_forward(const {input_type}* input, int8_t* output) {{")
    code.append(f"    int32_t acc;")
    code.append("")
    
    # Generate unrolled convolution
    code.append(f"    for (int oy = 0; oy < {out_h}; oy++) {{")
    code.append(f"        for (int ox = 0; ox < {out_w}; ox++) {{")
    
    for oc in range(out_ch):
        code.append(f"            // Output channel {oc}")
        code.append(f"            acc = 0;")
        
        # Generate weight operations
        for ic in range(in_ch):
            for ky in range(kh):
                for kx in range(kw):
                    w = weight[oc, ic, ky, kx].item()
                    if abs(w) < 1e-9:  # Skip zero weights
                        continue
                    
                    # weight_q = pot_value * alpha
                    # pot_value = weight_q / alpha ∈ {0, 1, 2, 4, 8, 16}
                    w_abs = abs(w)
                    pot_value = w_abs
                    
                    # pot_value should be close to 1, 2, 4, 8, or 16
                    if pot_value < 0.5:  # Too small, skip
                        continue
                    
                    # Find k where 2^k ≈ pot_value
                    k = round(np.log2(pot_value + 1e-10))
                    k = max(0, min(4, k))  # k in [0, 4] for {1, 2, 4, 8, 16}
                    
                    sign = '+' if w > 0 else '-'
                    
                    # Calculate input coordinates with stride (BUG FIX!)
                    # iy = oy * stride + ky - padding
                    # ix = ox * stride + kx - padding
                    iy_base = ky - padding  # offset from oy * stride
                    ix_base = kx - padding  # offset from ox * stride
                    
                    # Build index expression with stride
                    idx_parts = []
                    if in_ch > 1:
                        idx_parts.append(f"{ic} * {in_h} * {in_w}")
                    
                    # Y coordinate: oy * stride + iy_base
                    if stride == 1:
                        if iy_base == 0:
                            idx_parts.append(f"oy * {in_w}")
                        elif iy_base > 0:
                            idx_parts.append(f"(oy + {iy_base}) * {in_w}")
                        else:
                            idx_parts.append(f"(oy - {-iy_base}) * {in_w}")
                    else:
                        if iy_base == 0:
                            idx_parts.append(f"oy * {stride} * {in_w}")
                        elif iy_base > 0:
                            idx_parts.append(f"(oy * {stride} + {iy_base}) * {in_w}")
                        else:
                            idx_parts.append(f"(oy * {stride} - {-iy_base}) * {in_w}")
                    
                    # X coordinate: ox * stride + ix_base
                    if stride == 1:
                        if ix_base == 0:
                            idx_parts.append("ox")
                        elif ix_base > 0:
                            idx_parts.append(f"(ox + {ix_base})")
                        else:
                            idx_parts.append(f"(ox - {-ix_base})")
                    else:
                        if ix_base == 0:
                            idx_parts.append(f"ox * {stride}")
                        elif ix_base > 0:
                            idx_parts.append(f"(ox * {stride} + {ix_base})")
                        else:
                            idx_parts.append(f"(ox * {stride} - {-ix_base})")
                    
                    idx = " + ".join(idx_parts)
                    
                    # Boundary conditions for padding (with stride)
                    checks = []
                    if stride == 1:
                        if iy_base < 0:
                            checks.append(f"oy >= {-iy_base}")
                        if iy_base > 0:
                            checks.append(f"oy + {iy_base} < {in_h}")
                        if ix_base < 0:
                            checks.append(f"ox >= {-ix_base}")
                        if ix_base > 0:
                            checks.append(f"ox + {ix_base} < {in_w}")
                    else:
                        if iy_base < 0:
                            checks.append(f"oy * {stride} >= {-iy_base}")
                        if iy_base > 0:
                            checks.append(f"oy * {stride} + {iy_base} < {in_h}")
                        if ix_base < 0:
                            checks.append(f"ox * {stride} >= {-ix_base}")
                        if ix_base > 0:
                            checks.append(f"ox * {stride} + {ix_base} < {in_w}")
                    
                    # Generate code
                    if k == 0:
                        shift_expr = f"(int32_t)input[{idx}]"
                    else:
                        shift_expr = f"(int32_t)input[{idx}] << {k}"
                    
                    if checks:
                        cond = " && ".join(checks)
                        code.append(f"            if ({cond}) acc {sign}= {shift_expr};")
                    else:
                        code.append(f"            acc {sign}= {shift_expr};")
        
        # Apply scale
        code.append(f"            acc = scale_{name}(acc);")
        
        # Add scaled bias (CRITICAL FIX: use bias_scaled, not raw bias)
        if bias_scaled is not None:
            b = int(bias_scaled[oc].item())
            if b != 0:
                code.append(f"            acc += {b};")
        
        # ReLU (based on ONNX graph analysis)
        if info.get('has_relu', False):
            code.append(f"            if (acc < 0) acc = 0;")
        
        # Clamp and store
        out_idx = f"{oc * out_h * out_w} + oy * {out_w} + ox"
        code.append(f"            output[{out_idx}] = (int8_t)(acc > 127 ? 127 : (acc < -128 ? -128 : acc));")
        code.append("")
    
    code.append(f"        }}")
    code.append(f"    }}")
    code.append(f"}}")
    code.append("")
    
    return '\n'.join(code)


def generate_linear_layer(info: Dict, is_first_layer: bool = False) -> str:
    """Generate Linear layer with PoT weights."""
    name = info['name']
    weight = info['weight']
    bias_scaled = info.get('bias_scaled')
    alpha = info['alpha']
    is_last = info.get('is_last', False)
    
    in_features = info['in_features']
    out_features = info['out_features']
    
    input_type = "uint8_t" if is_first_layer else "int8_t"
    
    code = []
    code.append(f"// {name} - Linear: {in_features} -> {out_features}, alpha={alpha:.4f}")
    if is_last:
        code.append(f"// Last layer: no ReLU")
    code.append(f"static void {name}_forward(const {input_type}* input, int8_t* output) {{")
    code.append(f"    int32_t acc;")
    code.append("")
    
    for o in range(out_features):
        code.append(f"    // Output {o}")
        code.append(f"    acc = 0;")
        
        for i in range(in_features):
            w = weight[o, i].item()
            if abs(w) < 1e-9:
                continue
            
            w_abs = abs(w)
            pot_value = w_abs
            
            if pot_value < 0.5:
                continue
            
            k = round(np.log2(pot_value + 1e-10))
            k = max(0, min(4, k))
            
            sign = '+' if w > 0 else '-'
            
            if k == 0:
                code.append(f"    acc {sign}= (int32_t)input[{i}];")
            else:
                code.append(f"    acc {sign}= (int32_t)input[{i}] << {k};")
        
        code.append(f"    acc = scale_{name}(acc);")
        
        # Add scaled bias (CRITICAL FIX)
        if bias_scaled is not None:
            b = int(bias_scaled[o].item())
            if b != 0:
                code.append(f"    acc += {b};")
        
        # ReLU (based on ONNX graph analysis)
        if info.get('has_relu', False):
            code.append(f"    if (acc < 0) acc = 0;")
        
        code.append(f"    output[{o}] = (int8_t)(acc > 127 ? 127 : (acc < -128 ? -128 : acc));")
        code.append("")
    
    code.append(f"}}")
    code.append("")
    
    return '\n'.join(code)


def generate_maxpool_layer(info: Dict) -> str:
    """Generate MaxPool1d or MaxPool2d layer."""
    name = info['name']
    in_h = info['in_h']
    in_w = info['in_w']
    in_ch = info['in_channels']
    out_h = info['out_h']
    out_w = info['out_w']
    k = info['kernel_size']
    s = info['stride']
    is_1d = info.get('is_1d', False)
    
    code = []
    
    if is_1d:
        # MaxPool1d
        code.append(f"// {name} - MaxPool1d k={k}, stride={s}")
        code.append(f"static void {name}_forward(const int8_t* input, int8_t* output) {{")
        code.append(f"    for (int c = 0; c < {in_ch}; c++) {{")
        code.append(f"        for (int o = 0; o < {out_h}; o++) {{")
        code.append(f"            int8_t max_val = -128;")
        code.append(f"            for (int ki = 0; ki < {k}; ki++) {{")
        code.append(f"                int idx = c * {in_h} + (o * {s} + ki);")
        code.append(f"                if (input[idx] > max_val) max_val = input[idx];")
        code.append(f"            }}")
        code.append(f"            output[c * {out_h} + o] = max_val;")
        code.append(f"        }}")
        code.append(f"    }}")
        code.append(f"}}")
    else:
        # MaxPool2d
        code.append(f"// {name} - MaxPool2d {k}x{k}, stride {s}")
        code.append(f"static void {name}_forward(const int8_t* input, int8_t* output) {{")
        code.append(f"    for (int c = 0; c < {in_ch}; c++) {{")
        code.append(f"        for (int oy = 0; oy < {out_h}; oy++) {{")
        code.append(f"            for (int ox = 0; ox < {out_w}; ox++) {{")
        code.append(f"                int8_t max_val = -128;")
        code.append(f"                for (int ky = 0; ky < {k}; ky++) {{")
        code.append(f"                    for (int kx = 0; kx < {k}; kx++) {{")
        code.append(f"                        int idx = c * {in_h} * {in_w} + (oy * {s} + ky) * {in_w} + (ox * {s} + kx);")
        code.append(f"                        if (input[idx] > max_val) max_val = input[idx];")
        code.append(f"                    }}")
        code.append(f"                }}")
        code.append(f"                output[c * {out_h} * {out_w} + oy * {out_w} + ox] = max_val;")
        code.append(f"            }}")
        code.append(f"        }}")
        code.append(f"    }}")
        code.append(f"}}")
    
    code.append("")
    
    return '\n'.join(code)


def generate_predict_function(layer_infos: List[Dict], config: Config) -> str:
    """Generate main prediction function with proper skip connection handling."""
    code = []
    
    input_size = config.input_h * config.input_w
    
    # Calculate max buffer size
    max_buffer_size = input_size
    for info in layer_infos:
        if 'out_h' in info and 'out_w' in info:
            if 'out_channels' in info:
                size = info['out_channels'] * info['out_h'] * info['out_w']
            elif 'in_channels' in info:
                size = info['in_channels'] * info['out_h'] * info['out_w']
            else:
                size = info['out_h'] * info['out_w']
            max_buffer_size = max(max_buffer_size, size)
        elif 'out_features' in info:
            max_buffer_size = max(max_buffer_size, info['out_features'])
    
    # ========================================
    # Analyze skip connections
    # ========================================
    # Collect all skip source layers (where skip branches start)
    skip_sources = set()
    for info in layer_infos:
        if info['layer_type'] == 'add' and 'skip_source_layer' in info:
            skip_sources.add(info['skip_source_layer'])
    
    has_skip = len(skip_sources) > 0
    
    # Find number of output classes
    num_classes = 10
    for info in reversed(layer_infos):
        if 'out_features' in info and info['layer_type'] == 'pot':
            num_classes = info['out_features']
            break

    code.append("// Input: uint8 [0,255] - raw pixel values, no normalization needed")
    code.append("// Output: int8 [0,255] - raw logits")
    code.append("void potnn_predict(const uint8_t* input, int8_t* output) {")
    code.append(f"    // Intermediate buffers (max size: {max_buffer_size})")
    code.append(f"    static int8_t buffer1[{max_buffer_size}];")
    code.append(f"    static int8_t buffer2[{max_buffer_size}];")
    
    if has_skip:
        code.append(f"    static int8_t skip_buffer[{max_buffer_size}];  // for skip connections")
        code.append(f"    // Skip sources: layers {sorted(skip_sources)}")
    
    code.append(f"    int8_t *current = buffer1;")
    code.append(f"    int8_t *next = buffer2;")
    code.append("")

    first_pot_done = False
    
    for i, info in enumerate(layer_infos):
        layer_type = info['layer_type']
        
        if layer_type == 'flatten':
            code.append(f"    // Layer {i}: Flatten (no-op)")
            continue
        
        code.append(f"    // Layer {i}: {info['type']}")
        
        if layer_type == 'pot' and not first_pot_done:
            # First PoT layer: input is uint8
            code.append(f"    {info['name']}_forward(input, current);")
            first_pot_done = True
        
        elif layer_type == 'add':
            # Add layer: skip_buffer + current -> next
            skip_src = info.get('skip_source_layer', -1)
            conv_src = info.get('conv_source_layer', -1)
            code.append(f"    // skip from layer_{skip_src}, conv from layer_{conv_src}")
            code.append(f"    {info['name']}_forward(skip_buffer, current, next);")
            code.append(f"    {{ int8_t *tmp = current; current = next; next = tmp; }}")
        
        elif layer_type == 'global_avg_pool':
            # Global Average Pooling: C×H×W → C
            code.append(f"    {info['name']}_forward(current, next);")
            code.append(f"    {{ int8_t *tmp = current; current = next; next = tmp; }}")
        
        else:
            code.append(f"    {info['name']}_forward(current, next);")
            code.append(f"    {{ int8_t *tmp = current; current = next; next = tmp; }}")
        
        # ========================================
        # Check if this layer is a skip source → save to skip_buffer
        # ========================================
        if i in skip_sources:
            # Find the size of this layer's output
            if 'out_channels' in info and 'out_h' in info and 'out_w' in info:
                skip_size = info['out_channels'] * info['out_h'] * info['out_w']
            elif 'out_features' in info:
                skip_size = info['out_features']
            elif 'in_channels' in info and 'out_h' in info:
                skip_size = info['in_channels'] * info['out_h'] * info['out_w']
            else:
                skip_size = max_buffer_size
            
            code.append(f"    // Save skip (layer {i} is skip source)")
            code.append(f"    for (int _i = 0; _i < {skip_size}; _i++) skip_buffer[_i] = current[_i];")
        
        code.append("")

    code.append(f"    // Copy result to output buffer (num_classes = {num_classes})")
    code.append(f"    for (int i = 0; i < {num_classes}; i++) {{")
    code.append(f"        output[i] = current[i];")
    code.append(f"    }}")
    code.append(f"}}")

    return '\n'.join(code)

    return '\n'.join(code)
