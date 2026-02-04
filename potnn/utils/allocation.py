"""Memory allocation and hybrid unroll/loop decision.

Simple heuristic: Top 20% largest layers use loop mode, rest use unroll mode.
"""

import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LayerAllocation:
    """Allocation decision for a single layer."""
    name: str
    weight_count: int
    mode: str  # 'unroll' or 'loop'
    levels: int  # 11 for both (loop uses 4-bit packing)
    
    @property
    def estimated_flash(self) -> int:
        """Estimated Flash usage in bytes."""
        if self.mode == 'unroll':
            # Each weight becomes ~4 bytes of code on average
            return self.weight_count * 4
        else:
            # Loop mode: packed weights + loop code
            # 11 levels = 4-bit = 2 weights per byte
            return (self.weight_count + 1) // 2 + 300  # +300 for loop code + decode table
    
    @property 
    def estimated_ram(self) -> int:
        """Estimated RAM usage in bytes (stack only, weights in Flash)."""
        # Both modes store weights in Flash, RAM is just for accumulators
        return 32  # Fixed stack usage per layer


def allocate_layers(
    layer_infos: List[Dict],
    loop_ratio: float = 0.2,
    force_loop: Optional[List[str]] = None,
    force_unroll: Optional[List[str]] = None,
) -> Dict[str, LayerAllocation]:
    """Allocate layers between unroll and loop modes.
    
    Simple heuristic:
    - Sort layers by weight count (descending)
    - Top `loop_ratio` (default 20%) use loop mode (4 levels)
    - Rest use unroll mode (11 levels)
    
    Args:
        layer_infos: List of layer info dicts with 'name' and weight info
        loop_ratio: Fraction of layers (by count) to use loop mode (0.0-1.0)
        force_loop: Layer names to force into loop mode
        force_unroll: Layer names to force into unroll mode
        
    Returns:
        Dictionary mapping layer names to allocation decisions
    """
    force_loop = set(force_loop or [])
    force_unroll = set(force_unroll or [])
    
    # Collect PoT layers with weight counts
    pot_layers = []
    for info in layer_infos:
        if info.get('layer_type') != 'pot':
            continue
            
        name = info['name']
        
        # Calculate weight count from weight tensor
        weight = info.get('weight')
        if weight is not None:
            weight_count = weight.numel()
        else:
            # Estimate from layer dimensions
            layer_type = info.get('type', '')
            if 'Conv' in layer_type:
                in_ch = info.get('in_channels', 1)
                out_ch = info.get('out_channels', 1)
                ks = info.get('kernel_size', 3)
                weight_count = out_ch * in_ch * ks * ks
            elif 'Linear' in layer_type:
                weight_count = info.get('in_features', 1) * info.get('out_features', 1)
            else:
                weight_count = 0
        
        pot_layers.append({
            'name': name,
            'weight_count': weight_count,
            'info': info
        })
    
    if not pot_layers:
        return {}
    
    # Sort by weight count (largest first)
    pot_layers.sort(key=lambda x: x['weight_count'], reverse=True)
    
    # Calculate how many layers should be loop mode
    total_layers = len(pot_layers)
    num_loop = _calculate_loop_count(total_layers, loop_ratio)
    
    print(f"\nAllocation strategy: {num_loop}/{total_layers} layers will use loop mode")
    
    # Assign modes
    allocations = {}
    for i, layer in enumerate(pot_layers):
        name = layer['name']
        weight_count = layer['weight_count']
        
        # Check forced assignments
        if name in force_loop:
            mode = 'loop'
            levels = 11  # 11-level loop (4-bit packing, no accuracy loss)
        elif name in force_unroll:
            mode = 'unroll'
            levels = 11
        elif i < num_loop:
            # Top N largest → loop mode
            mode = 'loop'
            levels = 11  # 11-level loop (4-bit packing, no accuracy loss)
        else:
            # Rest → unroll mode
            mode = 'unroll'
            levels = 11
        
        allocations[name] = LayerAllocation(
            name=name,
            weight_count=weight_count,
            mode=mode,
            levels=levels
        )
    
    # Print summary
    _print_allocation_summary(allocations, pot_layers)
    
    return allocations


def _calculate_loop_count(total: int, ratio: float) -> int:
    """Calculate number of layers to use loop mode.
    
    Rules:
    - 1-4 layers: 0 loop (all unroll)
    - 5-9 layers: 1 loop
    - 10-14 layers: 2 loop
    - etc.
    """
    if total < 3:
        return 0
    
    # Round to nearest integer, minimum 1 if ratio > 0 and total >= 5
    num_loop = round(total * ratio)
    return max(1, num_loop) if ratio > 0 else 0


def _print_allocation_summary(
    allocations: Dict[str, LayerAllocation],
    pot_layers: List[Dict]
) -> None:
    """Print allocation summary."""
    print("\nLayer allocation summary:")
    print("-" * 70)
    print(f"{'Layer':<30} {'Weights':>10} {'Mode':<8} {'Levels':>6} {'Flash':>10}")
    print("-" * 70)
    
    total_flash = 0
    for layer in pot_layers:
        name = layer['name']
        alloc = allocations[name]
        flash = alloc.estimated_flash
        total_flash += flash
        
        mode_str = f"{alloc.mode}"
        print(f"{name:<30} {alloc.weight_count:>10} {mode_str:<8} {alloc.levels:>6} {flash:>10}")
    
    print("-" * 70)
    
    loop_count = sum(1 for a in allocations.values() if a.mode == 'loop')
    unroll_count = sum(1 for a in allocations.values() if a.mode == 'unroll')
    
    print(f"Total: {len(allocations)} layers ({unroll_count} unroll, {loop_count} loop)")
    print(f"Estimated Flash: {total_flash:,} bytes ({total_flash/1024:.1f} KB)")


def allocate_from_model(
    model: nn.Module,
    loop_ratio: float = 0.2,
) -> Dict[str, LayerAllocation]:
    """Convenience function to allocate directly from PyTorch model.
    
    Args:
        model: PyTorch model with PoT layers
        loop_ratio: Fraction of layers to use loop mode
        
    Returns:
        Dictionary mapping layer names to allocation decisions
    """
    from ..modules.base import PoTLayerBase
    
    # Collect layer infos
    layer_infos = []
    for name, module in model.named_modules():
        if isinstance(module, PoTLayerBase):
            weight_count = module.weight.numel()
            layer_infos.append({
                'name': name,
                'layer_type': 'pot',
                'type': type(module).__name__,
                'weight_count': weight_count,
                'weight': module.weight,
            })
    
    return allocate_layers(layer_infos, loop_ratio=loop_ratio)


# Backward compatibility alias
def allocate_hybrid(
    model: nn.Module,
    flash_budget: int,
    ram_budget: int,
    input_shape: Tuple = (1, 16, 16),
    loop_ratio: float = 0.2,
) -> Dict[str, LayerAllocation]:
    """Backward-compatible wrapper for allocate_from_model.
    
    This function is kept for compatibility with existing code.
    New code should use allocate_layers() or allocate_from_model().
    
    Args:
        model: PyTorch model with PoT layers
        flash_budget: Flash memory budget in bytes (currently ignored)
        ram_budget: RAM budget in bytes (currently ignored)
        input_shape: Input shape (currently ignored)
        loop_ratio: Fraction of layers to use loop mode
        
    Returns:
        Dictionary mapping layer names to allocation decisions
    """
    return allocate_from_model(model, loop_ratio=loop_ratio)