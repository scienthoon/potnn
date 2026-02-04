"""Model wrapper for potnn conversion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from torch.utils.data import DataLoader

from .config import Config
from .modules import PoTLinear, PoTConv2d
from .quantize.calibration import calibrate_model
from .quantize.qat import prepare_qat, alpha_reg_loss, enable_integer_sim
from .utils import validate_memory, allocate_hybrid
from .fuse import fuse_batchnorm


def _normalize_data(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Normalize input data to match C inference exactly.
    
    C inference uses:
    - /256 (via shift+8) instead of /255
    - avg_std (single value) instead of per-channel std
    
    QAT must match this for training = deployment consistency.
    
    Args:
        data: Input tensor [N, C, H, W] or [N, C, L], expected range [0, 1] from torchvision
        mean: Dataset mean - float or List[float]
        std: Dataset std - float or List[float]
    
    Returns:
        Normalized tensor matching C inference behavior
    """
    import torch
    
    # Calculate average std (C uses single scale value)
    if isinstance(std, (list, tuple)):
        avg_std = sum(std) / len(std)
    else:
        avg_std = std
    
    # Convert mean to tensor with proper shape for broadcasting
    # Dynamic: [1, C, 1] for 3D (Conv1d), [1, C, 1, 1] for 4D (Conv2d)
    if isinstance(mean, (list, tuple)):
        mean = torch.tensor(mean, dtype=data.dtype, device=data.device)
        if data.dim() == 3:  # Conv1d: (B, C, L)
            mean = mean.view(1, -1, 1)
        else:  # Conv2d: (B, C, H, W)
            mean = mean.view(1, -1, 1, 1)
    
    # Match C inference:
    # - data comes as [0,1] from torchvision (raw/255)
    # - C uses raw/256, so multiply by 256/255 to compensate
    # - C uses avg_std in scale, so divide by avg_std
    return (data * (256.0 / 255.0) - mean) / avg_std


def _validate_model(model: nn.Module) -> None:
    """모델에 일반 nn.Conv2d, nn.Linear가 있는지 검사.
    
    potnn은 PoTConv2d, PoTLinear만 지원한다.
    일반 레이어가 섞여 있으면 export 시 실패하므로 미리 경고.
    """
    from .modules.conv import PoTConv2d
    from .modules.depthwise import PoTDepthwiseConv2d
    from .modules.linear import PoTLinear
    
    errors = []
    
    for name, module in model.named_modules():
        # nn.Conv2d지만 PoTConv2d/PoTDepthwiseConv2d가 아닌 경우
        if isinstance(module, nn.Conv2d) and not isinstance(module, (PoTConv2d, PoTDepthwiseConv2d)):
            errors.append(f"  - {name}: nn.Conv2d → potnn.PoTConv2d로 교체 필요")
        
        # nn.Linear지만 PoTLinear가 아닌 경우
        if isinstance(module, nn.Linear) and not isinstance(module, PoTLinear):
            errors.append(f"  - {name}: nn.Linear → potnn.PoTLinear로 교체 필요")
    
    if errors:
        error_msg = "\n".join(errors)
        raise ValueError(
            f"potnn은 PoT 레이어만 지원합니다. 다음 레이어를 교체하세요:\n{error_msg}\n\n"
            f"예시:\n"
            f"  nn.Conv2d(1, 16, 3) → potnn.PoTConv2d(1, 16, 3)\n"
            f"  nn.Linear(256, 10) → potnn.PoTLinear(256, 10)"
        )


def train(model: nn.Module, 
          train_loader: DataLoader,
          test_loader: DataLoader,
          config: Config,
          float_epochs: int = 15,
          qat_epochs: int = 50,
          float_lr: float = 1e-3,
          qat_lr: float = 1e-4,
          device: str = 'cuda',
          fuse_bn: bool = True,
          verbose: bool = True) -> nn.Module:
    """Complete training pipeline: Float → (BN Fusion) → Calibration → QAT → Integer Sim
    
    Args:
        model: PoT model (must use PoTConv2d, PoTLinear)
        train_loader: Training data loader (raw [0,1] input, NO Normalize transform needed)
        test_loader: Test data loader (raw [0,1] input, NO Normalize transform needed)
        config: potnn Config (mean/std used for automatic normalization)
        float_epochs: Float training epochs (default: 15)
        qat_epochs: QAT training epochs (default: 50)
        float_lr: Float training learning rate (default: 1e-3)
        qat_lr: QAT training learning rate (default: 1e-4)
        device: 'cuda' or 'cpu' (default: 'cuda')
        fuse_bn: Fuse BatchNorm layers after float training (default: True)
        verbose: Print progress (default: True)
    
    Returns:
        Trained model with Integer Simulation enabled.
        - model.train(): uses Float QAT for fine-tuning
        - model.eval(): uses Integer Simulation (matches C exactly)
    
    Note:
        Input normalization is handled automatically using config.mean/std.
        Do NOT add transforms.Normalize() to your DataLoader.
    """
    # 모델 검증: 일반 nn.Conv2d, nn.Linear 사용 시 경고
    _validate_model(model)
    
    model = model.to(device)
    
    # Get normalization params from config
    mean = config.mean if config.mean is not None else 0.0
    std = config.std if config.std is not None else 1.0
    
    # Phase 1: Float Training
    if verbose:
        print(f"\n[Phase 1] Float Training ({float_epochs} epochs)...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float_epochs)
    
    best_float_acc = 0
    for epoch in range(float_epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data = _normalize_data(data, mean, std)  # Auto normalize
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        acc = _evaluate(model, test_loader, device, mean, std)
        best_float_acc = max(best_float_acc, acc)
        
        if verbose and (epoch % 5 == 0 or epoch == float_epochs - 1):
            print(f"  Epoch {epoch+1}/{float_epochs}: {acc:.2f}%")
    
    if verbose:
        print(f"  Best Float: {best_float_acc:.2f}%")
    
    # Phase 1.5: BatchNorm Fusion (optional)
    if fuse_bn:
        if verbose:
            print(f"\n[Phase 1.5] BatchNorm Fusion...")
        model = fuse_batchnorm(model)
    
    # Phase 2: Calibration
    if verbose:
        print(f"\n[Phase 2] Calibration...")
    
    calibrate_model(model, train_loader, mean=mean, std=std)
    
    # Phase 3: QAT Preparation
    if verbose:
        print(f"\n[Phase 3] Preparing QAT...")
    
    prepare_qat(model, config)
    
    # Set up first layer info for mean absorption during QAT
    # This ensures QAT uses the same bias as Integer Sim
    from .modules.base import PoTLayerBase
    pot_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, PoTLayerBase)]
    for i, (name, layer) in enumerate(pot_layers):
        is_first = (i == 0)
        is_last = (i == len(pot_layers) - 1)
        layer.set_layer_position(is_first, is_last)
        if is_first:
            layer.set_input_std(config.std, config.mean)
    
    # Phase 4: QAT Training (Hybrid: float first, integer sim last 20%)
    if verbose:
        print(f"\n[Phase 4] QAT Training ({qat_epochs} epochs)...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=qat_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, qat_epochs)
    
    # Activation epochs (last 20%)
    constraint_start_epoch = int(qat_epochs * 0.8)
    integer_sim_start_epoch = int(qat_epochs * 0.6)  # Float QAT 60% → Integer sim fine-tune 40%
    
    best_qat_acc = 0
    best_state = None
    
    for epoch in range(qat_epochs):
        # Enable 5level constraint for last 20% epochs
        if epoch == constraint_start_epoch:
            for name, module in model.named_modules():
                if hasattr(module, 'enforce_5level_constraint'):
                    if hasattr(module, 'encoding') and module.encoding == '5level':
                        module.enforce_5level_constraint = True
                        if verbose:
                            print(f"  [{name}] 5level constraint enabled (epoch {epoch+1})")
        
        # Enable integer sim for last 20% epochs (fine-tune phase)
        if epoch == integer_sim_start_epoch:
            if verbose:
                print(f"  [Integer Sim] Enabled for fine-tuning (epoch {epoch+1})")
            enable_integer_sim(model, input_std=config.std, input_mean=config.mean, verbose=False)
            # Lower learning rate for fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
        
        # Update integer params each epoch if using integer sim
        if epoch >= integer_sim_start_epoch:
            for name, module in model.named_modules():
                if isinstance(module, PoTLayerBase) and module.use_integer_sim:
                    module.compute_integer_params()
        
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data = _normalize_data(data, mean, std)  # Auto normalize
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target) + alpha_reg_loss(model, 0.01)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        acc = _evaluate(model, test_loader, device, mean, std)
        
        # Only update best after integer sim starts (to ensure C-compatible weights)
        if epoch >= integer_sim_start_epoch and acc > best_qat_acc:
            best_qat_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if verbose and (epoch % 10 == 0 or epoch == qat_epochs - 1):
            print(f"  Epoch {epoch+1}/{qat_epochs}: {acc:.2f}%")
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    
    if verbose:
        print(f"  Best QAT: {best_qat_acc:.2f}%")
    
    # Ensure integer sim is enabled for final model
    enable_integer_sim(model, input_std=config.std, input_mean=config.mean, verbose=verbose)
    
    # Final integer params update
    for name, module in model.named_modules():
        if isinstance(module, PoTLayerBase) and module.use_integer_sim:
            module.compute_integer_params()
    
    # Final accuracy (with integer sim)
    final_acc = _evaluate(model, test_loader, device, mean, std)
    
    if verbose:
        print(f"\n[Summary] Float: {best_float_acc:.2f}% → QAT: {best_qat_acc:.2f}% → C-Ready: {final_acc:.2f}%")
    
    # Attach stats to model for reporting
    model.train_stats = {
        'float_acc': best_float_acc,
        'qat_acc': best_qat_acc,
        'final_acc': final_acc
    }
    
    return model


def _evaluate(model: nn.Module, test_loader: DataLoader, device: str, 
              mean: float = 0.0, std: float = 1.0) -> float:
    """Evaluate model accuracy with automatic normalization."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = _normalize_data(data, mean, std)  # Auto normalize
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return 100. * correct / total


# Note: wrap() function has been removed.
# Users must define models using PoT layers directly:
#   potnn.PoTConv2d, potnn.PoTConv1d, potnn.PoTLinear, etc.
# This ensures proper initialization of alpha, QAT parameters, and encoding.