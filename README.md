# PoT-NN: Multiplication-Free Neural Networks for Ultra-Low-Power MCUs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**PoT-NN** is a quantization framework that enables **deep learning inference without multiplication**.  
Run neural networks on ultra-low-cost MCUs without hardware multipliers (CH32V003, PY32F003, etc.).

> 🇰🇷 [한국어 문서](README_ko.md)

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Multiplication-Free** | All weights quantized to powers-of-two, using only `<<`, `>>`, `+` operations |
| **Integer-Only Inference** | No floating-point operations, only `int8`/`int32` arithmetic |
| **5 Encoding Modes** | Choose between accuracy vs. memory tradeoff |
| **Auto C Export** | Generates standalone C header files with zero dependencies |
| **Bit-Exact Matching** | Guaranteed 100% match between Python simulation and C code |

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/potnn.git
cd potnn
pip install -e .
```

## 🚀 Quick Start

### Method 1: One-Line Training (Recommended)

```python
import torch
import torch.nn as nn
import potnn
from potnn import PoTConv2d, PoTLinear

# 1. Define model using PoT layers
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = PoTConv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = PoTConv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)  # Auto-replaced with PoTGlobalAvgPool
        self.fc = PoTLinear(16, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

model = TinyNet()

# 2. Configure
config = potnn.Config(
    flash=16384,      # Target MCU Flash (bytes)
    ram=2048,         # Target MCU RAM (bytes)
    mean=0.1307,      # Dataset mean
    std=0.3081,       # Dataset std
    input_h=16, input_w=16, input_channels=1,
)

# 3. Train (Float → Calibrate → QAT → Integer Sim)
model = potnn.train(model, train_loader, test_loader, config,
                    float_epochs=15, qat_epochs=50)

# 4. Export to C
potnn.export(model, "model.h", config)
```

### Method 2: Manual Pipeline

```python
import potnn

# Step 1: Train float model (standard PyTorch training)
train_float(model, train_loader, epochs=15)

# Step 2: Fuse BatchNorm into Conv (if any)
potnn.fuse_batchnorm(model)

# Step 3: Calibrate activation scales
potnn.calibrate(model, train_loader, config)

# Step 4: Prepare for QAT
potnn.prepare_qat(model, config)

# Step 5: QAT training
train_qat(model, train_loader, epochs=50)

# Step 6: Enable integer simulation (C-compatible)
potnn.enable_integer_sim(model, input_std=config.std, input_mean=config.mean)

# Step 7: Export
potnn.export(model, "model.h", config)
```

## 📊 Encoding Modes

Choose encoding based on accuracy vs. memory tradeoff:

| Encoding | Levels | Values | Bits/Weight | Best For |
|----------|--------|--------|-------------|----------|
| `unroll` | 17 | 0, ±1, ±2, ±4, ..., ±128 | Code-unrolled | Highest accuracy |
| `fp130` | 16 | ±1, ±2, ±4, ..., ±128 | 4-bit | Dense layers |
| `5level` | 5 | -8, -1, 0, +1, +8 | 4-bit (skip) | Balanced |
| `2bit` | 4 | -2, -1, +1, +2 | 2-bit | Smallest memory |
| `ternary` | 3 | -1, 0, +1 | 2-bit (RLE) | Sparse models |

### Per-Layer Encoding

```python
config = potnn.Config(
    flash=16384, ram=2048,
    layer_encodings={
        'conv1': 'unroll',  # First layer: max accuracy
        'conv2': '5level',  # Middle layer
        'fc': 'unroll',     # Last layer: max accuracy
    },
    default_encoding='5level'
)
```

### Encoding Details

#### `unroll` (Default)
- Weights embedded directly as shift-add operations
- Zero weights omitted entirely (sparse-friendly)
- Largest code size, highest accuracy

#### `fp130` (FP1.3.0 Format)
- 4-bit packing: `[sign(1)][exp(3)]`
- No zero (zeros replaced with ±1)
- Good for dense layers

#### `5level` (Skip Encoding)
- 4-bit packing: `[skip(2)][sign(1)][mag(1)]`
- Skip field compresses consecutive zeros (0-3)
- **Constraint**: Max 3 consecutive zeros (4th+ replaced with +1)

#### `2bit`
- 2-bit packing: `[sign(1)][shift(1)]`
- Smallest memory (16 weights per uint32)
- No zero (zeros replaced with ±1)

#### `ternary` (Triple-Run)
- 2-bit codes with run-length encoding
- `11` code = repeat previous value 2 more times
- Best for very sparse models

## 📁 Supported Layers

| Layer | Class | Notes |
|-------|-------|-------|
| Conv2D | `PoTConv2d` | All standard parameters supported |
| Conv1D | `PoTConv1d` | For time series |
| Depthwise | `PoTDepthwiseConv2d` | MobileNet-style |
| Linear | `PoTLinear` | Fully connected |
| GAP | Auto-replaced | `nn.AdaptiveAvgPool2d(1)` → `PoTGlobalAvgPool` |
| Add | `PoTAdd` | For residual connections |
| BatchNorm | Auto-fused | Merged into preceding Conv/Linear |

## ⚙️ API Reference

### `potnn.Config`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `flash` | int | ✅ | Flash memory budget (bytes) |
| `ram` | int | ✅ | RAM budget (bytes) |
| `mean` | float/list | ❌ | Dataset mean (single or per-channel) |
| `std` | float/list | ❌ | Dataset std |
| `input_h`, `input_w` | int | ❌ | Input dimensions (default: 16×16) |
| `input_channels` | int | ❌ | Input channels (default: 1) |
| `layer_encodings` | dict | ❌ | Per-layer encoding override |
| `default_encoding` | str | ❌ | Default encoding (default: 'unroll') |

### Key Functions

```python
potnn.train(model, train_loader, test_loader, config, ...)  # Full pipeline
potnn.calibrate(model, data_loader, config)                  # Calibrate scales
potnn.prepare_qat(model, config)                             # Enable QAT mode
potnn.enable_integer_sim(model, input_std, input_mean)       # C-compatible mode
potnn.export(model, output_path, config)                     # Generate C code
potnn.fuse_batchnorm(model)                                  # Fuse BN layers
```

## 🧪 Verified Results

- **Bit-Exact Matching**: Python integer simulation matches C output 100%
- **MNIST**: 97%+ accuracy with 12KB binary
- **100-Model Stress Test**: Verified across random architectures

## 📝 License

**Dual License**: GPL-3.0 + Commercial

| Use Case | License |
|----------|---------|
| Open Source Projects | GPL-3.0 (Free) |
| Proprietary/Commercial | Commercial License (Contact us) |

See [LICENSE](LICENSE) for details.

## 🙏 Contributing

This project was created by a solo developer without formal CS education.  
There may be bugs, inefficiencies, or areas for improvement.

**Any contributions are greatly appreciated!**
- 🐛 Bug reports
- 💡 Feature suggestions  
- 🔧 Pull requests
- 📖 Documentation improvements

If you find issues or have ideas, please open an issue or PR. Thank you!

---

**Made with ❤️ for ultra-low-power AI**
