# potnn 라이브러리 리팩토링 프롬프트

## 1. 프로젝트 개요

**potnn = "초저가 MCU를 위한 에너지 효율적 PoT 양자화 컴파일러"**

PyTorch 모델을 받아서 $0.10~$0.50 MCU(CH32V003, PY32F003 등)에서 실행 가능한 C 헤더 파일을 생성하는 라이브러리다.

### 타겟 환경
- RAM: 2~8KB
- Flash: 16~64KB
- MUL 명령어: 없거나 느림
- 32비트 MCU 전용

---

## 2. 절대 위배 불가 핵심 철학 (⚠️ 최우선)

### 2.1 Weight는 반드시 PoT (Power-of-Two)
```
W ∈ {0, ±1, ±2, ±4, ±8, ±16}  (k ∈ {0,1,2,3,4})
```
- 곱셈 → shift 변환의 핵심
- 절대 float weight 허용 안 함

### 2.2 MUL 최소화
```
허용되는 MUL: 레이어당 1회 (combined scale)
나머지: 전부 shift+add
```

### 2.3 데이터 타입 고정
```
입력: int8
Activation: int8
누적: int32 (레지스터, RAM 아님)
Weight: PoT k값 (정수)
```

### 2.4 α 스케일링 필수
```
output = round((Σ(input × w_q) × α + bias) × act_scale)
```
- α 없으면 정확도 -9.6%
- α 있으면 정확도 -0.2%
- α는 학습 가능 파라미터

### 2.5 Calibration 기반 고정 scale
```python
# 학습 전 calibration으로 act_scale 결정
# 학습 중 고정 (STE로 gradient 통과)
act_scale = calibrate(layer, data)  # 학습 전 1회
```

### 2.6 표준화 흡수 (런타임 비용 0)
```
mean → 첫 레이어 bias에 흡수: b' = b - mean × Σ(W)
/std → 첫 레이어 α에 흡수: combined_scale = α × act_scale / std
```

### 2.7 /256 정규화
```
학습: input / 256 (255 아님!)
추론: combined_shift = shift + 8  (/256이 shift에 흡수)
```

---

## 3. 지원해야 할 레이어

### 3.1 필수 (MVP)
| 레이어 | 설명 |
|--------|------|
| PoTConv2d | PoT weight + α 스케일링 |
| PoTLinear | PoT weight + α 스케일링 |
| MaxPool2d | 그대로 통과 |
| ReLU | 그대로 통과 |
| Flatten | 차원 변환만 |

### 3.2 선택 (확장)
| 레이어 | 설명 |
|--------|------|
| PoTDepthwiseConv2d | MobileNet 스타일 |
| AvgPool2d | 드물게 사용 |
| Add | Skip connection |
| BatchNorm2d | Conv에 folding |

---

## 4. 현재 문제점

### 4.1 nn.Sequential 강제
```python
# 현재 (제한적)
model = nn.Sequential(
    nn.Conv2d(1, 8, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    ...
)
model = potnn.wrap(model, config)  # Sequential만 가능
```

### 4.2 개선 목표
```python
# 목표 (자유로운 구조)
class MyModel(nn.Module):
    def __init__(self):
        self.conv1 = potnn.PoTConv2d(1, 8, 3)
        self.conv2 = potnn.PoTConv2d(8, 16, 3)
        self.fc = potnn.PoTLinear(256, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        # skip connection 등 자유롭게
        x = x.flatten(1)
        x = self.fc(x)
        return x

# 학습/추론/export 통합
model.calibrate(train_loader)
model.prepare_qat()
# ... QAT 학습 ...
potnn.export(model, "model.h", config)
```

---

## 5. 모듈 구조 설계

### 5.1 디렉토리 구조
```
potnn/
├── __init__.py          # 공개 API
├── config.py            # MCU 설정 (RAM, Flash, mean, std)
├── modules/
│   ├── __init__.py
│   ├── conv.py          # PoTConv2d
│   ├── linear.py        # PoTLinear
│   ├── depthwise.py     # PoTDepthwiseConv2d (선택)
│   └── base.py          # 공통 PoT 레이어 베이스
├── quantize/
│   ├── __init__.py
│   ├── pot.py           # PoT 양자화 함수 (quantize_to_pot, STE)
│   ├── calibration.py   # act_scale 측정
│   └── qat.py           # QAT 관련 유틸
├── codegen/
│   ├── __init__.py
│   ├── header.py        # C 헤더 생성 진입점
│   ├── scale.py         # scale 함수 생성 (shift+add 분해)
│   ├── unroll.py        # 언롤 코드 생성
│   └── loop.py          # 루프 코드 생성 (선택)
├── export.py            # 모델 → C 헤더 변환
├── fuse.py              # BatchNorm folding
└── utils/
    ├── __init__.py
    └── memory.py        # 메모리 계산
```

### 5.2 PoT 레이어 베이스 클래스
```python
# modules/base.py
class PoTLayerBase(nn.Module):
    """모든 PoT 레이어의 베이스 클래스"""
    
    def __init__(self, levels=11):
        super().__init__()
        # levels: PoT 레벨 수 (11 = {0, ±1, ±2, ±4, ±8, ±16})
        self.levels = levels
        
        # α 스케일링 (학습 가능)
        # raw_alpha → softplus → clamp(0.01) → alpha
        self.raw_alpha = nn.Parameter(torch.tensor(0.5))
        
        # Activation scale (calibration 후 고정)
        self.register_buffer('act_scale', None)
        
        # QAT 모드 플래그
        self.quantize = False
    
    @property
    def alpha(self):
        """softplus + clamp로 양수 보장"""
        return F.softplus(self.raw_alpha).clamp(min=0.01)
    
    def calibrate(self, act_max):
        """act_scale 설정 (calibration 결과)"""
        self.act_scale = 127.0 / act_max
    
    def prepare_qat(self):
        """QAT 모드 활성화"""
        self.quantize = True
    
    def alpha_reg_loss(self, lambda_reg=0.01):
        """α regularization loss"""
        alpha_init = 0.5  # 초기값
        return lambda_reg * (self.alpha - alpha_init) ** 2
```

### 5.3 PoTConv2d 구현
```python
# modules/conv.py
class PoTConv2d(PoTLayerBase):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, bias=True, levels=11):
        super().__init__(levels)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        # Weight (float로 학습, 추론 시 PoT)
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels, *self.kernel_size
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # 초기화
        nn.init.kaiming_normal_(self.weight)
    
    def forward(self, x):
        if self.quantize:
            # PoT 양자화 + α 스케일링
            w_q = quantize_to_pot_ste(self.weight, self.alpha, self.levels)
            out = F.conv2d(x, w_q * self.alpha, self.bias, 
                          self.stride, self.padding)
            
            # Activation 양자화 (act_scale이 있으면)
            if self.act_scale is not None:
                out = torch.round(out * self.act_scale).clamp(-128, 127) / self.act_scale
            
            return out
        else:
            # Float 모드 (warmup)
            return F.conv2d(x, self.weight, self.bias, 
                           self.stride, self.padding)
```

### 5.4 PoTLinear 구현
```python
# modules/linear.py
class PoTLinear(PoTLayerBase):
    def __init__(self, in_features, out_features, bias=True, levels=11):
        super().__init__(levels)
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        nn.init.kaiming_normal_(self.weight)
    
    def forward(self, x):
        if self.quantize:
            w_q = quantize_to_pot_ste(self.weight, self.alpha, self.levels)
            out = F.linear(x, w_q * self.alpha, self.bias)
            
            if self.act_scale is not None:
                out = torch.round(out * self.act_scale).clamp(-128, 127) / self.act_scale
            
            return out
        else:
            return F.linear(x, self.weight, self.bias)
```

---

## 6. PoT 양자화 함수

### 6.1 quantize_to_pot (forward)
```python
# quantize/pot.py
def quantize_to_pot(weight, alpha, levels=11):
    """
    Float weight → PoT 양자화
    
    Args:
        weight: float tensor
        alpha: 스케일 파라미터
        levels: PoT 레벨 수 (11 = {0, ±1, ±2, ±4, ±8, ±16})
    
    Returns:
        PoT 양자화된 weight (정수: -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16)
    """
    # 허용 PoT 값
    if levels == 11:
        pot_values = torch.tensor([0., 1., 2., 4., 8., 16.], device=weight.device)
    elif levels == 9:
        pot_values = torch.tensor([0., 1., 2., 4., 8.], device=weight.device)
    elif levels == 5:
        pot_values = torch.tensor([0., 1., 2.], device=weight.device)
    elif levels == 3:
        pot_values = torch.tensor([0., 1.], device=weight.device)
    else:
        raise ValueError(f"Unsupported levels: {levels}")
    
    # 부호 분리
    sign = torch.sign(weight)
    abs_weight = torch.abs(weight / alpha)
    
    # 가장 가까운 PoT 찾기
    distances = torch.abs(abs_weight.unsqueeze(-1) - pot_values)
    indices = torch.argmin(distances, dim=-1)
    quantized_abs = pot_values[indices]
    
    # 부호 복원
    return sign * quantized_abs
```

### 6.2 STE (Straight-Through Estimator)
```python
class PoTQuantizeSTE(torch.autograd.Function):
    """STE: forward는 양자화, backward는 gradient 그대로 통과"""
    
    @staticmethod
    def forward(ctx, weight, alpha, levels):
        return quantize_to_pot(weight, alpha, levels)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Gradient 그대로 통과 (STE 핵심)
        return grad_output, None, None

def quantize_to_pot_ste(weight, alpha, levels=11):
    return PoTQuantizeSTE.apply(weight, alpha, levels)
```

---

## 7. Calibration

### 7.1 act_scale 측정
```python
# quantize/calibration.py
def calibrate_model(model, data_loader, num_batches=10):
    """
    각 PoT 레이어의 activation 최대값을 측정하여 act_scale 설정
    
    중요: 이 함수는 QAT 전에 1회만 호출
    """
    model.eval()
    
    # Hook으로 각 레이어 출력 수집
    activation_max = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            if name not in activation_max:
                activation_max[name] = 0
            activation_max[name] = max(activation_max[name], output.abs().max().item())
        return hook
    
    # PoT 레이어에 hook 등록
    for name, module in model.named_modules():
        if isinstance(module, PoTLayerBase):
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    # 데이터 통과
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            _ = model(data)
    
    # Hook 제거
    for hook in hooks:
        hook.remove()
    
    # act_scale 설정
    for name, module in model.named_modules():
        if isinstance(module, PoTLayerBase) and name in activation_max:
            module.calibrate(activation_max[name])
            print(f"  {name}: act_max={activation_max[name]:.2f}, "
                  f"act_scale={module.act_scale:.4f}")
```

---

## 8. C 코드 생성

### 8.1 Combined Scale 계산
```python
# codegen/scale.py
def calculate_combined_scale(alpha, act_scale, prev_act_scale, std=None, is_first=False):
    """
    Combined scale 계산
    
    공식:
    - 일반: combined_scale = alpha * act_scale / prev_act_scale
    - 첫 레이어: combined_scale = alpha * act_scale / std  (std 흡수)
    """
    if is_first and std is not None:
        scale = alpha * act_scale / std
    else:
        scale = alpha * act_scale / prev_act_scale
    
    return scale
```

### 8.2 Scale 함수 생성 (shift+add 분해)
```python
def generate_scale_func(layer_name, scale, shift):
    """
    scale 값을 shift+add로 분해하여 C 함수 생성
    
    예: scale=21217, shift=18
    21217 = 2^0 + 2^5 + 2^6 + 2^7 + 2^9 + 2^12 + 2^14
    
    생성 코드:
    static inline int32_t scale_layer_0(int32_t x) {
        return ((x + (x<<5) + (x<<6) + (x<<7) + (x<<9) + (x<<12) + (x<<14)) 
                + (1<<17)) >> 18;  // rounding
    }
    """
    # scale을 2의 거듭제곱 합으로 분해
    shifts = []
    for i in range(20):
        if scale & (1 << i):
            shifts.append(i)
    
    # C 코드 생성
    terms = []
    for s in shifts:
        if s == 0:
            terms.append("x")
        else:
            terms.append(f"(x << {s})")
    
    expr = " + ".join(terms)
    rounding = f"(1 << {shift - 1})"
    
    code = f"""static inline int32_t scale_{layer_name}(int32_t x) {{
    return (({expr}) + {rounding}) >> {shift};
}}
"""
    return code
```

### 8.3 언롤 코드 생성
```python
# codegen/unroll.py
def generate_conv2d_unrolled(layer_info):
    """
    Conv2d 언롤 코드 생성
    
    핵심: weight가 코드에 내장됨 (메모리 접근 없음)
    
    예:
    acc += input[0] << 3;   // weight=8
    acc -= input[1] << 2;   // weight=-4
    acc += input[2];        // weight=1
    // weight=0인 경우 생략
    """
    name = layer_info['name']
    w_q = layer_info['weight_quantized']  # PoT 양자화된 weight
    bias = layer_info['bias_scaled']
    
    code = f"static void {name}_forward(const int8_t* input, int8_t* output) {{\n"
    
    for oc in range(out_channels):
        code += f"    {{ // output[{oc}]\n"
        code += f"        int32_t acc = 0;\n"
        
        for ic in range(in_channels):
            for ky in range(kh):
                for kx in range(kw):
                    w = int(w_q[oc, ic, ky, kx])
                    if w == 0:
                        continue  # 0은 생략 (핵심 최적화)
                    
                    idx = ...  # 입력 인덱스 계산
                    
                    if w > 0:
                        shift = int(math.log2(w))
                        if shift == 0:
                            code += f"        acc += input[{idx}];\n"
                        else:
                            code += f"        acc += input[{idx}] << {shift};\n"
                    else:
                        shift = int(math.log2(-w))
                        if shift == 0:
                            code += f"        acc -= input[{idx}];\n"
                        else:
                            code += f"        acc -= input[{idx}] << {shift};\n"
        
        # Scale 적용
        code += f"        acc = scale_{name}(acc);\n"
        
        # Bias 추가
        if bias[oc] != 0:
            code += f"        acc += {bias[oc]};\n"
        
        # ReLU + Clamp
        code += f"        if (acc < 0) acc = 0;\n"
        code += f"        output[{oc}] = (int8_t)(acc > 127 ? 127 : acc);\n"
        code += f"    }}\n"
    
    code += "}\n"
    return code
```

---

## 9. Export 흐름

### 9.1 전체 파이프라인
```python
# export.py
def export(model, output_path, config):
    """
    모델 → C 헤더 변환
    
    단계:
    1. 모델 그래프 분석 (어떤 레이어가 어떤 순서로)
    2. BatchNorm folding (있으면)
    3. 표준화 흡수 (첫 레이어 bias, scale 조정)
    4. 각 레이어 코드 생성
    5. 버퍼 할당 계산
    6. 헤더 파일 조립
    """
    
    # 1. PoT 레이어 추출 (순서대로)
    pot_layers = []
    for name, module in model.named_modules():
        if isinstance(module, PoTLayerBase):
            pot_layers.append((name, module))
    
    # 2. 표준화 흡수 (첫 레이어만)
    first_layer = pot_layers[0][1]
    absorb_standardization(first_layer, config.mean, config.std)
    
    # 3. 각 레이어 정보 수집
    layer_infos = []
    prev_act_scale = 1.0
    
    for i, (name, layer) in enumerate(pot_layers):
        info = {
            'name': f'layer_{i}',
            'type': type(layer).__name__,
            'weight': layer.weight,
            'bias': layer.bias,
            'alpha': layer.alpha.item(),
            'act_scale': layer.act_scale.item() if layer.act_scale else 1.0,
            'prev_act_scale': prev_act_scale,
            'is_first': (i == 0),
            'is_last': (i == len(pot_layers) - 1),
        }
        
        # Combined scale 계산
        info['combined_scale'] = calculate_combined_scale(
            info['alpha'], info['act_scale'], info['prev_act_scale'],
            config.std if info['is_first'] else None,
            info['is_first']
        )
        
        layer_infos.append(info)
        prev_act_scale = info['act_scale']
    
    # 4. C 코드 생성
    code = generate_header(layer_infos, config)
    
    # 5. 파일 저장
    with open(output_path, 'w') as f:
        f.write(code)
```

---

## 10. 표준화 흡수

### 10.1 수학적 근거
```
원래: y = Σ((x - mean) / std × W) + b
     = (1/std) × Σ((x - mean) × W) + b
     = (1/std) × (Σ(x × W) - mean × Σ(W)) + b
     = (1/std) × Σ(x × W) + (b - mean × Σ(W) / std)
     = (1/std) × Σ(x × W) + b'

변환 후:
- b' = b - mean × Σ(W)  (bias 조정)
- combined_scale = alpha × act_scale / std  (1/std 흡수)
```

### 10.2 구현
```python
def absorb_standardization(first_layer, mean, std):
    """
    표준화를 첫 레이어에 흡수
    
    - mean → bias 조정
    - 1/std → combined_scale에 흡수 (export 시)
    """
    with torch.no_grad():
        # Weight 합 계산
        weight_sum = first_layer.weight.sum(dim=(1, 2, 3))  # Conv의 경우
        # 또는 weight_sum = first_layer.weight.sum(dim=1)  # Linear의 경우
        
        # Bias 조정: b' = b - mean × Σ(W)
        if first_layer.bias is not None:
            first_layer.bias.data -= mean * weight_sum
```

---

## 11. 공개 API

### 11.1 __init__.py
```python
# potnn/__init__.py
from .modules.conv import PoTConv2d
from .modules.linear import PoTLinear
from .config import Config
from .export import export
from .quantize.calibration import calibrate_model

__all__ = [
    'PoTConv2d',
    'PoTLinear', 
    'Config',
    'export',
    'calibrate_model',
]
```

### 11.2 사용 예시
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import potnn

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = potnn.PoTConv2d(1, 8, 3, padding=1)
        self.conv2 = potnn.PoTConv2d(8, 16, 3, padding=1)
        self.fc = potnn.PoTLinear(16 * 4 * 4, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = self.fc(x)
        return x
    
    def alpha_reg_loss(self, lambda_reg=0.01):
        """전체 α regularization loss"""
        loss = 0
        for m in self.modules():
            if isinstance(m, potnn.PoTLayerBase):
                loss += m.alpha_reg_loss(lambda_reg)
        return loss
    
    def prepare_qat(self):
        """전체 모델 QAT 모드 활성화"""
        for m in self.modules():
            if isinstance(m, potnn.PoTLayerBase):
                m.prepare_qat()

# 학습 흐름
model = MNISTModel()

# 1. Float warmup (5~10 epochs)
train_float(model, train_loader, epochs=10)

# 2. Calibration
potnn.calibrate_model(model, train_loader)

# 3. QAT 준비
model.prepare_qat()

# 4. QAT 학습 (50~100 epochs)
for epoch in range(50):
    for data, target in train_loader:
        output = model(data)
        loss = F.cross_entropy(output, target) + model.alpha_reg_loss(0.01)
        loss.backward()
        optimizer.step()

# 5. Export
config = potnn.Config(
    flash=16384,
    ram=2048,
    mean=0.1307,
    std=0.3081
)
potnn.export(model, "model.h", config)
```

---

## 12. 테스트 요구사항

### 12.1 단위 테스트
```python
# test_pot.py
def test_quantize_to_pot():
    """PoT 양자화 정확성"""
    weight = torch.tensor([0.3, 0.9, 1.5, 3.2, 7.8])
    alpha = 0.5
    q = quantize_to_pot(weight, alpha, levels=11)
    # 예상: [1, 2, 4, 8, 16]
    
def test_ste_gradient():
    """STE gradient 통과 확인"""
    weight = nn.Parameter(torch.randn(10))
    alpha = 0.5
    q = quantize_to_pot_ste(weight, alpha, 11)
    loss = q.sum()
    loss.backward()
    assert weight.grad is not None

def test_combined_scale():
    """Combined scale 계산 정확성"""
    alpha = 0.5
    act_scale = 10.0
    prev_act_scale = 20.0
    std = 0.3
    
    # 첫 레이어
    scale = calculate_combined_scale(alpha, act_scale, 1.0, std, is_first=True)
    expected = alpha * act_scale / std
    assert abs(scale - expected) < 1e-6
```

### 12.2 통합 테스트
```python
# test_export.py
def test_export_mnist():
    """MNIST 모델 export 테스트"""
    model = MNISTModel()
    # ... 학습 ...
    
    potnn.export(model, "test.h", config)
    
    # C 코드 검증
    with open("test.h") as f:
        code = f.read()
    
    assert "static inline int32_t scale_" in code
    assert "input[" in code
    assert "<<" in code
    assert "float" not in code  # float 없어야 함
    assert "*" not in code or "/*" in code  # MUL 없어야 함 (주석 제외)
```

### 12.3 정확도 테스트
```python
# test_accuracy.py
def test_qat_accuracy():
    """QAT 정확도 검증: 손실 < 1.5%"""
    model = MNISTModel()
    
    # Float 학습
    float_acc = train_and_eval(model, qat=False)
    
    # Calibration + QAT
    potnn.calibrate_model(model, train_loader)
    model.prepare_qat()
    qat_acc = train_and_eval(model, qat=True)
    
    # 손실 검증
    assert float_acc - qat_acc < 1.5
```

---

## 13. 주의사항

### 13.1 절대 하지 말 것
- Float weight 허용 ❌
- MUL 명령어 생성 (scale 제외) ❌
- Softmax 지원 ❌
- int16 activation ❌
- /255 정규화 ❌

### 13.2 반드시 할 것
- α 스케일링 ✅
- Calibration 기반 고정 act_scale ✅
- 표준화 흡수 (mean→bias, /std→scale) ✅
- /256 정규화 ✅
- Rounding (shift 전 +0.5 효과) ✅

---

## 14. 기존 코드 참고

현재 potnn_fixed/ 디렉토리에 작동하는 코드가 있다.
- C 코드 생성 로직은 `codegen/` 참고
- 양자화 로직은 `quantize/` 참고
- 핵심 차이: `wrap()` 대신 직접 `PoTConv2d`, `PoTLinear` 사용하도록 변경

---

## 15. 완료 기준

1. `PoTConv2d`, `PoTLinear` 모듈이 독립적으로 동작
2. 사용자가 자유로운 모델 구조 정의 가능
3. `calibrate_model()`, `prepare_qat()`, `export()` API 제공
4. MNIST 16x16에서 QAT 정확도 > 93%
5. 생성된 C 코드에 float 연산 없음
6. 생성된 C 코드의 MUL은 레이어당 1회 (scale)만
