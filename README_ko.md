# PoT-NN: 초저전력 MCU를 위한 곱셈 없는 신경망

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**PoT-NN**은 **곱셈 없이 딥러닝 추론이 가능한** 양자화 프레임워크입니다.  
하드웨어 곱셈기가 없는 초저가 MCU (CH32V003, PY32F003 등)에서도 신경망을 실행할 수 있습니다.

> 🇺🇸 [English Documentation](README.md)

## 🎯 핵심 특징

| 특징 | 설명 |
|------|------|
| **곱셈 제거** | 모든 가중치를 2의 거듭제곱으로 양자화, `<<`, `>>`, `+` 연산만 사용 |
| **정수 전용 추론** | 부동소수점 연산 없이 `int8`/`int32`만 사용 |
| **5가지 인코딩** | 정확도 vs 메모리 트레이드오프 선택 가능 |
| **C 코드 자동 생성** | 의존성 없는 단독 실행 가능한 C 헤더 파일 |
| **비트 정확 일치** | Python 시뮬레이션과 C 코드 출력 100% 동일 |

## 📦 설치

```bash
pip install potnn
```

## 🚀 빠른 시작

### 방법 1: 한 줄 학습 (권장)

```python
import torch
import torch.nn as nn
import potnn
from potnn import PoTConv2d, PoTLinear

# 1. PoT 레이어로 모델 정의
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = PoTConv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = PoTConv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)  # PoTGlobalAvgPool로 자동 교체됨
        self.fc = PoTLinear(16, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

model = TinyNet()

# 2. 설정
config = potnn.Config(
    flash=16384,      # 타겟 MCU Flash (bytes)
    ram=2048,         # 타겟 MCU RAM (bytes)
    mean=0.1307,      # 데이터셋 평균
    std=0.3081,       # 데이터셋 표준편차
    input_h=16, input_w=16, input_channels=1,
)

# 3. 학습 (Float → Calibrate → QAT → Integer Sim)
model = potnn.train(model, train_loader, test_loader, config,
                    float_epochs=15, qat_epochs=50)

# 4. C 코드 변환
potnn.export(model, "model.h", config)
```

### 방법 2: 수동 파이프라인

```python
import potnn

# 1단계: Float 학습 (일반 PyTorch 학습)
train_float(model, train_loader, epochs=15)

# 2단계: BatchNorm 퓨전 (있는 경우)
potnn.fuse_batchnorm(model)

# 3단계: Activation Scale 보정
potnn.calibrate(model, train_loader, config)

# 4단계: QAT 준비
potnn.prepare_qat(model, config)

# 5단계: QAT 학습
train_qat(model, train_loader, epochs=50)

# 6단계: 정수 시뮬레이션 활성화 (C 호환)
potnn.enable_integer_sim(model, input_std=config.std, input_mean=config.mean)

# 7단계: 변환
potnn.export(model, "model.h", config)
```

## � 인코딩 모드

정확도 vs 메모리 트레이드오프에 따라 선택:

| 인코딩 | 레벨 수 | 값 | bit/가중치 | 용도 |
|--------|---------|-----|-----------|------|
| `unroll` | 17 | 0, ±1, ±2, ±4, ..., ±128 | 코드 언롤 | 최고 정확도 |
| `fp130` | 16 | ±1, ±2, ±4, ..., ±128 | 4-bit | Dense 레이어 |
| `5level` | 5 | -8, -1, 0, +1, +8 | 4-bit (skip) | 균형 |
| `2bit` | 4 | -2, -1, +1, +2 | 2-bit | 최소 메모리 |
| `ternary` | 3 | -1, 0, +1 | 2-bit (RLE) | 희소 모델 |

### 레이어별 인코딩 지정

```python
config = potnn.Config(
    flash=16384, ram=2048,
    layer_encodings={
        'conv1': 'unroll',  # 첫 레이어: 최대 정확도
        'conv2': '5level',  # 중간 레이어
        'fc': 'unroll',     # 마지막 레이어: 최대 정확도
    },
    default_encoding='5level'
)
```

### 인코딩 상세

#### `unroll` (기본값)
- 가중치를 직접 shift-add 연산으로 언롤
- **Zero 가중치 생략** (희소 모델에 유리)
- 코드 크기 가장 큼, 정확도 가장 높음
```c
// 가중치 -8인 경우
acc -= input[i] << 3;  // -8 = -(1<<3)
```

#### `fp130` (FP1.3.0 포맷)
- 4-bit 패킹: `[sign(1)][exp(3)]`
- **Zero 없음** (0은 ±1로 교대 대체)
- Dense 레이어에 적합
```c
// 8개 가중치 → 1개 uint32
val = (1 << exp) * (sign ? -1 : 1);
```

#### `5level` (Skip 인코딩)
- 4-bit 패킹: `[skip(2)][sign(1)][mag(1)]`
- **Skip으로 연속 0 압축** (0~3개)
- ⚠️ **제약**: 4개 이상 연속 0 불가 (4번째부터 +1로 강제 대체)
```c
skip = (code >> 2) & 0x3;
i += skip;  // 0들 건너뛰기
val = (mag ? 8 : 1) * (sign ? -1 : 1);
```

#### `2bit`
- 2-bit 패킹: `[sign(1)][shift(1)]`
- **최소 메모리** (16개 가중치 → 1개 uint32)
- Zero 없음
```c
shifted = input[i] << (code & 1);  // ×1 or ×2
acc += (code & 2) ? -shifted : shifted;
```

#### `ternary` (Triple-Run)
- 2-bit 코드 + Run-Length 인코딩
- `11` 코드 = 이전 값 2번 더 반복
- 매우 희소한 모델용

## 📁 지원 레이어

| 레이어 | 클래스 | 비고 |
|--------|--------|------|
| Conv2D | `PoTConv2d` | 모든 표준 파라미터 지원 |
| Conv1D | `PoTConv1d` | 시계열용 |
| Depthwise | `PoTDepthwiseConv2d` | MobileNet 스타일 |
| Linear | `PoTLinear` | Fully Connected |
| GAP | 자동 교체 | `nn.AdaptiveAvgPool2d(1)` → `PoTGlobalAvgPool` |
| Add | `PoTAdd` | Residual 연결용 |
| BatchNorm | 자동 퓨전 | 이전 Conv/Linear에 흡수됨 |

## ⚙️ API 레퍼런스

### `potnn.Config`

| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| `flash` | int | ✅ | Flash 메모리 예산 (bytes) |
| `ram` | int | ✅ | RAM 예산 (bytes) |
| `mean` | float/list | ❌ | 데이터셋 평균 |
| `std` | float/list | ❌ | 데이터셋 표준편차 |
| `input_h`, `input_w` | int | ❌ | 입력 크기 (기본: 16×16) |
| `input_channels` | int | ❌ | 입력 채널 수 (기본: 1) |
| `layer_encodings` | dict | ❌ | 레이어별 인코딩 지정 |
| `default_encoding` | str | ❌ | 기본 인코딩 (기본: 'unroll') |

### 주요 함수

```python
potnn.train(model, train_loader, test_loader, config, ...)  # 전체 파이프라인
potnn.calibrate(model, data_loader, config)                  # Scale 보정
potnn.prepare_qat(model, config)                             # QAT 모드 활성화
potnn.enable_integer_sim(model, input_std, input_mean)       # C 호환 모드
potnn.export(model, output_path, config)                     # C 코드 생성
potnn.fuse_batchnorm(model)                                  # BN 퓨전
```

## 🧪 검증 결과

- **비트 정확 일치**: Python 정수 시뮬레이션 = C 출력 100%
- **MNIST**: 97%+ 정확도, 12KB 바이너리
- **100개 모델 스트레스 테스트**: 다양한 랜덤 아키텍처에서 검증 완료

## 📝 라이선스

**듀얼 라이선스**: GPL-3.0 + 상용 라이선스

| 사용 용도 | 라이선스 |
|-----------|----------|
| 오픈소스 프로젝트 | GPL-3.0 (무료) |
| 상용/비공개 프로젝트 | 상용 라이선스 (문의) |

자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.

## 🙏 기여하기

이 프로젝트는 고졸 1인 개발자가 만들었습니다.  
부족한 부분이 많고, 버그나 개선할 점이 있을 수 있습니다.

**어떤 기여든 진심으로 감사드립니다!**
- 🐛 버그 제보
- 💡 기능 제안
- 🔧 Pull Request
- 📖 문서 개선

이슈나 아이디어가 있으시면 언제든 Issue나 PR을 열어주세요. 감사합니다!

---

**Made with ❤️ for ultra-low-power AI**
