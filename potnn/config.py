"""Configuration module for potnn."""

from typing import Optional, Union, List, Dict


# 지원하는 인코딩 목록
VALID_ENCODINGS = {'unroll', 'fp130', '5level', '2bit', 'ternary'}


class Config:
    """Configuration for potnn model compilation.

    Args:
        flash: Flash memory budget in bytes (e.g., 16384 for 16KB)
        ram: RAM memory budget in bytes (e.g., 2048 for 2KB)
        input_norm: Input normalization method (255, 256, or 'standardize')
        mean: Mean for standardization. Can be float (1-channel) or List[float] (multi-channel)
        std: Standard deviation for standardization. Can be float or List[float]
        input_h: Input height (default 16 for 16x16 MNIST)
        input_w: Input width (default 16 for 16x16 MNIST)
        input_channels: Number of input channels (default 1 for grayscale, 3 for RGB)
        layer_encodings: Dict mapping layer names to encoding types.
            Example: {'conv1': 'unroll', 'fc': '5level'}
            Valid encodings: 'unroll', 'fp130', '5level', '2bit', 'ternary'
        default_encoding: Default encoding for layers not in layer_encodings.
            Default is 'unroll' for backward compatibility.
    """

    def __init__(
        self,
        flash: int,
        ram: int,
        input_norm: Optional[int] = 256,
        mean: Optional[Union[float, List[float]]] = None,
        std: Optional[Union[float, List[float]]] = None,
        input_h: int = 16,
        input_w: int = 16,
        input_channels: int = 1,
        layer_encodings: Optional[Dict[str, str]] = None,
        default_encoding: str = 'unroll',
    ):
        self.flash = flash
        self.ram = ram
        self.input_norm = input_norm
        self.input_h = input_h
        self.input_w = input_w
        self.input_channels = input_channels

        # Normalize mean/std to list format for consistency
        # 1-channel: mean=0.5 -> [0.5]
        # 3-channel: mean=[0.4914, 0.4822, 0.4465] -> as-is
        if mean is not None:
            if isinstance(mean, (int, float)):
                self.mean = [float(mean)]
            else:
                self.mean = [float(m) for m in mean]
        else:
            self.mean = None

        if std is not None:
            if isinstance(std, (int, float)):
                self.std = [float(std)]
            else:
                self.std = [float(s) for s in std]
        else:
            self.std = None

        # Validate input_norm
        if input_norm not in [255, 256, 'standardize', None]:
            raise ValueError(f"input_norm must be 255, 256, 'standardize', or None, got {input_norm}")

        if input_norm == 'standardize' and (self.mean is None or self.std is None):
            raise ValueError("mean and std must be provided when input_norm='standardize'")

        # Validate mean/std length matches input_channels
        if self.mean is not None and len(self.mean) != input_channels:
            raise ValueError(f"mean length ({len(self.mean)}) must match input_channels ({input_channels})")
        if self.std is not None and len(self.std) != input_channels:
            raise ValueError(f"std length ({len(self.std)}) must match input_channels ({input_channels})")

        # Store normalization type
        if self.mean is not None and self.std is not None:
            self.use_standardization = True
        else:
            self.use_standardization = False

        # Validate and store encoding settings
        if default_encoding not in VALID_ENCODINGS:
            raise ValueError(
                f"Invalid default_encoding '{default_encoding}'. "
                f"Valid options: {VALID_ENCODINGS}"
            )
        self.default_encoding = default_encoding

        self.layer_encodings = layer_encodings or {}
        for layer_name, encoding in self.layer_encodings.items():
            if encoding not in VALID_ENCODINGS:
                raise ValueError(
                    f"Invalid encoding '{encoding}' for layer '{layer_name}'. "
                    f"Valid options: {VALID_ENCODINGS}"
                )

    def get_encoding(self, layer_name: str) -> str:
        """Get encoding for a specific layer.
        
        Args:
            layer_name: Name of the layer (e.g., 'conv1', 'features.0')
        
        Returns:
            Encoding type ('unroll', 'fp130', '5level', '2bit', 'ternary')
        """
        return self.layer_encodings.get(layer_name, self.default_encoding)