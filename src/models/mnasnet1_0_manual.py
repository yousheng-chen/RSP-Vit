from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn


# The official TorchVision MnasNet uses TensorFlow-style BN momentum 0.9997,
# which maps to a very small PyTorch momentum of about 0.0003.
# That setting updates running statistics very slowly and can be unstable for
# this project's small-data training setup, so the manual variant uses 0.1.
_BN_MOMENTUM = 0.1


def _round_to_multiple_of(val: float, divisor: int, round_up_bias: float = 0.9) -> int:
    if not 0.0 < round_up_bias < 1.0:
        raise ValueError(f"round_up_bias should be in (0, 1), got {round_up_bias}")
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha: float) -> List[int]:
    base_depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in base_depths]


class MnasNetInvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expansion_factor: int,
        bn_momentum: float = _BN_MOMENTUM,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride should be 1 or 2, got {stride}")
        if kernel_size not in (3, 5):
            raise ValueError(f"kernel_size should be 3 or 5, got {kernel_size}")

        mid_channels = in_channels * expansion_factor
        self.use_residual = stride == 1 and in_channels == out_channels
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=mid_channels,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.layers(x)
        if self.use_residual:
            return out + x
        return out


def _make_stack(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    expansion_factor: int,
    repeats: int,
    bn_momentum: float,
) -> nn.Sequential:
    if repeats < 1:
        raise ValueError(f"repeats should be >= 1, got {repeats}")

    blocks = [
        MnasNetInvertedResidual(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            expansion_factor=expansion_factor,
            bn_momentum=bn_momentum,
        )
    ]
    for _ in range(1, repeats):
        blocks.append(
            MnasNetInvertedResidual(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                expansion_factor=expansion_factor,
                bn_momentum=bn_momentum,
            )
        )
    return nn.Sequential(*blocks)


class ManualMnasNet(nn.Module):
    """
    Manual reimplementation of TorchVision's MNASNet architecture.

    This keeps the official 1.0-width stage layout, block configuration, and
    initialization logic, while using a more practical BN momentum for this
    project's small-data experiments.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        num_classes: int = 1000,
        dropout: float = 0.2,
        img_channels: int = 3,
        bn_momentum: float = _BN_MOMENTUM,
    ) -> None:
        super().__init__()
        if alpha <= 0.0:
            raise ValueError(f"alpha should be > 0, got {alpha}")

        depths = _get_depths(alpha)
        self.features = nn.Sequential(
            nn.Conv2d(img_channels, depths[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(depths[0], momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(depths[0], depths[0], kernel_size=3, stride=1, padding=1, groups=depths[0], bias=False),
            nn.BatchNorm2d(depths[0], momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(depths[0], depths[1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(depths[1], momentum=bn_momentum),
            _make_stack(depths[1], depths[2], 3, 2, 3, 3, bn_momentum),
            _make_stack(depths[2], depths[3], 5, 2, 3, 3, bn_momentum),
            _make_stack(depths[3], depths[4], 5, 2, 6, 3, bn_momentum),
            _make_stack(depths[4], depths[5], 3, 1, 6, 2, bn_momentum),
            _make_stack(depths[5], depths[6], 5, 2, 6, 4, bn_momentum),
            _make_stack(depths[6], depths[7], 3, 1, 6, 1, bn_momentum),
            nn.Conv2d(depths[7], 1280, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1280, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1280, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, mode="fan_out", nonlinearity="sigmoid")
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = x.mean([2, 3])
        return self.classifier(x)


def build_mnasnet1_0_manual(
    num_classes: int,
    use_pretrained: bool = False,
    img_channels: int = 3,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Build a manual reimplementation of MnasNet1.0.

    Differences from the direct torchvision builder:
    - The network is constructed entirely from locally defined modules.
    - We keep the official 1.0 block layout and initialization logic.
    - BN momentum is tuned to 0.1 for more stable running statistics on this project's small-data setup.
    - Pretrained weights are intentionally not supported in this manual version.
    """
    if use_pretrained:
        raise ValueError(
            "Manual MnasNet1.0 does not support --use-pretrained. "
            "Use the official torchvision-backed mnasnet1_0 builder if you need pretrained weights."
        )

    return ManualMnasNet(
        alpha=1.0,
        num_classes=num_classes,
        dropout=dropout,
        img_channels=img_channels,
    )
