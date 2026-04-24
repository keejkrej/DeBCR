"""PyTorch DeBCR model.

The public API is channel-first and multi-input multi-output:
``model(x0, x2, x4) -> [z0, z2, z4]``.

For convenience, ``model(x0)`` derives the half- and quarter-resolution inputs
with area interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _check_image_tensor(name: str, x: Tensor) -> None:
    if torch.onnx.is_in_onnx_export():
        return
    if x.ndim != 4:
        raise ValueError(f"{name} must have shape (N, C, H, W), got {tuple(x.shape)}")
    if x.shape[-2] < 4 or x.shape[-1] < 4:
        raise ValueError(f"{name} spatial dimensions must be at least 4 pixels")


def _downsample(x: Tensor, factor: int) -> Tensor:
    return F.avg_pool2d(x, kernel_size=factor, stride=factor)


def _resize_like(x: Tensor, reference: Tensor) -> Tensor:
    return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)


class ResidualDenseBlock(nn.Module):
    """Compact residual dense block used by each DeBCR scale branch."""

    def __init__(self, channels: int, growth: int, layers: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv2d(channels + growth * i, growth, kernel_size=3, padding=1)
            for i in range(layers)
        )
        self.fuse = nn.Conv2d(channels + growth * layers, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        features = [x]
        for conv in self.convs:
            y = torch.cat(features, dim=1)
            features.append(F.relu(conv(y), inplace=True))
        return x + 0.2 * self.fuse(torch.cat(features, dim=1))


class ScaleBranch(nn.Module):
    """Feature extractor plus residual prediction head for one image scale."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int,
        blocks: int,
        growth: int,
        dense_layers: int,
    ) -> None:
        super().__init__()
        self.input = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[ResidualDenseBlock(width, growth, dense_layers) for _ in range(blocks)]
        )
        self.output = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, out_channels, kernel_size=3, padding=1),
        )

    def features(self, x: Tensor) -> Tensor:
        return self.body(F.relu(self.input(x), inplace=True))

    def predict(self, x: Tensor, features: Tensor) -> Tensor:
        return x + self.output(features)


@dataclass(frozen=True)
class DeBCRConfig:
    in_channels: int = 1
    out_channels: int = 1
    width: int = 32
    blocks: int = 4
    growth: int = 16
    dense_layers: int = 3


class DeBCR(nn.Module):
    """Multi-resolution DeBCR restoration model implemented in PyTorch."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 32,
        blocks: int = 4,
        growth: int = 16,
        dense_layers: int = 3,
    ) -> None:
        super().__init__()
        self.config = DeBCRConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            blocks=blocks,
            growth=growth,
            dense_layers=dense_layers,
        )
        self.branch0 = ScaleBranch(in_channels, out_channels, width, blocks, growth, dense_layers)
        self.branch2 = ScaleBranch(in_channels, out_channels, width, blocks, growth, dense_layers)
        self.branch4 = ScaleBranch(in_channels, out_channels, width, blocks, growth, dense_layers)

        self.fuse4_to_2 = nn.Sequential(
            nn.Conv2d(width * 2, width, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
        )
        self.fuse2_to_0 = nn.Sequential(
            nn.Conv2d(width * 2, width, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
        )

    def forward(self, x0: Tensor, x2: Tensor | None = None, x4: Tensor | None = None) -> list[Tensor]:
        _check_image_tensor("x0", x0)
        if x2 is None:
            x2 = _downsample(x0, 2)
        if x4 is None:
            x4 = _downsample(x0, 4)

        _check_image_tensor("x2", x2)
        _check_image_tensor("x4", x4)
        if not torch.onnx.is_in_onnx_export() and (
            x2.shape[:2] != x0.shape[:2] or x4.shape[:2] != x0.shape[:2]
        ):
            raise ValueError("x0, x2, and x4 must share batch and channel dimensions")

        f0 = self.branch0.features(x0)
        f2 = self.branch2.features(x2)
        f4 = self.branch4.features(x4)

        z4 = self.branch4.predict(x4, f4)

        f2 = f2 + self.fuse4_to_2(torch.cat([f2, _resize_like(f4, f2)], dim=1))
        z2 = self.branch2.predict(x2, f2)

        f0 = f0 + self.fuse2_to_0(torch.cat([f0, _resize_like(f2, f0)], dim=1))
        z0 = self.branch0.predict(x0, f0)

        return [z0, z2, z4]


def build_debcr(**kwargs: int) -> DeBCR:
    """Build a DeBCR model with keyword arguments matching :class:`DeBCR`."""

    return DeBCR(**kwargs)
