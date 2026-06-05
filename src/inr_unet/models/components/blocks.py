"""Convolutional building blocks for the UNet backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from inr_unet.registry import BLOCKS

_NORMS: dict[str, type[nn.Module]] = {
    "batch": nn.BatchNorm2d,
    "instance": nn.InstanceNorm2d,
}
_ACTS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
}


@BLOCKS.register("double_conv")
class DoubleConv(nn.Module):
    """Two consecutive (conv -> norm -> activation) layers; preserves H x W."""

    def __init__(
        self, in_channels: int, out_channels: int, *, norm: str = "batch", act: str = "relu"
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm
        self.act = act
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _NORMS[norm](out_channels),
            _ACTS[act](),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _NORMS[norm](out_channels),
            _ACTS[act](),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@BLOCKS.register("down")
class Down(nn.Module):
    """Downsampling step: 2x maxpool followed by a DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


@BLOCKS.register("up")
class Up(nn.Module):
    """Upsampling step: upsample, align + concat the skip, then DoubleConv.

    ``in_channels`` is the channel count of the deep feature being upsampled;
    ``skip_channels`` is the channel count of the skip connection it is concatenated with.
    """

    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int, *, bilinear: bool = True
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        if bilinear:
            self.up: nn.Module = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy = skip.shape[-2] - x.shape[-2]
        dx = skip.shape[-1] - x.shape[-1]
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


@BLOCKS.register("res_block")
class ResBlock(nn.Module):
    """Residual convolutional block; shape-preserving."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + x)
