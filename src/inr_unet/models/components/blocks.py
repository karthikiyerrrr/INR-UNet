"""Convolutional building blocks for the UNet backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from inr_unet.registry import BLOCKS


@BLOCKS.register("double_conv")
class DoubleConv(nn.Module):
    """Two consecutive (conv -> norm -> activation) layers."""

    def __init__(
        self, in_channels: int, out_channels: int, *, norm: str = "batch", act: str = "relu"
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("DoubleConv.forward is not implemented yet.")


@BLOCKS.register("down")
class Down(nn.Module):
    """Downsampling step: spatial reduction followed by a DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Down.forward is not implemented yet.")


@BLOCKS.register("up")
class Up(nn.Module):
    """Upsampling step: upsample, concatenate the skip connection, then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, *, bilinear: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Up.forward is not implemented yet.")


@BLOCKS.register("res_block")
class ResBlock(nn.Module):
    """Residual convolutional block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("ResBlock.forward is not implemented yet.")
