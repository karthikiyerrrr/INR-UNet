"""UNet encoder/backbone (AtomSegNet style) producing dense feature maps."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from inr_unet.models.components.blocks import DoubleConv, Down, Up
from inr_unet.registry import ENCODERS


@ENCODERS.register("unet")
class UNetEncoder(nn.Module):
    """UNet backbone mapping an input image to a dense feature map at input resolution.

    Channel schedule is ``base_channels * 2**k`` for k in [0, depth]. The input is
    reflect-padded up to a multiple of ``2**depth`` so any size works, then the output
    feature map is cropped back to the original H x W.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_channels = int(cfg.in_channels)
        base = int(cfg.base_channels)
        self.depth = int(cfg.depth)
        feature_dim = int(cfg.feature_dim)

        chans = [base * (2**k) for k in range(self.depth + 1)]
        self.inc = DoubleConv(in_channels, chans[0])
        self.downs = nn.ModuleList(Down(chans[k], chans[k + 1]) for k in range(self.depth))
        self.ups = nn.ModuleList(
            Up(chans[k + 1], chans[k], chans[k], bilinear=True)
            for k in reversed(range(self.depth))
        )
        self.head = nn.Conv2d(chans[0], feature_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        mult = 2**self.depth
        pad_h = (mult - h % mult) % mult
        pad_w = (mult - w % mult) % mult
        if pad_h or pad_w:
            x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")

        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        out = skips[-1]
        for i, up in enumerate(self.ups):
            out = up(out, skips[-(i + 2)])
        out = self.head(out)
        return out[..., :h, :w]
