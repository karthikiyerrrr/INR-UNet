"""UNet encoder/backbone (AtomSegNet style) producing dense feature maps."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from inr_unet.registry import ENCODERS


@ENCODERS.register("unet")
class UNetEncoder(nn.Module):
    """UNet backbone mapping an input image to a dense feature map at input resolution."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``x`` of shape [B, C, H, W] to a feature map of shape [B, D, H, W]."""
        raise NotImplementedError("UNetEncoder.forward is not implemented yet.")
