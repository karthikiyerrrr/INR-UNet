"""INR-UNet model assembly. Importing this package registers all components and models."""

from inr_unet.models.components import blocks, liif, unet  # noqa: F401  (triggers registration)
from inr_unet.models.inr_unet import BaselineUNet, INRUNet

__all__ = ["INRUNet", "BaselineUNet"]
