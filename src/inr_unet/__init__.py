"""INR-UNet: a resolution-agnostic UNet for S/TEM image segmentation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("inr-unet")
except PackageNotFoundError:  # package not installed (e.g. source checkout without install)
    __version__ = "0.0.0"

from inr_unet import config, losses, metrics, registry  # noqa: F401
from inr_unet.config import load_config
from inr_unet.models import BaselineUNet, INRUNet  # noqa: F401  (triggers registration)
from inr_unet.registry import build_model

__all__ = [
    "__version__",
    "config",
    "registry",
    "losses",
    "metrics",
    "load_config",
    "build_model",
    "INRUNet",
    "BaselineUNet",
]
