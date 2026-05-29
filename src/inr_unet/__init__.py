"""INR-UNet: a resolution-agnostic UNet for S/TEM image segmentation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("inr-unet")
except PackageNotFoundError:  # package not installed (e.g. source checkout without install)
    __version__ = "0.0.0"

__all__ = ["__version__"]
