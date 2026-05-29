# INR-UNet

A resolution-agnostic UNet for scanning/transmission electron microscopy (S/TEM) image
segmentation. INR-UNet pairs a UNet feature encoder (after AtomSegNet) with a LIIF-style
continuous local implicit decoder, so the model can segment atomic-resolution images at
arbitrary input and output resolutions.

> **Status:** Early development. Interfaces are being scaffolded; model implementation is in progress.

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) with Python 3.11.

```bash
uv sync                 # install runtime dependencies
uv sync --group dev     # also install dev/exploration tooling (pytest, ruff, marimo, plotly)
```

## Repository structure

```text
src/inr_unet/
├── data/                 # synthetic data generation (TEMImageNet) + datasets
├── models/
│   ├── components/       # UNet encoder, LIIF decoder, conv blocks
│   └── inr_unet.py       # assembled INRUNet + baseline UNet
├── config.py             # OmegaConf configuration schemas
├── registry.py           # model registry + factory
├── losses.py             # segmentation losses
└── metrics.py            # segmentation metrics
configs/                  # example experiment configs
tests/                    # smoke / interface-contract tests
notebooks/                # Colab + marimo notebooks (see notebooks/README.md)
```

## References

- M. Lin et al., "TEMImageNet training library and AtomSegNet deep-learning models for
  high-precision atom segmentation, localization, denoising, and deblurring of
  atomic-resolution images," *Scientific Reports*, 2021.
- Y. Chen, S. Liu, X. Wang, "Learning Continuous Image Representation with Local Implicit
  Image Function," *CVPR*, 2021.

## License

MIT — see [LICENSE](LICENSE).
