# Notebooks

Marimo notebooks are committed as `.py` files and are always edited with the
`marimo-pair` workflow. Open one with `uv run marimo edit notebooks/<name>.py`.

Available:

- **TEMImageNet explorer** (`temimagenet_explorer.py`, marimo) — interactively render
  synthetic ADF-STEM images and their coordinate-derived labels from the forward model,
  sweeping the imaging condition, dose, background, scan noise, and the ADF weighting
  exponent. Columns are synthesized in-notebook; in the pipeline they come from upstream
  CIF / zone-axis projection.

Planned (deferred until the model API and run-artifact format exist):

- **Colab quickstart** (committed `.ipynb`) — boilerplate notebook showing how to install
  the package and run the models on Colab. Personal Drive paths are stripped from the
  committed copy. Jupyter notebooks are gitignored; this one is force-added.
- **Single-run explorer** (marimo) — explore one run's artifacts from `runs/` in depth.
- **Multi-run comparison** (marimo) — compare runs based on what is available in `runs/`.
