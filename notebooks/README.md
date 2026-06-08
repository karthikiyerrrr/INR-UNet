# Notebooks

Marimo notebooks are committed as `.py` files and are always edited with the
`marimo-pair` workflow. Open one with `uv run marimo edit notebooks/<name>.py`.

Available:

- **TEMImageNet explorer** (`temimagenet_explorer.py`, marimo) — interactively render
  synthetic ADF-STEM images and their coordinate-derived labels from the forward model,
  sweeping the imaging condition, dose, background, scan noise, and the ADF weighting
  exponent. Columns are synthesized in-notebook; in the pipeline they come from upstream
  CIF / zone-axis projection.
- **Single-run explorer** (`run_explorer.py`, marimo) — explore one run's artifacts from
  `runs/`: the run metadata and config, the overfit loss curve (loss/bce/dice), the
  predicted-vs-gt image, and the profiling table with per-sweep throughput. Run artifacts
  are produced by the Colab smoke+profile notebook and pulled into `runs/`.
- **Run comparison** (`run_comparison.py`, marimo) — overlay any subset of training runs
  from `runs/`: a side-by-side summary table, validation F1 and localization-offset curves,
  and the held-out test metrics (detection rates and offsets). Built to weigh the INR-UNet
  against the fixed-resolution UNet baseline, whose runs share a byte-identical data pipeline
  and differ only in `model.name`.

Planned (deferred until the model API and run-artifact format exist):

- **Colab quickstart** (committed `.ipynb`) — boilerplate notebook showing how to install
  the package and run the models on Colab. Personal Drive paths are stripped from the
  committed copy. Jupyter notebooks are gitignored; this one is force-added.
