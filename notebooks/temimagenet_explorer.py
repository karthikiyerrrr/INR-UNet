import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import pathlib
    from dataclasses import replace

    import marimo as mo
    import numpy as np
    import torch
    import plotly.graph_objects as go
    from PIL import Image
    from plotly.subplots import make_subplots
    from omegaconf import OmegaConf

    from inr_unet.config import ExperimentConfig, OccupancyConfig, SamplerConfig
    from inr_unet.data import (
        IMAGING_CONDITIONS,
        AugmentationSampler,
        CIFProvider,
        ColumnList,
        FiniteSupportProvider,
        LIIFSegDataset,
        RenderParams,
        STEMSegDataset,
        SyntheticLatticeProvider,
        SyntheticRenderSource,
        TEMRenderer,
        project_structure,
    )
    from inr_unet.data.generation.calibration import intensity_histogram, radial_power_spectrum
    from inr_unet.data.generation.labels import column_radius
    from inr_unet.data.generation.psf import wavelength_A
    from inr_unet.data.generation.structures import BackgroundSpec, ImagingCondition, NoiseSpec
    from inr_unet.data.occupancy import support_mask

    return (
        AugmentationSampler,
        BackgroundSpec,
        CIFProvider,
        ColumnList,
        ExperimentConfig,
        FiniteSupportProvider,
        IMAGING_CONDITIONS,
        Image,
        LIIFSegDataset,
        NoiseSpec,
        OccupancyConfig,
        OmegaConf,
        RenderParams,
        STEMSegDataset,
        SamplerConfig,
        SyntheticLatticeProvider,
        TEMRenderer,
        column_radius,
        go,
        intensity_histogram,
        make_subplots,
        mo,
        np,
        pathlib,
        radial_power_spectrum,
        replace,
        support_mask,
        torch,
        wavelength_A,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # TEMImageNet Forward-Model Explorer

    Render synthetic ADF-STEM images and their coordinate-derived labels from a
    synthetic atomic-column lattice, and explore how the imaging condition, dose,
    background, scan noise, and the ADF weighting exponent shape the result.

    The columns are generated in-notebook for exploration; in the training
    pipeline they come from upstream CIF / zone-axis projection.

    *Knob guide:* **Z** and the **exponent n** set ADF Z-contrast, which is
    *relative* (`weight = count * Z**n`) - with a single species every column
    scales identically and the normalized image is unchanged, so the
    **two-species checkerboard** is on by default to make them visible.
    **n_peak** is the electron dose: lower = grainier shot noise.
    **scan freq** drives the row-jitter distortion.
    """)
    return


@app.cell
def _(ColumnList, torch):
    def make_lattice(fov_A, spacing_A, z_a, z_b, alternate):
        """Square lattice of columns spanning the FOV; optionally a two-species checkerboard."""
        coords = torch.arange(spacing_A / 2, fov_A, spacing_A)
        xx, yy = torch.meshgrid(coords, coords, indexing="ij")
        pos = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        n = pos.shape[0]
        if alternate and n > 0:
            ix = torch.arange(coords.numel())
            gi, gj = torch.meshgrid(ix, ix, indexing="ij")
            parity = ((gi + gj) % 2).reshape(-1).bool()
            z = torch.where(parity, torch.full((n,), float(z_a)), torch.full((n,), float(z_b)))
        else:
            z = torch.full((n,), float(z_a))
        return ColumnList(positions_A=pos, z=z, count=torch.ones(n), fov_A=fov_A)

    return (make_lattice,)


@app.cell(hide_code=True)
def _(IMAGING_CONDITIONS, mo):
    condition = mo.ui.dropdown(options=list(IMAGING_CONDITIONS), value="cond1", label="condition")
    output_size = mo.ui.slider(32, 256, value=128, step=32, label="output_size (px)")
    pixel_size_A = mo.ui.slider(0.05, 0.40, value=0.10, step=0.01, label="pixel_size (A/px)")
    spacing_A = mo.ui.slider(1.0, 5.0, value=2.5, step=0.1, label="lattice spacing (A)")
    z_a = mo.ui.slider(6, 92, value=78, step=1, label="Z (species A)")
    z_b = mo.ui.slider(6, 92, value=22, step=1, label="Z (species B)")
    alternate = mo.ui.checkbox(value=True, label="two-species checkerboard")
    z_exponent = mo.ui.slider(1.0, 2.5, value=1.7, step=0.1, label="Z exponent n")
    background = mo.ui.dropdown(options=["constant", "linear_ramp", "nonlinear"], value="nonlinear", label="background")
    n_peak = mo.ui.slider(30, 3000, value=100, step=10, label="n_peak (e-/px)")
    scan_freq = mo.ui.slider(0.0, 0.5, value=0.1, step=0.01, label="scan freq (cyc/row)")
    rotation_deg = mo.ui.slider(0, 90, value=0, step=15, label="rotation (deg)")
    seed = mo.ui.slider(0, 50, value=0, step=1, label="seed")

    mo.vstack(
        [
            mo.hstack([condition, output_size, pixel_size_A], justify="start", gap=1),
            mo.hstack([spacing_A, z_a, z_b, alternate], justify="start", gap=1),
            mo.hstack([z_exponent, background, n_peak], justify="start", gap=1),
            mo.hstack([scan_freq, rotation_deg, seed], justify="start", gap=1),
        ]
    )
    return (
        alternate,
        background,
        condition,
        n_peak,
        output_size,
        pixel_size_A,
        rotation_deg,
        scan_freq,
        seed,
        spacing_A,
        z_a,
        z_b,
        z_exponent,
    )


@app.cell(hide_code=True)
def _(
    BackgroundSpec,
    IMAGING_CONDITIONS,
    NoiseSpec,
    OmegaConf,
    RenderParams,
    TEMRenderer,
    alternate,
    background,
    condition,
    make_lattice,
    n_peak,
    output_size,
    pixel_size_A,
    rotation_deg,
    scan_freq,
    seed,
    spacing_A,
    wavelength_A,
    z_a,
    z_b,
    z_exponent,
):
    renderer = TEMRenderer(
        OmegaConf.create({"potential_backend": "z_power", "sigma_potential_A": 0.4, "aperture_soft": True})
    )
    cond_obj = IMAGING_CONDITIONS[condition.value]
    fov = output_size.value * pixel_size_A.value
    cols = make_lattice(fov, spacing_A.value, z_a.value, z_b.value, alternate.value)
    params = RenderParams(
        output_size=output_size.value,
        pixel_size_A=pixel_size_A.value,
        rotation_deg=float(rotation_deg.value),
        z_exponent=float(z_exponent.value),
        background=BackgroundSpec(kind=background.value, params={}),
        noise=NoiseSpec(n_peak=float(n_peak.value), scan_freq_cyc_per_row=float(scan_freq.value)),
        seed=int(seed.value),
    )

    k_cut = (cond_obj.alpha_max_mrad * 1e-3) / wavelength_A(cond_obj.energy_keV)
    k_nyq = 1.0 / (2.0 * pixel_size_A.value)
    render_error = None
    out = None
    if k_cut >= k_nyq:
        render_error = (
            f"Aliasing: k_cut={k_cut:.3f} >= k_Nyq={k_nyq:.3f} /A. "
            f"Lower pixel_size below {1.0 / (2.0 * k_cut):.3f} A for this condition."
        )
    else:
        out = renderer.render(cols, cond_obj, params)
    return cols, cond_obj, fov, out, params, render_error


@app.cell(hide_code=True)
def _(
    cols,
    column_radius,
    cond_obj,
    condition,
    fov,
    mo,
    output_size,
    pixel_size_A,
    wavelength_A,
    z_exponent,
):
    _lam = wavelength_A(cond_obj.energy_keV)
    _r = column_radius(cond_obj)
    _knyq = 1.0 / (2.0 * pixel_size_A.value)
    _kcut = (cond_obj.alpha_max_mrad * 1e-3) / _lam
    _safe = _kcut < _knyq

    _spec = mo.md(
        f"**`{condition.value}`** &nbsp; {cond_obj.energy_keV:.0f} keV"
        f" &nbsp;&middot;&nbsp; convergence &alpha; = {cond_obj.alpha_max_mrad} mrad"
        f" &nbsp;&middot;&nbsp; source size {cond_obj.source_size_A} &#8491;"
        f" &nbsp;&middot;&nbsp; &lambda; = {_lam:.4f} &#8491;"
    )

    _tiles = mo.hstack(
        [
            mo.stat(
                f"{fov:.1f} \u00c5",
                label="Field of view",
                caption=f"{output_size.value} px \u00d7 {pixel_size_A.value:.3f} \u00c5/px",
                bordered=True,
            ),
            mo.stat(
                f"{cols.positions_A.shape[0]}",
                label="Columns",
                caption=f"Z-weighted, n = {z_exponent.value:.1f}",
                bordered=True,
            ),
            mo.stat(
                f"{_r:.3f} \u00c5",
                label="Column radius r",
                caption=f"2r = {2 * _r:.3f} \u00c5",
                bordered=True,
            ),
            mo.stat(
                f"{_kcut / _knyq:.2f}",
                label="k_cut / k_Nyq",
                caption="\u2713 Nyquist-safe" if _safe else "\u26a0 aliasing",
                bordered=True,
            ),
        ],
        justify="start",
        gap=1,
        wrap=True,
    )

    mo.vstack([_spec, _tiles], gap=0.75)
    return


@app.cell(hide_code=True)
def _(go, make_subplots, mo, out, params, render_error):
    if out is None:
        plot = mo.callout(mo.md(f"### {render_error}"), kind="warn")
    else:
        panels = [
            ("noisy image", out.image, "gray"),
            ("no noise (signal + background)", out.no_noise, "gray"),
            ("clean (sigma (x) PSF)", out.no_background_no_noise, "gray"),
            ("gaussian mask", out.gaussian_mask, "viridis"),
            ("circular mask", out.circular_mask, "viridis"),
        ]
        titles = [p[0] for p in panels] + ["clean + column positions"]
        fig = make_subplots(rows=2, cols=3, subplot_titles=titles, horizontal_spacing=0.04, vertical_spacing=0.10)
        grid = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
        for (title, arr, cs), (rr, cc) in zip(panels, grid):
            fig.add_trace(go.Heatmap(z=arr.numpy(), colorscale=cs, showscale=False), row=rr, col=cc)
        fig.add_trace(go.Heatmap(z=out.no_background_no_noise.numpy(), colorscale="gray", showscale=False), row=2, col=3)
        px = out.positions_A.numpy() / params.pixel_size_A
        fig.add_trace(
            go.Scatter(x=px[:, 0], y=px[:, 1], mode="markers", marker=dict(size=4, color="red"), showlegend=False),
            row=2,
            col=3,
        )
        fig.update_xaxes(showticklabels=False, ticks="")
        fig.update_yaxes(showticklabels=False, ticks="", autorange="reversed")
        fig.update_layout(height=560, width=860, margin=dict(l=8, r=8, t=28, b=8))
        plot = fig
    plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Augmentation sampler — draw diversity

    The `AugmentationSampler` turns the pure renderer into a dataset generator: from a
    single structure it draws an imaging condition and a full `RenderParams` (FOV,
    pixel size, rotation, background, dose, position offset) **deterministically per
    draw index**, across the paper's Table-1 grid. Scrub the master seed and draw
    count to see the augmentation diversity one structure yields.

    `max_fov_A` is the structure's physical extent; the sampler only picks render
    FOVs that fit inside it (with rotation headroom), so every draw renders. Each
    panel is titled with the condition, render FOV, rotation, and background it drew.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    samp_seed = mo.ui.slider(0, 50, value=0, step=1, label="master seed")
    n_draws = mo.ui.slider(2, 9, value=6, step=1, label="draw count")
    samp_max_fov = mo.ui.slider(20, 80, value=60, step=5, label="structure FOV / max_fov_A (A)")
    samp_spacing = mo.ui.slider(1.0, 5.0, value=2.5, step=0.1, label="lattice spacing (A)")

    mo.hstack([samp_seed, n_draws, samp_max_fov, samp_spacing], justify="start", gap=1)
    return n_draws, samp_max_fov, samp_seed, samp_spacing


@app.cell
def _(
    AugmentationSampler,
    OmegaConf,
    SamplerConfig,
    TEMRenderer,
    make_lattice,
    n_draws,
    samp_max_fov,
    samp_seed,
    samp_spacing,
):
    samp_renderer = TEMRenderer(
        OmegaConf.create(
            {"potential_backend": "z_power", "sigma_potential_A": 0.4, "aperture_soft": True}
        )
    )
    sampler = AugmentationSampler(
        OmegaConf.structured(SamplerConfig), master_seed=int(samp_seed.value)
    )
    samp_cols = make_lattice(float(samp_max_fov.value), samp_spacing.value, 78, 22, True)

    draws = []
    for _i in range(int(n_draws.value)):
        _cond, _p = sampler.sample(_i, max_fov_A=float(samp_max_fov.value))
        _out = samp_renderer.render(samp_cols, _cond, _p)
        draws.append((_i, _cond, _p, _out))
    return (draws,)


@app.cell(hide_code=True)
def _(draws, go, make_subplots):
    _ncols = 3
    _nrows = (len(draws) + _ncols - 1) // _ncols
    _bg_abbr = {"constant": "const", "linear_ramp": "ramp", "nonlinear": "nonlin"}
    _titles = [
        f"#{_i} {_c.name} · {_p.output_size * _p.pixel_size_A:.0f}Å · "
        f"{_p.rotation_deg:.0f}° · {_bg_abbr.get(_p.background.kind, _p.background.kind)}"
        for (_i, _c, _p, _o) in draws
    ]
    samp_fig = make_subplots(
        rows=_nrows, cols=_ncols, subplot_titles=_titles,
        horizontal_spacing=0.03, vertical_spacing=0.12,
    )
    for _k, (_i, _c, _p, _o) in enumerate(draws):
        samp_fig.add_trace(
            go.Heatmap(z=_o.image.numpy(), colorscale="gray", showscale=False),
            row=_k // _ncols + 1, col=_k % _ncols + 1,
        )
    samp_fig.update_xaxes(showticklabels=False, ticks="")
    samp_fig.update_yaxes(showticklabels=False, ticks="", autorange="reversed")
    samp_fig.update_annotations(font_size=12)
    samp_fig.update_layout(height=260 * _nrows, width=860, margin=dict(l=8, r=8, t=34, b=8))
    samp_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Datasets — render source -> trainable tensors

    `SyntheticRenderSource` maps a flat index to a fixed `crop_size`x`crop_size` input
    (deterministically cropped when the render is larger, reflect-padded when smaller),
    together with the **analytic** column coordinates inside that crop. Two datasets
    format it for training:

    - **`STEMSegDataset`** -> `(image[1,S,S], mask[1,S,S])` -- the label field rasterized
      onto the input grid, for the baseline fixed-resolution UNet.
    - **`LIIFSegDataset`** -> `(image[1,S,S], coords[Q,2], cell[Q,2], gt[Q,1])` -- `Q`
      query points with **continuous** `(x, y)` coords (normalized to [-1, 1] over the
      crop), a physical target pixel size `cell` (A/px), and the label sampled
      analytically at those exact coordinates. No rasterization -- supervision is
      resolution-free.

    The label is a **sharp center-marker** (gaussian FWHM 0.2 A, ~1-2 px) -- deliberately
    far narrower than the PSF-broadened, often background-dominated column in the input.
    Below, the input row carries red crosses at each labeled column center so the
    label<->image correspondence is visible (the input and its label share the exact same
    extent and pixel scale). Everything is reproducible from `master_seed + idx`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    ds_seed = mo.ui.slider(0, 20, value=0, step=1, label="master seed")
    ds_label = mo.ui.dropdown(options=["gaussian", "circular"], value="gaussian", label="label field")
    ds_q = mo.ui.slider(256, 4096, value=1024, step=256, label="LIIF query points Q")

    mo.hstack([ds_seed, ds_label, ds_q], justify="start", gap=1)
    return ds_label, ds_q, ds_seed


@app.cell
def _(
    ExperimentConfig,
    LIIFSegDataset,
    OmegaConf,
    STEMSegDataset,
    ds_label,
    ds_q,
    ds_seed,
    mo,
):
    ds_cfg = OmegaConf.structured(ExperimentConfig)
    ds_cfg.data.synthetic.n_scenes = 4
    ds_cfg.data.synthetic.draws_per_scene = 8
    ds_cfg.data.synthetic.master_seed = int(ds_seed.value)
    ds_cfg.data.synthetic.label_kind = ds_label.value
    ds_cfg.data.synthetic.sample_q = int(ds_q.value)

    stem_ds = STEMSegDataset(ds_cfg)
    liif_ds = LIIFSegDataset(ds_cfg)
    ds_S = stem_ds.source.crop_size

    # Split indices by branch: the render filled the crop (crop) vs needed reflect-pad.
    ds_crop_idx, ds_pad_idx = [], []
    for _i in range(len(stem_ds)):
        _s = stem_ds.source.get(_i)
        if _s.valid_extent_A < ds_S * _s.input_pixel_size_A - 1e-9:
            ds_pad_idx.append(_i)
        else:
            ds_crop_idx.append(_i)

    mo.md(
        f"`{len(stem_ds)}` samples &middot; **{len(ds_crop_idx)}** crop, "
        f"**{len(ds_pad_idx)}** reflect-pad &middot; input {ds_S}x{ds_S} &middot; "
        f"Q = {ds_cfg.data.synthetic.sample_q}"
    )
    return ds_S, ds_crop_idx, ds_pad_idx, liif_ds, stem_ds


@app.cell(hide_code=True)
def _(ds_S, ds_crop_idx, ds_label, ds_pad_idx, go, make_subplots, stem_ds):
    _show = ds_crop_idx[:3] + ds_pad_idx[:1]
    _titles = [f"#{_i} input + centers" for _i in _show] + [f"#{_i} {ds_label.value} label" for _i in _show]
    stem_fig = make_subplots(
        rows=2, cols=len(_show), subplot_titles=_titles,
        horizontal_spacing=0.02, vertical_spacing=0.12,
    )
    for _c, _i in enumerate(_show):
        _img, _mask = stem_ds[_i]
        _s = stem_ds.source.get(_i)
        _ppx = _s.positions_A.numpy() / _s.input_pixel_size_A
        _in = (_ppx[:, 0] >= 0) & (_ppx[:, 0] < ds_S) & (_ppx[:, 1] >= 0) & (_ppx[:, 1] < ds_S)
        stem_fig.add_trace(go.Heatmap(z=_img[0].numpy(), colorscale="gray", showscale=False), row=1, col=_c + 1)
        stem_fig.add_trace(
            go.Scatter(
                x=_ppx[_in, 0], y=_ppx[_in, 1], mode="markers",
                marker=dict(symbol="cross-thin", size=7, color="#ff3b3b", line=dict(width=1.3, color="#ff3b3b")),
                showlegend=False,
            ),
            row=1, col=_c + 1,
        )
        stem_fig.add_trace(
            go.Heatmap(z=_mask[0].numpy(), colorscale="viridis", zmin=0, zmax=1, showscale=False),
            row=2, col=_c + 1,
        )
    _n = 2 * len(_show)
    for _k in range(1, _n + 1):
        _xa = "x" if _k == 1 else f"x{_k}"
        _yk = "yaxis" if _k == 1 else f"yaxis{_k}"
        _xk = "xaxis" if _k == 1 else f"xaxis{_k}"
        stem_fig.layout[_yk].update(scaleanchor=_xa, constrain="domain", range=[ds_S, 0], showticklabels=False, ticks="")
        stem_fig.layout[_xk].update(constrain="domain", range=[0, ds_S], showticklabels=False, ticks="")
    stem_fig.update_annotations(font_size=12)
    stem_fig.update_layout(height=520, width=860, margin=dict(l=8, r=8, t=30, b=8))
    stem_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### LIIF supervision -- continuous queries over the crop

    Each panel overlays the `Q` query points on its input image, colored by the analytic
    `gt` at that point (viridis, 0 -> 1). Titles show the **input** pixel size (the
    encoder's resolution) and the per-item **cell** (the decode target resolution) -- both
    vary across items, which is exactly the continuous-resolution signal the LIIF head
    learns from. In **reflect-pad** items the queries stay inside the valid (non-padded)
    region, so the padded border carries no supervision.
    """)
    return


@app.cell(hide_code=True)
def _(ds_S, ds_crop_idx, ds_pad_idx, go, liif_ds, make_subplots):
    _show = ds_crop_idx[:2] + ds_pad_idx[:1]
    _items = []
    for _i in _show:
        _im, _coords, _cell, _gt = liif_ds[_i]
        _s = liif_ds.source.get(_i)
        _pad = _s.valid_extent_A < ds_S * _s.input_pixel_size_A - 1e-9
        _items.append((_i, _im, _coords, _cell, _gt, _s, _pad))
    _titles = [
        f"#{_i} · {'pad' if _pad else 'crop'} · "
        f"in {_s.input_pixel_size_A:.2f} -> cell {float(_cell[0, 0]):.2f} A/px"
        for (_i, _im, _coords, _cell, _gt, _s, _pad) in _items
    ]
    liif_fig = make_subplots(rows=1, cols=len(_items), subplot_titles=_titles, horizontal_spacing=0.06)
    for _c, (_i, _im, _coords, _cell, _gt, _s, _pad) in enumerate(_items):
        _qpx = (_coords.numpy() + 1.0) / 2.0 * ds_S
        _last = _c == len(_items) - 1
        liif_fig.add_trace(go.Heatmap(z=_im[0].numpy(), colorscale="gray", showscale=False), row=1, col=_c + 1)
        liif_fig.add_trace(
            go.Scatter(
                x=_qpx[:, 0], y=_qpx[:, 1], mode="markers",
                marker=dict(
                    size=4, color=_gt[:, 0].numpy(), colorscale="viridis", cmin=0, cmax=1,
                    showscale=_last, colorbar=dict(title="gt", thickness=12, len=0.9),
                ),
                showlegend=False,
            ),
            row=1, col=_c + 1,
        )
    _n = len(_items)
    for _k in range(1, _n + 1):
        _xa = "x" if _k == 1 else f"x{_k}"
        _yk = "yaxis" if _k == 1 else f"yaxis{_k}"
        _xk = "xaxis" if _k == 1 else f"xaxis{_k}"
        liif_fig.layout[_yk].update(scaleanchor=_xa, constrain="domain", range=[ds_S, 0], showticklabels=False, ticks="")
        liif_fig.layout[_xk].update(constrain="domain", range=[0, ds_S], showticklabels=False, ticks="")
    liif_fig.update_annotations(font_size=11)
    liif_fig.update_layout(height=360, width=900, margin=dict(l=8, r=8, t=34, b=8))
    liif_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CIF ingestion + crystallographic occupancy

    Real crystal structures (bundled CIFs) projected along a **zone axis** become a
    `ColumnList`: atoms stacking along the beam collapse into 2D columns, with an
    **effective Z** so `count * Z**n` matches the true ADF power-sum `sum(Z_i**n)`
    (pure columns keep their real atomic number). The projection also exports the
    **in-plane lattice basis** -- the two repeat vectors faceted shapes align to.

    Occupancy then clips a frame-filling lattice to a **finite particle**:
    `facet_polygon` (convex intersection of half-planes normal to low-index lattice
    directions -- facet-plausible, *not* a Wulff shape), `blob` (isotropic disk with
    smooth roughness), and a single-facet half-plane = a straight crystal edge.
    Color = atomic number Z.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    cif_seed = mo.ui.slider(0, 20, value=0, step=1, label="master seed")
    cif_support = mo.ui.slider(0.25, 0.85, value=0.5, step=0.05, label="support size (frac of FOV)")
    cif_occ_idx = mo.ui.slider(0, 9, value=4, step=1, label="occupancy demo scene")
    mo.hstack([cif_seed, cif_support, cif_occ_idx], justify="start", gap=2)
    return cif_occ_idx, cif_seed, cif_support


@app.cell
def _(CIFProvider, ExperimentConfig, OmegaConf, cif_seed, mo):
    cif_cfg = OmegaConf.structured(ExperimentConfig)
    cif_prov = CIFProvider(
        manifest_path=str(cif_cfg.data.cif.manifest_path),
        n_scenes=10,
        master_seed=int(cif_seed.value),
        n_exponent=float(cif_cfg.generation.sampler.z_exponent),
        group_tol_A=float(cif_cfg.data.cif.group_tol_A),
        scene_fov_A=(float(cif_cfg.data.cif.scene_fov_A_min), float(cif_cfg.data.cif.scene_fov_A_max)),
        rotation_jitter_deg=float(cif_cfg.data.cif.rotation_jitter_deg),
    )
    import yaml as _yaml
    import pathlib as _pl
    _entries = _yaml.safe_load(_pl.Path(cif_cfg.data.cif.manifest_path).read_text())["entries"]
    cif_labels = [
        "{}[{}]".format(e["cif"].replace(".cif", ""), "".join(str(x) for x in e["zone_axis"]))
        for e in _entries
    ]
    cif_scenes = [cif_prov.get(_i) for _i in range(10)]
    mo.md(
        "Projected **{}** bundled structures &middot; {}-{} columns/scene &middot; "
        "effective-Z reduction with n = {}".format(
            len(cif_scenes),
            min(s.positions_A.shape[0] for s in cif_scenes),
            max(s.positions_A.shape[0] for s in cif_scenes),
            cif_cfg.generation.sampler.z_exponent,
        )
    )
    return cif_labels, cif_scenes


@app.cell(hide_code=True)
def _(cif_labels, cif_scenes, go, make_subplots, np):
    cif_gallery = make_subplots(
        rows=2, cols=5, subplot_titles=cif_labels,
        horizontal_spacing=0.02, vertical_spacing=0.10,
    )
    for _k, _sc in enumerate(cif_scenes):
        _row, _col = _k // 5 + 1, _k % 5 + 1
        _p = _sc.positions_A.numpy()
        cif_gallery.add_trace(
            go.Scatter(
                x=_p[:, 0], y=_p[:, 1], mode="markers",
                marker=dict(size=3.5, color=_sc.z.numpy(), colorscale="Turbo", cmin=0, cmax=80,
                            showscale=False),
                showlegend=False,
            ),
            row=_row, col=_col,
        )
        _o = np.array([_sc.fov_A * 0.1, _sc.fov_A * 0.1])
        for _vec in _sc.lattice_basis_A.numpy():
            _e = _o + 3.0 * _vec
            cif_gallery.add_trace(
                go.Scatter(x=[_o[0], _e[0]], y=[_o[1], _e[1]], mode="lines",
                           line=dict(color="#00e5ff", width=2), showlegend=False),
                row=_row, col=_col,
            )
    for _k in range(1, 11):
        _xa = "x" if _k == 1 else f"x{_k}"
        _yk = "yaxis" if _k == 1 else f"yaxis{_k}"
        _xk = "xaxis" if _k == 1 else f"xaxis{_k}"
        _fv = cif_scenes[_k - 1].fov_A
        cif_gallery.layout[_yk].update(scaleanchor=_xa, constrain="domain", range=[0, _fv],
                                       showticklabels=False, ticks="")
        cif_gallery.layout[_xk].update(constrain="domain", range=[0, _fv],
                                       showticklabels=False, ticks="")
    cif_gallery.update_annotations(font_size=12)
    cif_gallery.update_layout(height=460, width=980, margin=dict(l=8, r=8, t=30, b=8),
                              plot_bgcolor="#111", title_text="Projected columns (color = Z), cyan = in-plane lattice basis")
    cif_gallery
    return


@app.cell(hide_code=True)
def _(
    OccupancyConfig,
    OmegaConf,
    cif_labels,
    cif_occ_idx,
    cif_scenes,
    cif_seed,
    cif_support,
    go,
    make_subplots,
    np,
    support_mask,
):
    occ_idx = int(cif_occ_idx.value)
    occ_scene = cif_scenes[occ_idx]
    occ_frac = float(cif_support.value)
    occ_modes = ["full", "facet_polygon", "blob"]
    occ_masks = []
    for _m in occ_modes:
        _occ = OmegaConf.structured(OccupancyConfig)
        _occ.mode = _m
        _occ.support_frac_min = occ_frac
        _occ.support_frac_max = occ_frac
        _occ.edge_clip_prob = 0.0
        _rng = np.random.default_rng(np.random.SeedSequence([int(cif_seed.value), occ_idx, 4099]))
        occ_masks.append(
            support_mask(occ_scene.positions_A, occ_scene.fov_A, occ_scene.lattice_basis_A, _occ, _rng).numpy()
        )
    occ_fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f"{_m} ({int(_k.sum())} cols)" for _m, _k in zip(occ_modes, occ_masks)],
        horizontal_spacing=0.03,
    )
    _p = occ_scene.positions_A.numpy()
    _z = occ_scene.z.numpy()
    for _c, _mask in enumerate(occ_masks):
        _out = ~_mask
        occ_fig.add_trace(
            go.Scatter(x=_p[_out, 0], y=_p[_out, 1], mode="markers",
                       marker=dict(size=3, color="#333"), showlegend=False),
            row=1, col=_c + 1,
        )
        occ_fig.add_trace(
            go.Scatter(x=_p[_mask, 0], y=_p[_mask, 1], mode="markers",
                       marker=dict(size=4.5, color=_z[_mask], colorscale="Turbo", cmin=0, cmax=80,
                                   showscale=False),
                       showlegend=False),
            row=1, col=_c + 1,
        )
    _fv = occ_scene.fov_A
    for _k in range(1, 4):
        _xa = "x" if _k == 1 else f"x{_k}"
        _yk = "yaxis" if _k == 1 else f"yaxis{_k}"
        _xk = "xaxis" if _k == 1 else f"xaxis{_k}"
        occ_fig.layout[_yk].update(scaleanchor=_xa, constrain="domain", range=[0, _fv],
                                   showticklabels=False, ticks="")
        occ_fig.layout[_xk].update(constrain="domain", range=[0, _fv],
                                   showticklabels=False, ticks="")
    occ_fig.update_annotations(font_size=13)
    occ_fig.update_layout(height=380, width=980, margin=dict(l=8, r=8, t=34, b=8),
                          plot_bgcolor="#111",
                          title_text=f"Occupancy on {cif_labels[occ_idx]} -- kept (color=Z) vs clipped (grey)")
    occ_fig
    return


@app.cell(hide_code=True)
def _(
    ExperimentConfig,
    OmegaConf,
    STEMSegDataset,
    cif_seed,
    cif_support,
    go,
    make_subplots,
):
    e2e_cfg = OmegaConf.structured(ExperimentConfig)
    e2e_cfg.data.provider = "cif"
    e2e_cfg.data.occupancy.mode = "facet_polygon"
    e2e_cfg.data.occupancy.support_frac_min = float(cif_support.value)
    e2e_cfg.data.occupancy.support_frac_max = float(cif_support.value)
    e2e_cfg.data.synthetic.n_scenes = 10
    e2e_cfg.data.synthetic.draws_per_scene = 3
    e2e_cfg.data.synthetic.master_seed = int(cif_seed.value)
    e2e_ds = STEMSegDataset(e2e_cfg)
    e2e_S = e2e_ds.source.crop_size
    e2e_show = [0, 12, 21, 3]
    e2e_titles = []
    for _i in e2e_show:
        _s = e2e_ds.source.get(_i)
        e2e_titles.append(f"#{_i} input ({_s.positions_A.shape[0]} cols)")
    e2e_titles += [f"#{_i} label" for _i in e2e_show]
    e2e_fig = make_subplots(rows=2, cols=len(e2e_show), subplot_titles=e2e_titles,
                            horizontal_spacing=0.02, vertical_spacing=0.12)
    for _c, _i in enumerate(e2e_show):
        _img, _mask = e2e_ds[_i]
        _s = e2e_ds.source.get(_i)
        _ppx = _s.positions_A.numpy() / _s.input_pixel_size_A
        _in = (_ppx[:, 0] >= 0) & (_ppx[:, 0] < e2e_S) & (_ppx[:, 1] >= 0) & (_ppx[:, 1] < e2e_S)
        e2e_fig.add_trace(go.Heatmap(z=_img[0].numpy(), colorscale="gray", showscale=False), row=1, col=_c + 1)
        e2e_fig.add_trace(
            go.Scatter(x=_ppx[_in, 0], y=_ppx[_in, 1], mode="markers",
                       marker=dict(symbol="cross-thin", size=6, color="#ff3b3b", line=dict(width=1.2, color="#ff3b3b")),
                       showlegend=False),
            row=1, col=_c + 1,
        )
        e2e_fig.add_trace(go.Heatmap(z=_mask[0].numpy(), colorscale="viridis", zmin=0, zmax=1, showscale=False),
                          row=2, col=_c + 1)
    _n = 2 * len(e2e_show)
    for _k in range(1, _n + 1):
        _xa = "x" if _k == 1 else f"x{_k}"
        _yk = "yaxis" if _k == 1 else f"yaxis{_k}"
        _xk = "xaxis" if _k == 1 else f"xaxis{_k}"
        e2e_fig.layout[_yk].update(scaleanchor=_xa, constrain="domain", range=[e2e_S, 0],
                                   showticklabels=False, ticks="")
        e2e_fig.layout[_xk].update(constrain="domain", range=[0, e2e_S], showticklabels=False, ticks="")
    e2e_fig.update_annotations(font_size=12)
    e2e_fig.update_layout(height=520, width=860, margin=dict(l=8, r=8, t=30, b=8),
                          title_text="cif + facet_polygon -- finite particles (and #3: a valid empty-vacuum negative)")
    e2e_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visual calibration against real S/TEM images

    The forward model is tuned by eye against a sample of real micrographs from the
    `TEM-ImageNet` set (`data/reference/`, fetched by `scripts/fetch_reference_images.py`).
    Two checks: a **side-by-side gallery** (do synthetic crops read as the same kind of
    image?) and **pooled summary statistics** -- an intensity histogram and an
    azimuthally-averaged power spectrum overlaying the real and synthetic distributions.

    The augmentation sampler now also draws, per render, a **defocus** and a **2-fold
    astigmatism** (magnitude + azimuth), so columns are no longer perfectly circular.
    Astigmatism is only visible **away from focus** -- at the mean focal plane the probe
    is the near-circular disk of least confusion -- which is why a defocus is drawn
    alongside it. The final panel makes that dependence explicit.
    """)
    return


@app.cell
def _(Image, mo, np, pathlib, torch):
    real_paths = sorted(pathlib.Path("data/reference").glob("*.png"))
    real_imgs = []
    for _p in real_paths:
        _a = np.asarray(Image.open(_p).convert("L"), dtype=np.float32)
        _t = torch.from_numpy(_a)
        real_imgs.append((_t - _t.min()) / (_t.max() - _t.min() + 1e-12))

    mo.md(
        f"Loaded **{len(real_imgs)}** reference images "
        f"({'x'.join(str(s) for s in real_imgs[0].shape)} px, grayscale, "
        f"normalized to [0, 1])."
        if real_imgs
        else "No reference images found -- run "
        "`uv run python scripts/fetch_reference_images.py --count 50`."
    )
    return (real_imgs,)


@app.cell
def _(
    AugmentationSampler,
    CIFProvider,
    ExperimentConfig,
    FiniteSupportProvider,
    OccupancyConfig,
    OmegaConf,
    SyntheticLatticeProvider,
    TEMRenderer,
    mo,
    np,
    real_imgs,
    replace,
):
    # Tuning candidates for configs/default.yaml (Task 8): widen the noisy / low-contrast tail.
    cal_cfg = OmegaConf.structured(ExperimentConfig)
    cal_cfg.generation.sampler.n_peak_min = 8.0       # was 30  -- lower dose floor => noisier extreme
    cal_cfg.generation.sampler.n_bg_frac_max = 0.18   # was 0.05 -- higher dark floor => lower contrast
    cal_cfg.generation.sampler.bg_constant_c_max = 0.55  # was 0.40 -- stronger flat background
    cal_cfg.generation.sampler.bg_ramp_c0_max = 0.45     # was 0.30 -- stronger ramp background
    cal_sampler = AugmentationSampler(cal_cfg.generation.sampler, master_seed=7)
    cal_renderer = TEMRenderer(cal_cfg.generation)

    # Scene sources mirror the training pipeline: square grids, real CIF lattices, and
    # occupancy-clipped finite particles (edges + vacuum) -- not just a single grid.
    _occ_blob = OmegaConf.structured(OccupancyConfig)
    _occ_blob.mode = "blob"
    _occ_facet = OmegaConf.structured(OccupancyConfig)
    _occ_facet.mode = "facet_polygon"
    _syn = cal_cfg.data.synthetic
    _cifkw = dict(
        manifest_path=str(cal_cfg.data.cif.manifest_path), n_scenes=24,
        n_exponent=float(cal_cfg.generation.sampler.z_exponent),
        group_tol_A=float(cal_cfg.data.cif.group_tol_A),
        scene_fov_A=(float(cal_cfg.data.cif.scene_fov_A_min), float(cal_cfg.data.cif.scene_fov_A_max)),
        rotation_jitter_deg=float(cal_cfg.data.cif.rotation_jitter_deg),
    )
    cal_sources = [
        ("grid", SyntheticLatticeProvider(_syn, 7)),
        ("grid+occ", FiniteSupportProvider(SyntheticLatticeProvider(_syn, 8), _occ_blob, 8)),
        ("cif", CIFProvider(master_seed=11, **_cifkw)),
        ("cif+facet", FiniteSupportProvider(CIFProvider(master_seed=12, **_cifkw), _occ_facet, 12)),
    ]

    _n_synth = min(len(real_imgs), 24) or 16
    synth_imgs = []
    synth_meta = []
    for _i in range(_n_synth):
        _kind, _prov = cal_sources[_i % len(cal_sources)]
        _scene = _prov.get(_i)
        _cond, _p = cal_sampler.sample(_i, max_fov_A=float(_scene.fov_A))
        _p = replace(_p, output_size=256, pixel_size_A=float(_scene.fov_A) / 256)
        synth_imgs.append(cal_renderer.render(_scene, _cond, _p).image)
        synth_meta.append((_kind, _cond.defocus_A, _cond.astig_a1_A, _p.noise.n_peak))

    _npeak = np.array([_m[3] for _m in synth_meta])
    mo.md(
        f"Rendered **{len(synth_imgs)}** synthetic 256x256 micrographs across "
        f"{len(cal_sources)} scene kinds ({', '.join(k for k, _ in cal_sources)}) -- grids, real "
        f"lattices, and finite particles. Dose n_peak in [{_npeak.min():.0f}, {_npeak.max():.0f}] "
        f"(lower = noisier); background and dark-floor ceilings raised for a lower-contrast tail."
    )
    return cal_renderer, synth_imgs


@app.cell(hide_code=True)
def _(go, make_subplots, np, real_imgs, synth_imgs):
    _rng = np.random.default_rng(3)
    _n = min(5, len(real_imgs), len(synth_imgs))
    _ri = _rng.choice(len(real_imgs), size=_n, replace=False)
    _si = _rng.choice(len(synth_imgs), size=_n, replace=False)
    _fig = make_subplots(
        rows=2, cols=_n, row_titles=["real", "synthetic"],
        horizontal_spacing=0.02, vertical_spacing=0.06,
    )
    for _c in range(_n):
        _fig.add_trace(go.Heatmap(z=real_imgs[int(_ri[_c])].numpy(), colorscale="gray", showscale=False), row=1, col=_c + 1)
        _fig.add_trace(go.Heatmap(z=synth_imgs[int(_si[_c])].numpy(), colorscale="gray", showscale=False), row=2, col=_c + 1)
    for _k in range(1, 2 * _n + 1):
        _xa = "x" if _k == 1 else f"x{_k}"
        _yk = "yaxis" if _k == 1 else f"yaxis{_k}"
        _xk = "xaxis" if _k == 1 else f"xaxis{_k}"
        _fig.layout[_yk].update(scaleanchor=_xa, constrain="domain", autorange="reversed",
                                showticklabels=False, ticks="")
        _fig.layout[_xk].update(constrain="domain", showticklabels=False, ticks="")
    _fig.update_annotations(font_size=12)
    _fig.update_layout(height=380, width=860, margin=dict(l=8, r=24, t=22, b=8),
                       title_text="Reference (top) vs synthetic (bottom)")
    _fig
    return


@app.cell(hide_code=True)
def _(
    go,
    intensity_histogram,
    make_subplots,
    radial_power_spectrum,
    real_imgs,
    synth_imgs,
    torch,
):
    _rc, _re = intensity_histogram(torch.cat([_x.reshape(-1) for _x in real_imgs]), bins=64)
    _sc, _se = intensity_histogram(torch.cat([_x.reshape(-1) for _x in synth_imgs]), bins=64)
    _rd = (_rc / (_rc.sum() * (_re[1] - _re[0]))).numpy()
    _sd = (_sc / (_sc.sum() * (_se[1] - _se[0]))).numpy()
    _rcent = ((_re[:-1] + _re[1:]) / 2).numpy()
    _scent = ((_se[:-1] + _se[1:]) / 2).numpy()


    def _avg_ps(_imgs):
        _ps = [radial_power_spectrum(_x) for _x in _imgs]
        _L = min(len(_p[1]) for _p in _ps)
        return _ps[0][0][:_L].numpy(), torch.stack([_p[1][:_L] for _p in _ps]).mean(0).numpy()


    _frq, _Pr = _avg_ps(real_imgs)
    _frs, _Ps = _avg_ps(synth_imgs)

    _fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                         subplot_titles=["intensity histogram (density)",
                                         "radial power spectrum (mean, log)"])
    _fig.add_trace(go.Scatter(x=_rcent, y=_rd, mode="lines", name="real",
                              line=dict(color="#1f77b4", shape="hv")), row=1, col=1)
    _fig.add_trace(go.Scatter(x=_scent, y=_sd, mode="lines", name="synthetic",
                              line=dict(color="#d62728", shape="hv")), row=1, col=1)
    _fig.add_trace(go.Scatter(x=_frq[1:], y=_Pr[1:], mode="lines", name="real",
                              line=dict(color="#1f77b4"), showlegend=False), row=1, col=2)
    _fig.add_trace(go.Scatter(x=_frs[1:], y=_Ps[1:], mode="lines", name="synthetic",
                              line=dict(color="#d62728"), showlegend=False), row=1, col=2)
    _fig.update_yaxes(type="log", row=1, col=2)
    _fig.update_xaxes(title_text="normalized intensity", row=1, col=1)
    _fig.update_xaxes(title_text="cycles / image", row=1, col=2)
    _fig.update_layout(height=340, width=880, margin=dict(l=8, r=8, t=40, b=8),
                       legend=dict(x=0.46, y=0.98, xanchor="right", yanchor="top"))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Astigmatism: elliptical columns require defocus

    A single column rendered at the sampler's **maximum** defocus and astigmatism
    (`defocus_A_max`, `astig_a1_A_max`), sweeping the 2-fold magnitude `A1` (rows) and
    azimuth (columns) -- i.e. the strongest ellipticity the current defaults allow; most
    random draws are milder. At **zero defocus** the column stays circular regardless of
    `A1`; here defocus is held nonzero, so increasing `A1` stretches the column into an
    ellipse and rotating the azimuth by 90 deg rotates the ellipse by 90 deg. The axis
    ratio (from the image second moments) is annotated on each panel.
    """)
    return


@app.cell
def _(
    BackgroundSpec,
    ColumnList,
    IMAGING_CONDITIONS,
    NoiseSpec,
    OmegaConf,
    RenderParams,
    SamplerConfig,
    cal_renderer,
    mo,
    np,
    replace,
    torch,
):
    _scfg = OmegaConf.structured(SamplerConfig)
    astig_defocus_A = float(_scfg.defocus_A_max)
    astig_mags = [0.0, _scfg.astig_a1_A_max / 2.0, float(_scfg.astig_a1_A_max)]
    astig_azimuths = [0.0, np.pi / 2]
    astig_fov_A, astig_px = 10.0, 0.06
    astig_cols = ColumnList(
        positions_A=torch.tensor([[astig_fov_A / 2, astig_fov_A / 2]]),
        z=torch.tensor([60.0]), count=torch.ones(1), fov_A=astig_fov_A,
    )
    astig_base = IMAGING_CONDITIONS["cond1"]
    astig_params = RenderParams(
        output_size=int(astig_fov_A / astig_px), pixel_size_A=astig_px, rotation_deg=0.0,
        z_exponent=1.7, background=BackgroundSpec(kind="constant", params={}),
        noise=NoiseSpec(n_peak=1e9, scan_freq_cyc_per_row=0.0), seed=0,
    )


    def _axis_ratio(_img):
        _a = _img.numpy().astype(np.float64)
        _a = _a - _a.min()
        _ys, _xs = np.mgrid[0:_a.shape[0], 0:_a.shape[1]]
        _t = _a.sum()
        if _t <= 0:
            return 1.0
        _my, _mx = (_ys * _a).sum() / _t, (_xs * _a).sum() / _t
        _cyy = ((_ys - _my) ** 2 * _a).sum() / _t
        _cxx = ((_xs - _mx) ** 2 * _a).sum() / _t
        _cxy = ((_xs - _mx) * (_ys - _my) * _a).sum() / _t
        _w, _ = np.linalg.eigh(np.array([[_cxx, _cxy], [_cxy, _cyy]]))
        return float((_w.max() / _w.min()) ** 0.5)


    astig_panel = []
    for _mag in astig_mags:
        _row = []
        for _az in astig_azimuths:
            _c = replace(astig_base, defocus_A=astig_defocus_A, astig_a1_A=float(_mag), astig_a1_azimuth_rad=_az)
            _img = cal_renderer.render(astig_cols, _c, astig_params).image
            _row.append((_img, _axis_ratio(_img)))
        astig_panel.append(_row)
    mo.md(
        f"Rendered the astigmatism sweep at defocus = {int(astig_defocus_A)} A "
        f"(sampler max), A1 up to {int(astig_mags[-1])} A."
    )
    return astig_azimuths, astig_defocus_A, astig_mags, astig_panel


@app.cell(hide_code=True)
def _(
    astig_azimuths,
    astig_defocus_A,
    astig_mags,
    astig_panel,
    go,
    make_subplots,
    np,
):
    _titles = [
        f"A1={int(_m)} A, az={int(np.degrees(_az))} deg (r={_ar:.2f})"
        for _mi, _m in enumerate(astig_mags)
        for _az, (_img, _ar) in zip(astig_azimuths, astig_panel[_mi])
    ]
    _fig = make_subplots(rows=len(astig_mags), cols=len(astig_azimuths),
                         subplot_titles=_titles, horizontal_spacing=0.04, vertical_spacing=0.10)
    for _mi in range(len(astig_mags)):
        for _ai in range(len(astig_azimuths)):
            _img, _ar = astig_panel[_mi][_ai]
            _fig.add_trace(go.Heatmap(z=_img.numpy(), colorscale="gray", showscale=False),
                           row=_mi + 1, col=_ai + 1)
    _n = len(astig_mags) * len(astig_azimuths)
    for _k in range(1, _n + 1):
        _xa = "x" if _k == 1 else f"x{_k}"
        _yk = "yaxis" if _k == 1 else f"yaxis{_k}"
        _xk = "xaxis" if _k == 1 else f"xaxis{_k}"
        _fig.layout[_yk].update(scaleanchor=_xa, constrain="domain", autorange="reversed",
                                showticklabels=False, ticks="")
        _fig.layout[_xk].update(constrain="domain", showticklabels=False, ticks="")
    _fig.update_annotations(font_size=11)
    _fig.update_layout(height=620, width=520, margin=dict(l=8, r=8, t=30, b=8),
                       title_text=f"2-fold astigmatism at defocus {int(astig_defocus_A)} A")
    _fig
    return


if __name__ == "__main__":
    app.run()
