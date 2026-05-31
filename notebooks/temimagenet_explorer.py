import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import torch
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from omegaconf import OmegaConf

    from inr_unet.config import ExperimentConfig, SamplerConfig
    from inr_unet.data import (
        IMAGING_CONDITIONS,
        AugmentationSampler,
        ColumnList,
        LIIFSegDataset,
        RenderParams,
        STEMSegDataset,
        TEMRenderer,
    )
    from inr_unet.data.generation.structures import BackgroundSpec, NoiseSpec
    from inr_unet.data.generation.labels import column_radius
    from inr_unet.data.generation.psf import wavelength_A

    return (
        AugmentationSampler,
        BackgroundSpec,
        ColumnList,
        ExperimentConfig,
        IMAGING_CONDITIONS,
        LIIFSegDataset,
        NoiseSpec,
        OmegaConf,
        RenderParams,
        STEMSegDataset,
        SamplerConfig,
        TEMRenderer,
        column_radius,
        go,
        make_subplots,
        mo,
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


if __name__ == "__main__":
    app.run()
