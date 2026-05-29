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

    from inr_unet.data import (
        IMAGING_CONDITIONS,
        ColumnList,
        RenderParams,
        TEMRenderer,
    )
    from inr_unet.data.generation.structures import BackgroundSpec, NoiseSpec
    from inr_unet.data.generation.labels import column_radius
    from inr_unet.data.generation.psf import wavelength_A

    return (
        BackgroundSpec,
        ColumnList,
        IMAGING_CONDITIONS,
        NoiseSpec,
        OmegaConf,
        RenderParams,
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


if __name__ == "__main__":
    app.run()
