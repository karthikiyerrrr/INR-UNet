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
    from inr_unet.data.generation.calibration import (
            intensity_histogram,
            noise_autocorrelation,
            radial_power_spectrum,
        )
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
        noise_autocorrelation,
        np,
        pathlib,
        project_structure,
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
    ExperimentConfig,
    IMAGING_CONDITIONS,
    NoiseSpec,
    OmegaConf,
    RenderParams,
    SamplerConfig,
    TEMRenderer,
    mo,
    np,
    replace,
    torch,
):
    _scfg = OmegaConf.structured(SamplerConfig)
    astig_renderer = TEMRenderer(OmegaConf.structured(ExperimentConfig).generation)
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
            _img = astig_renderer.render(astig_cols, _c, astig_params).image
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
        n_scenes=12,
        master_seed=int(cif_seed.value),
        n_exponent=float(cif_cfg.data.cif.z_eff_exponent),
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
    cif_scenes = [cif_prov.get(_i) for _i in range(12)]
    _cif_nmin = min(s.positions_A.shape[0] for s in cif_scenes)
    _cif_nmax = max(s.positions_A.shape[0] for s in cif_scenes)
    mo.md(
        f"Projected **{len(cif_scenes)}** bundled structures &middot; {_cif_nmin}-{_cif_nmax} "
        f"columns/scene &middot; effective-Z reduction with n = {cif_cfg.data.cif.z_eff_exponent}"
    )
    return cif_labels, cif_scenes


@app.cell(hide_code=True)
def _(cif_labels, cif_scenes, go, make_subplots, np):
    cif_gallery = make_subplots(
        rows=2, cols=6, subplot_titles=cif_labels,
        horizontal_spacing=0.02, vertical_spacing=0.10,
    )
    for _k, _sc in enumerate(cif_scenes):
        _row, _col = _k // 6 + 1, _k % 6 + 1
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
    for _k in range(1, 13):
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
    ## Visual calibration against real S/TEM images

    Tune the forward model **by eye** against a sample of real micrographs from the
    `TEM-ImageNet` set (`data/reference/`). The sliders below drive the paused and the
    newly-added knobs; scrub them until the synthetic distributions track the real ones,
    then read off the values and persist them to `configs/default.yaml`.

    Three pooled checks overlay **real (blue)** vs **synthetic (red)**:

    - **intensity histogram** — pixel-value density on a *fixed* `[0, 1]` range, so the
      two are directly comparable (contrast, dark floor, saturation tail).
    - **radial power spectrum** — azimuthally-averaged `|FFT|^2`; lattice peaks and the
      high-frequency noise floor.
    - **noise autocorrelation** — mean autocorrelation lag slice. **x-lag (solid)** is
      along the scan rows, **y-lag (dotted)** across them. Real ADF shows *elevated x-lag*
      at small lags (row-correlated scan noise) that the synthetic should approach.

    Scene coverage spans grids, real CIF lattices (now incl. graphene / MoS2), occupancy-
    clipped finite particles, and the partial-FOV "pillar-in-vacuum" path. Each render
    also draws a per-image **defocus + 2-fold astigmatism** and a per-render **Z-exponent**.
    """)
    return


@app.cell
def _(Image, mo, np, pathlib, torch):
    NOMINAL_PIXEL_SIZE_A = 0.26  # fallback when a params file is missing/short

    real_paths = sorted(pathlib.Path("data/reference").glob("*.png"))
    real_imgs = []
    real_pixel_sizes = []
    real_indices = []
    for _p in real_paths:
        _a = np.asarray(Image.open(_p).convert("L"), dtype=np.float32)
        _t = torch.from_numpy(_a)
        real_imgs.append((_t - _t.min()) / (_t.max() - _t.min() + 1e-12))
        real_indices.append(int(_p.stem))
        _pf = _p.parent / "params" / (_p.stem + ".txt")
        _px = NOMINAL_PIXEL_SIZE_A
        if _pf.exists():
            _vals = [float(_v) for _v in _pf.read_text().split()]
            if len(_vals) > 11:
                _px = _vals[11]
        real_pixel_sizes.append(_px)

    _matched = sum(
        1 for _p in real_paths if (_p.parent / "params" / (_p.stem + ".txt")).exists()
    )
    mo.md(
        f"Loaded **{len(real_imgs)}** reference images "
        f"({'x'.join(str(s) for s in real_imgs[0].shape)} px, grayscale, [0, 1]); "
        f"pixel size from params for {_matched}/{len(real_imgs)} "
        f"(range {min(real_pixel_sizes):.3f}-{max(real_pixel_sizes):.3f} A/px)."
        if real_imgs
        else "No reference images found -- run "
        "`uv run python scripts/fetch_reference_images.py --count 50`."
    )
    return real_imgs, real_indices


@app.cell(hide_code=True)
def _(mo):
    # Calibration tuning knobs. Defaults = current configs/default.yaml; scrub to match real.
    cal_master_seed = mo.ui.slider(0, 30, value=7, step=1, label="master seed")
    cal_n_synth = mo.ui.slider(8, 48, value=24, step=4, label="num synthetic")
    # paused knobs (current defaults: 30 / 0.05 / 0.40 / 0.30)
    cal_n_peak_min = mo.ui.slider(8, 100, value=15, step=2, label="n_peak_min (dose floor)")
    cal_n_peak_max = mo.ui.slider(500, 3000, value=3000, step=100, label="n_peak_max (dose ceil)")
    cal_n_bg_frac_max = mo.ui.slider(0.0, 0.30, value=0.05, step=0.01, label="n_bg_frac_max (dark floor)")
    cal_bg_const_max = mo.ui.slider(0.10, 0.70, value=0.40, step=0.05, label="bg_constant_c_max")
    cal_bg_ramp_max = mo.ui.slider(0.10, 0.60, value=0.30, step=0.05, label="bg_ramp_c0_max")
    # new knobs
    cal_perlin_amp_max = mo.ui.slider(0.10, 0.60, value=0.35, step=0.05, label="bg_perlin_amp_max")
    cal_perlin_cells_max = mo.ui.slider(3, 10, value=6, step=1, label="bg_perlin_cells_max")
    cal_perlin_w = mo.ui.slider(0.0, 3.0, value=1.0, step=0.5, label="perlin bg weight")
    cal_partial_fov_prob = mo.ui.slider(0.0, 0.6, value=0.0, step=0.05, label="partial_fov_prob")
    cal_zexp_min = mo.ui.slider(1.3, 1.9, value=1.5, step=0.05, label="z_exponent_min")
    cal_zexp_max = mo.ui.slider(1.6, 2.2, value=2.0, step=0.05, label="z_exponent_max")
    cal_z_eff = mo.ui.slider(1.3, 2.0, value=1.7, step=0.05, label="z_eff_exponent (projection)")
    # scan-noise (sideways smear) knobs: amplitude lives on the imaging condition (default 0.2 A),
    # frequency + vertical coupling are sampler ranges.
    cal_sigma_jitter_A = mo.ui.slider(0.0, 1.0, value=0.10, step=0.05, label="sigma_jitter_A (smear amplitude)")
    cal_scan_freq_min = mo.ui.slider(0.0, 0.5, value=0.02, step=0.02, label="scan_freq_min")
    cal_scan_freq_max = mo.ui.slider(0.0, 0.5, value=0.5, step=0.02, label="scan_freq_max")
    cal_scan_beta_max = mo.ui.slider(0.0, 1.0, value=0.5, step=0.05, label="scan_beta_max (vert coupling)")

    mo.vstack([
        mo.hstack([cal_master_seed, cal_n_synth], justify="start", gap=1),
        mo.hstack([cal_n_peak_min, cal_n_peak_max, cal_n_bg_frac_max, cal_bg_const_max, cal_bg_ramp_max],
                  justify="start", gap=1),
        mo.hstack([cal_sigma_jitter_A, cal_scan_freq_min, cal_scan_freq_max, cal_scan_beta_max],
                  justify="start", gap=1),
        mo.hstack([cal_perlin_amp_max, cal_perlin_cells_max, cal_perlin_w, cal_partial_fov_prob],
                  justify="start", gap=1),
        mo.hstack([cal_zexp_min, cal_zexp_max, cal_z_eff], justify="start", gap=1),
    ])
    return (
        cal_bg_const_max,
        cal_bg_ramp_max,
        cal_master_seed,
        cal_n_bg_frac_max,
        cal_n_peak_max,
        cal_n_peak_min,
        cal_n_synth,
        cal_perlin_amp_max,
        cal_perlin_cells_max,
        cal_perlin_w,
        cal_scan_beta_max,
        cal_scan_freq_max,
        cal_scan_freq_min,
        cal_sigma_jitter_A,
        cal_z_eff,
        cal_zexp_max,
        cal_zexp_min,
    )


@app.cell(hide_code=True)
def _(
    AugmentationSampler,
    CIFProvider,
    ExperimentConfig,
    FiniteSupportProvider,
    OccupancyConfig,
    OmegaConf,
    SyntheticLatticeProvider,
    TEMRenderer,
    cal_bg_const_max,
    cal_bg_ramp_max,
    cal_master_seed,
    cal_n_bg_frac_max,
    cal_n_peak_max,
    cal_n_peak_min,
    cal_n_synth,
    cal_perlin_amp_max,
    cal_perlin_cells_max,
    cal_perlin_w,
    cal_scan_beta_max,
    cal_scan_freq_max,
    cal_scan_freq_min,
    cal_sigma_jitter_A,
    cal_z_eff,
    cal_zexp_max,
    cal_zexp_min,
    mo,
    np,
    replace,
    torch,
):
    # Build a config from the calibration sliders, then render synthetic micrographs across
    # every scene kind the training pipeline produces (grids, occupancy-clipped particles, real
    # CIF lattices incl. graphene/MoS2, and the partial-FOV pillar path).
    cal_cfg = OmegaConf.structured(ExperimentConfig)
    _s = cal_cfg.generation.sampler
    _s.n_peak_min = float(cal_n_peak_min.value)
    _s.n_peak_max = float(cal_n_peak_max.value)
    _s.n_bg_frac_max = float(cal_n_bg_frac_max.value)
    _s.bg_constant_c_max = float(cal_bg_const_max.value)
    _s.bg_ramp_c0_max = float(cal_bg_ramp_max.value)
    _s.bg_perlin_amp_max = float(cal_perlin_amp_max.value)
    _s.bg_perlin_cells_max = int(cal_perlin_cells_max.value)
    _s.bg_weights["perlin"] = float(cal_perlin_w.value)
    _s.z_exponent_min = float(cal_zexp_min.value)
    _s.z_exponent_max = float(cal_zexp_max.value)
    # scan noise (sideways smear): frequency + vertical-coupling ranges on the sampler; amplitude
    # (sigma_jitter_A) lives on the imaging condition, so it is overridden per render below.
    _s.scan_freq_min, _s.scan_freq_max = sorted(
        (float(cal_scan_freq_min.value), float(cal_scan_freq_max.value))
    )
    _s.scan_beta_max = float(cal_scan_beta_max.value)
    _s.scan_beta_min = min(float(_s.scan_beta_min), _s.scan_beta_max)
    cal_cfg.data.cif.z_eff_exponent = float(cal_z_eff.value)

    cal_sampler = AugmentationSampler(_s, master_seed=int(cal_master_seed.value))
    cal_renderer = TEMRenderer(cal_cfg.generation)

    _occ_blob = OmegaConf.structured(OccupancyConfig)
    _occ_blob.mode = "blob"
    _occ_facet = OmegaConf.structured(OccupancyConfig)
    _occ_facet.mode = "facet_polygon"
    _syn = cal_cfg.data.synthetic
    _cifkw = dict(
        manifest_path=str(cal_cfg.data.cif.manifest_path), n_scenes=64,
        n_exponent=float(cal_cfg.data.cif.z_eff_exponent),
        group_tol_A=float(cal_cfg.data.cif.group_tol_A),
        scene_fov_A=(float(cal_cfg.data.cif.scene_fov_A_min), float(cal_cfg.data.cif.scene_fov_A_max)),
        rotation_jitter_deg=float(cal_cfg.data.cif.rotation_jitter_deg),
    )
    _cifkw_partial = dict(
        _cifkw, partial_fov_prob=1.0,
        supercell_nx_range=tuple(cal_cfg.data.cif.supercell_nx_range),
        supercell_ny_range=tuple(cal_cfg.data.cif.supercell_ny_range),
    )
    cal_sources = [
        ("grid", SyntheticLatticeProvider(_syn, 7)),
        ("grid+occ", FiniteSupportProvider(SyntheticLatticeProvider(_syn, 8), _occ_blob, 8)),
        ("cif", CIFProvider(master_seed=11, **_cifkw)),
        ("cif+facet", FiniteSupportProvider(CIFProvider(master_seed=12, **_cifkw), _occ_facet, 12)),
        ("cif+partial", CIFProvider(master_seed=13, **_cifkw_partial)),
    ]

    _n_synth = int(cal_n_synth.value)
    synth_imgs = []
    synth_meta = []
    for _i in range(_n_synth):
        _kind, _prov = cal_sources[_i % len(cal_sources)]
        _scene = _prov.get(_i)
        _cond, _p = cal_sampler.sample(_i, max_fov_A=float(_scene.fov_A))
        _cond = replace(_cond, sigma_jitter_A=float(cal_sigma_jitter_A.value))  # smear amplitude
        _fov_render = float(_p.output_size * _p.pixel_size_A)  # sampler crop FOV (<= scene); training scale
        _p = replace(_p, output_size=256, pixel_size_A=_fov_render / 256)
        _out = cal_renderer.render(_scene, _cond, _p)
        # If the sampled crop shows no columns inside the visible window (e.g. a partial-FOV pillar
        # in vacuum landed off-frame), recenter on the structure centroid and re-render -- mirrors
        # the training source's columns-in-crop guarantee so we never display a pure-noise crop.
        _q = _out.positions_A
        _w = _out.image.shape[0] * _p.pixel_size_A
        _n_vis = int(((_q >= 0) & (_q <= _w)).all(dim=1).sum()) if _q.shape[0] else 0
        if _n_vis == 0 and _scene.positions_A.shape[0] > 0:
            _ctr = _scene.positions_A.mean(dim=0)
            _scn = _scene.fov_A / 2.0
            _th = float(np.radians(_p.rotation_deg))
            _rot = torch.tensor([[np.cos(_th), -np.sin(_th)], [np.sin(_th), np.cos(_th)]],
                                dtype=_scene.positions_A.dtype)
            _p = replace(_p, position_offset_A=-((_ctr - _scn) @ _rot.T))
            _out = cal_renderer.render(_scene, _cond, _p)
        synth_imgs.append(_out.image)
        synth_meta.append((_kind, _cond.defocus_A, _cond.astig_a1_A, _p.noise.n_peak, _fov_render))

    _npeak = np.array([_m[3] for _m in synth_meta])
    mo.md(
        f"Rendered **{len(synth_imgs)}** synthetic 256x256 micrographs across "
        f"{len(cal_sources)} scene kinds ({', '.join(k for k, _ in cal_sources)}). "
        f"Dose n_peak in [{_npeak.min():.0f}, {_npeak.max():.0f}] (lower = noisier); "
        f"crop FOV in [{min(m[4] for m in synth_meta):.0f}, {max(m[4] for m in synth_meta):.0f}] A; "
        f"scan smear: sigma_jitter={float(cal_sigma_jitter_A.value):.2f} A, "
        f"freq ~ U[{_s.scan_freq_min:.2f}, {_s.scan_freq_max:.2f}] cyc/row, "
        f"beta ~ U[{_s.scan_beta_min:.2f}, {_s.scan_beta_max:.2f}]; "
        f"Z-exponent ~ U[{_s.z_exponent_min:.2f}, {_s.z_exponent_max:.2f}], "
        f"z_eff = {cal_cfg.data.cif.z_eff_exponent:.2f}."
    )
    return (synth_imgs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reference vs augmented synthetic

    The TEM-ImageNet params fix only the **imaging condition** (200 kV, 24 mrad, 0.9 A =
    `cond1`); pixel size varies per image and the structure is not recorded. So here we
    **pin cond1** and let the training **augmentation sampler** randomize everything else --
    crystal/zone, rotation, defocus, astigmatism, background family, scan noise, dose,
    z-exponent, and **FOV (the zoom)** -- to reproduce the varied character of real micrographs
    rather than a sterile grid. This mirrors the dataset we intend to generate: the crystal is
    drawn at random per row, and the seed reshuffles the reference selection, the crystal, and
    the augmentation draws. Below the pairs: synthetic-only capabilities with no
    reference-set counterpart.
    """)
    return


@app.cell(hide_code=True)
def _(
    BackgroundSpec,
    ColumnList,
    ExperimentConfig,
    IMAGING_CONDITIONS,
    NoiseSpec,
    OmegaConf,
    RenderParams,
    TEMRenderer,
    pathlib,
    project_structure,
    replace,
    torch,
):
    from pymatgen.core import Structure

    _cmp_cfg = OmegaConf.structured(ExperimentConfig)
    cmp_renderer = TEMRenderer(_cmp_cfg.generation)
    cmp_cif_dir = pathlib.Path(_cmp_cfg.data.cif.manifest_path).parent
    _cmp_manifest = OmegaConf.load(_cmp_cfg.data.cif.manifest_path)
    _ZONE_NAMES = {(0, 0, 1): "[001]", (1, 1, 0): "[110]", (1, 0, 0): "[100]"}

    # label -> (cif_filename, zone_axis list)
    cmp_choices = {}
    for _e in _cmp_manifest.entries:
        _za = tuple(int(_v) for _v in _e.zone_axis)
        _label = f"{pathlib.Path(_e.cif).stem} {_ZONE_NAMES.get(_za, str(_za))}"
        cmp_choices[_label] = (_e.cif, [int(_v) for _v in _e.zone_axis])
    cmp_labels = list(cmp_choices)
    CMP_STRIP_PROB = 0.4  # P(row-strip | partial); the remaining partial scenes are compact blobs


    def cmp_full_scene(cif, zone_axis, fov_A):
        _struct = Structure.from_file(cmp_cif_dir / cif)
        _pos, _z, _count, _basis = project_structure(
            _struct, zone_axis, fov_A=float(fov_A),
            n_exponent=float(_cmp_cfg.data.cif.z_eff_exponent),
            group_tol_A=float(_cmp_cfg.data.cif.group_tol_A),
        )
        return ColumnList(positions_A=_pos, z=_z, count=_count, fov_A=float(fov_A),
                          lattice_basis_A=_basis)


    def cmp_partial_scene(cif, zone_axis, fov_A, rng):
        # A finite patch placed with margins inside a fov-sized scene: either a compact blob
        # or an elongated row-strip (a thin supercell that reads as a tilted, isolated row once
        # the sampler rotates it). Strip length is sized to the FOV so the row stays framed.
        _struct = Structure.from_file(cmp_cif_dir / cif)
        _ze = float(_cmp_cfg.data.cif.z_eff_exponent)
        _gt = float(_cmp_cfg.data.cif.group_tol_A)
        if rng.random() < CMP_STRIP_PROB:
            _long_axis = int(rng.random() < 0.5)  # which in-plane lattice axis runs long
            _, _, _, _basis0 = project_structure(
                _struct, zone_axis, fov_A=float(fov_A), n_exponent=_ze, group_tol_A=_gt,
                supercell=(1, 1),
            )
            _spacing = float(torch.linalg.norm(_basis0[_long_axis]))
            _ncell = max(3, round(float(rng.uniform(0.4, 0.65)) * float(fov_A) / max(_spacing, 1e-6)))
            _sc = (_ncell, 1) if _long_axis == 0 else (1, _ncell)
        else:
            _sc = (int(rng.integers(1, 4)), int(rng.integers(1, 3)))  # compact blob
        _pos, _z, _count, _basis = project_structure(
            _struct, zone_axis, fov_A=float(fov_A), n_exponent=_ze, group_tol_A=_gt, supercell=_sc,
        )
        if _pos.shape[0] > 0:
            _span = _pos.max(dim=0).values - _pos.min(dim=0).values
            _slack = torch.clamp(torch.tensor(float(fov_A)) - _span, min=0.0)
            _off = torch.tensor([float(rng.uniform(0.0, float(_slack[0]))),
                                 float(rng.uniform(0.0, float(_slack[1])))], dtype=_pos.dtype)
            _pos = _pos - _pos.min(dim=0).values + _off
        return ColumnList(positions_A=_pos, z=_z, count=_count, fov_A=float(fov_A),
                          lattice_basis_A=_basis)


    def cmp_render(cif, zone_axis, px, n_peak, z_exp, defocus=0.0, astig=0.0,
                   background=None, col_width=None):
        _fov = 256.0 * float(px)
        _scene = cmp_full_scene(cif, zone_axis, _fov)
        _cond = replace(IMAGING_CONDITIONS["cond1"], defocus_A=float(defocus), astig_a1_A=float(astig))
        if col_width is None:
            _renderer = cmp_renderer
        else:
            _gcfg = OmegaConf.structured(ExperimentConfig).generation
            _gcfg.sigma_potential_A = float(col_width)
            _renderer = TEMRenderer(_gcfg)
        _params = RenderParams(
            output_size=256, pixel_size_A=float(px), rotation_deg=0.0,
            z_exponent=float(z_exp),
            background=background or BackgroundSpec(kind="constant", params={}),
            noise=NoiseSpec(n_peak=float(n_peak), scan_freq_cyc_per_row=0.1, scan_beta=0.4),
        )
        return _renderer.render(_scene, _cond, _params).image


    return (
        cmp_choices,
        cmp_full_scene,
        cmp_labels,
        cmp_partial_scene,
        cmp_render,
        cmp_renderer,
    )


@app.cell(hide_code=True)
def _(mo, real_imgs):
    _maxn = max(1, len(real_imgs))
    cmp_n = mo.ui.slider(1, _maxn, value=min(5, _maxn), step=1, label="num comparisons")
    cmp_seed = mo.ui.slider(0, 100, value=0, step=1, label="seed (selection + augmentation)")
    cmp_partial_prob = mo.ui.slider(0.0, 1.0, value=0.4, step=0.05, label="partial-FOV fraction")
    mo.vstack([cmp_n, cmp_seed, cmp_partial_prob])
    return cmp_n, cmp_partial_prob, cmp_seed


@app.cell(hide_code=True)
def _(
    AugmentationSampler,
    ExperimentConfig,
    IMAGING_CONDITIONS,
    OmegaConf,
    cmp_choices,
    cmp_full_scene,
    cmp_labels,
    cmp_n,
    cmp_partial_prob,
    cmp_partial_scene,
    cmp_renderer,
    cmp_seed,
    go,
    make_subplots,
    np,
    real_imgs,
    real_indices,
    replace,
    torch,
):
    # Pin cond1 (the only thing params.txt fixes); randomize the rest via the augmentation
    # sampler (rotation, defocus, astig, background, scan noise, dose, z-exponent, FOV=zoom) plus
    # the crystal/zone per row. Some rows render a finite particle in vacuum (partial FOV).
    _acfg = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), OmegaConf.load("configs/default.yaml"))
    _sampler = AugmentationSampler(_acfg.generation.sampler, master_seed=int(cmp_seed.value))
    _n = min(cmp_n.value, len(real_imgs))
    _sel = np.random.default_rng(int(cmp_seed.value)).permutation(len(real_imgs))[:_n]
    _titles = []
    _pairs = []
    for _r, _idx in enumerate(_sel):
        _idx = int(_idx)
        _srng = np.random.default_rng(np.random.SeedSequence([int(cmp_seed.value), _idx, 555]))
        _label = cmp_labels[int(_srng.integers(len(cmp_labels)))]  # crystal drawn at random per row
        _cif, _zone = cmp_choices[_label]
        _cond, _p = _sampler.sample(_idx, max_fov_A=80.0)
        _fov = _p.output_size * _p.pixel_size_A
        _rrng = np.random.default_rng(np.random.SeedSequence([int(cmp_seed.value), _idx, 777]))
        _is_partial = bool(_rrng.random() < cmp_partial_prob.value)
        if _is_partial:
            _scene = cmp_partial_scene(_cif, _zone, _fov, _rrng)
            _p = replace(_p, position_offset_A=torch.zeros(2))  # keep the particle framed
        else:
            _scene = cmp_full_scene(_cif, _zone, 80.0)
        _cond = replace(IMAGING_CONDITIONS["cond1"], defocus_A=_cond.defocus_A,
                        astig_a1_A=_cond.astig_a1_A, astig_a1_azimuth_rad=_cond.astig_a1_azimuth_rad)
        _synth = cmp_renderer.render(_scene, _cond, _p).image
        _pairs.append((real_imgs[_idx], _synth))
        _titles.append(f"#{real_indices[_idx]} . {_label} . {_fov:.0f}A . {'partial' if _is_partial else 'full'}")

    # Square subplot cells in pixels: every image is 1:1 (refs 256x256, synth square), so square cells
    # leave no slack for constrain="domain" to letterbox -> images fill, stable across cmp_n.
    _CELL, _GAP_V, _GAP_H = 220, 18, 8
    _TTOP, _MB, _ML, _MR = 24, 8, 8, 120
    _Hp = _n * _CELL + (_n - 1) * _GAP_V
    _Wp = 2 * _CELL + _GAP_H
    _fig = make_subplots(
        rows=_n, cols=2, column_titles=["reference", "synthetic"],
        row_titles=_titles, horizontal_spacing=_GAP_H / _Wp, vertical_spacing=_GAP_V / _Hp,
    )
    for _i, (_real, _synth) in enumerate(_pairs):
        _fig.add_trace(go.Heatmap(z=_real.numpy(), colorscale="gray", showscale=False),
                       row=_i + 1, col=1)
        _fig.add_trace(go.Heatmap(z=_synth.numpy(), colorscale="gray", zmin=0.0, zmax=1.0,
                                  showscale=False), row=_i + 1, col=2)
    for _k in range(1, 2 * _n + 1):
        _xa = "x" if _k == 1 else f"x{_k}"
        _yk = "yaxis" if _k == 1 else f"yaxis{_k}"
        _xk = "xaxis" if _k == 1 else f"xaxis{_k}"
        _fig.layout[_yk].update(scaleanchor=_xa, constrain="domain", autorange="reversed",
                                showticklabels=False, ticks="")
        _fig.layout[_xk].update(constrain="domain", showticklabels=False, ticks="")
    _fig.update_annotations(font_size=11)
    _fig.update_layout(height=_Hp + _TTOP + _MB, width=_Wp + _ML + _MR,
                       margin=dict(l=_ML, r=_MR, t=_TTOP, b=_MB),
                       title_text="Reference vs augmented synthetic (cond1 pinned, rest randomized)")
    _fig

    return


@app.cell(hide_code=True)
def _(
    BackgroundSpec,
    CIFProvider,
    ExperimentConfig,
    IMAGING_CONDITIONS,
    NoiseSpec,
    OmegaConf,
    RenderParams,
    cmp_render,
    cmp_renderer,
    go,
    make_subplots,
):
    _cfg = OmegaConf.structured(ExperimentConfig)
    _px = 0.25
    _base = ("SrTiO3.cif", [0, 0, 1])

    # partial-FOV pillar via CIFProvider (entry 0 = Pt [001]); fixed FOV so it is deterministic.
    _pp = CIFProvider(
        manifest_path=str(_cfg.data.cif.manifest_path), n_scenes=12, master_seed=5,
        n_exponent=float(_cfg.data.cif.z_eff_exponent),
        group_tol_A=float(_cfg.data.cif.group_tol_A),
        scene_fov_A=(256.0 * _px, 256.0 * _px), rotation_jitter_deg=0.0, partial_fov_prob=1.0,
    )
    _scene_pp = _pp.get(0)
    _partial = cmp_renderer.render(
        _scene_pp, IMAGING_CONDITIONS["cond1"],
        RenderParams(output_size=256, pixel_size_A=_px, z_exponent=1.7,
                     background=BackgroundSpec(kind="constant", params={}),
                     noise=NoiseSpec(n_peak=1500.0)),
    ).image

    _tiles = [
        ("graphene [001]", cmp_render("graphene.cif", [0, 0, 1], _px, 1500, 1.7)),
        ("MoS2 [001]", cmp_render("MoS2.cif", [0, 0, 1], _px, 1500, 1.7)),
        ("partial FOV (Pt pillar)", _partial),
        ("Perlin background", cmp_render(*_base, _px, 1500, 1.7,
                                         background=BackgroundSpec(kind="perlin",
                                                                   params={"amp": 0.3, "cells": 4}))),
        ("defocus 30 A", cmp_render(*_base, _px, 1500, 1.7, defocus=30.0)),
        ("astig 30 A", cmp_render(*_base, _px, 1500, 1.7, defocus=30.0, astig=30.0)),
        ("low dose (n_peak=30)", cmp_render(*_base, _px, 30, 1.7)),
    ]

    _fig = make_subplots(rows=1, cols=len(_tiles),
                         subplot_titles=[_t for _t, _ in _tiles], horizontal_spacing=0.01)
    for _i, (_t, _img) in enumerate(_tiles):
        _fig.add_trace(go.Heatmap(z=_img.numpy(), colorscale="gray", zmin=0.0, zmax=1.0,
                                  showscale=False), row=1, col=_i + 1)
    for _k in range(1, len(_tiles) + 1):
        _xa = "x" if _k == 1 else f"x{_k}"
        _yk = "yaxis" if _k == 1 else f"yaxis{_k}"
        _xk = "xaxis" if _k == 1 else f"xaxis{_k}"
        _fig.layout[_yk].update(scaleanchor=_xa, constrain="domain", autorange="reversed",
                                showticklabels=False, ticks="")
        _fig.layout[_xk].update(constrain="domain", showticklabels=False, ticks="")
    _fig.update_annotations(font_size=10)
    _fig.update_layout(height=200, width=140 * len(_tiles), margin=dict(l=8, r=8, t=28, b=8),
                       title_text="Synthetic-only capabilities (no reference-set counterpart)")
    _fig
    return


@app.cell(hide_code=True)
def _(
    go,
    intensity_histogram,
    make_subplots,
    noise_autocorrelation,
    radial_power_spectrum,
    real_imgs,
    synth_imgs,
    torch,
):
    _rc, _re = intensity_histogram(
        torch.cat([_x.reshape(-1) for _x in real_imgs]), bins=64, value_range=(0.0, 1.0)
    )
    _sc, _se = intensity_histogram(
        torch.cat([_x.reshape(-1) for _x in synth_imgs]), bins=64, value_range=(0.0, 1.0)
    )
    _rd = (_rc / (_rc.sum() * (_re[1] - _re[0]))).numpy()
    _sd = (_sc / (_sc.sum() * (_se[1] - _se[0]))).numpy()
    _cent = ((_re[:-1] + _re[1:]) / 2).numpy()


    def _avg_ps(_imgs):
        _ps = [radial_power_spectrum(_x) for _x in _imgs]
        _L = min(len(_p[1]) for _p in _ps)
        return _ps[0][0][:_L].numpy(), torch.stack([_p[1][:_L] for _p in _ps]).mean(0).numpy()


    def _ac_slices(_imgs, _half=24):
        _xs, _ys = [], []
        for _im in _imgs:
            _ac = noise_autocorrelation(_im)
            _h, _w = _ac.shape
            _cy, _cx = _h // 2, _w // 2
            if _cy - _half < 0 or _cx - _half < 0 or _cy + _half + 1 > _h or _cx + _half + 1 > _w:
                continue
            _xs.append(_ac[_cy, _cx - _half:_cx + _half + 1])
            _ys.append(_ac[_cy - _half:_cy + _half + 1, _cx])
        _lag = torch.arange(-_half, _half + 1).numpy()
        return _lag, torch.stack(_xs).mean(0).numpy(), torch.stack(_ys).mean(0).numpy()


    _frq, _Pr = _avg_ps(real_imgs)
    _frs, _Ps = _avg_ps(synth_imgs)
    _lag, _rx, _ry = _ac_slices(real_imgs)
    _lag2, _sx, _sy = _ac_slices(synth_imgs)

    _fig = make_subplots(
        rows=1, cols=3, horizontal_spacing=0.08,
        subplot_titles=["intensity histogram (density, [0,1])",
                        "radial power spectrum (mean, log)",
                        "noise autocorrelation (x solid, y dotted)"],
    )
    _fig.add_trace(go.Scatter(x=_cent, y=_rd, mode="lines", name="real",
                              line=dict(color="#1f77b4", shape="hv")), row=1, col=1)
    _fig.add_trace(go.Scatter(x=_cent, y=_sd, mode="lines", name="synthetic",
                              line=dict(color="#d62728", shape="hv")), row=1, col=1)
    _fig.add_trace(go.Scatter(x=_frq[1:], y=_Pr[1:], mode="lines",
                              line=dict(color="#1f77b4"), showlegend=False), row=1, col=2)
    _fig.add_trace(go.Scatter(x=_frs[1:], y=_Ps[1:], mode="lines",
                              line=dict(color="#d62728"), showlegend=False), row=1, col=2)
    _fig.add_trace(go.Scatter(x=_lag, y=_rx, mode="lines",
                              line=dict(color="#1f77b4"), showlegend=False), row=1, col=3)
    _fig.add_trace(go.Scatter(x=_lag, y=_ry, mode="lines",
                              line=dict(color="#1f77b4", dash="dot"), showlegend=False), row=1, col=3)
    _fig.add_trace(go.Scatter(x=_lag2, y=_sx, mode="lines",
                              line=dict(color="#d62728"), showlegend=False), row=1, col=3)
    _fig.add_trace(go.Scatter(x=_lag2, y=_sy, mode="lines",
                              line=dict(color="#d62728", dash="dot"), showlegend=False), row=1, col=3)
    _fig.update_yaxes(type="log", row=1, col=2)
    _fig.update_yaxes(range=[-0.05, 1.02], row=1, col=3)
    _fig.update_xaxes(title_text="normalized intensity", row=1, col=1)
    _fig.update_xaxes(title_text="cycles / image", row=1, col=2)
    _fig.update_xaxes(title_text="lag (px)", row=1, col=3)
    _fig.update_layout(height=340, width=1000, margin=dict(l=8, r=8, t=40, b=8),
                       legend=dict(x=0.30, y=0.98, xanchor="right", yanchor="top"))
    _fig
    return


if __name__ == "__main__":
    app.run()
