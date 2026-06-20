import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Cross-resolution head-to-head — INR-UNet vs. plain UNet baseline

    Compares the two model classes along two independent axes:

    - **Axis A — output resolution**: INR-UNet decodes its LIIF feature grid at an arbitrary
      query resolution; the baseline produces a fixed 128² heatmap which is bilinear-resized
      to match.  Both models were trained at 128²; sizes above 384 are **extrapolation**.
    - **Axis B — input field of view**: both models are evaluated at fixed 128² output but with
      varying input FOV (Å).  The training FOV band is 12–32 Å; 8, 48, and 64 Å are
      **extrapolation**.

    All curves are eval-only on the held-out **test split**.  Pick a resolution bundle below.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from runs import list_runs, load_resolution_sweep, run_kind

    RUNS_ROOT = Path(__file__).parent.parent / "runs"

    PALETTE = ["#2a9d8f", "#e76f51", "#264653", "#8338ec", "#3a86ff", "#e9c46a", "#ff006e"]
    return PALETTE, RUNS_ROOT, go, list_runs, load_resolution_sweep, make_subplots, mo, pl, run_kind


@app.cell(hide_code=True)
def _(RUNS_ROOT, list_runs, mo, run_kind):
    _resolution_runs = [p for p in list_runs(RUNS_ROOT) if run_kind(p) == "resolution"]
    _opts = {p.name: str(p) for p in _resolution_runs}
    run_picker = mo.ui.dropdown(
        options=_opts,
        label="resolution bundle",
    )
    run_picker if _resolution_runs else mo.callout(
        mo.md(
            f"No resolution bundles under `{RUNS_ROOT}`.  "
            "Run the Colab resolution driver and pull the bundle into `runs/` first."
        ),
        kind="warn",
    )
    return (run_picker,)


@app.cell
def _(load_resolution_sweep, mo, run_picker):
    mo.stop(
        not run_picker.value,
        mo.callout(
            mo.md("Select a resolution bundle above."),
            kind="warn",
        ),
    )
    art = load_resolution_sweep(run_picker.value)
    return (art,)


@app.cell(hide_code=True)
def _(PALETTE, art, go, make_subplots, mo):
    _models = art.output_sweep["model"].unique().sort().to_list()
    _colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(_models)}

    _fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("F1 vs output size", "mean offset (pm) vs output size"),
        horizontal_spacing=0.10,
    )

    _extrapolation_x0 = 384

    for _m in _models:
        _sub = art.output_sweep.filter(art.output_sweep["model"] == _m).sort("output_size")
        _col = _colors[_m]
        _fig.add_trace(
            go.Scatter(
                x=_sub["output_size"].to_list(),
                y=_sub["f1"].to_list(),
                mode="lines+markers",
                name=_m,
                legendgroup=_m,
                line=dict(color=_col),
            ),
            row=1, col=1,
        )
        _fig.add_trace(
            go.Scatter(
                x=_sub["output_size"].to_list(),
                y=(_sub["mean_offset_A"] * 100).to_list(),
                mode="lines+markers",
                name=_m,
                legendgroup=_m,
                showlegend=False,
                line=dict(color=_col),
            ),
            row=1, col=2,
        )

    # shade extrapolation region (output_size > 384)
    _x_max = int(art.output_sweep["output_size"].max()) + 32
    for _col_idx in (1, 2):
        _fig.add_vrect(
            x0=_extrapolation_x0, x1=_x_max,
            fillcolor="rgba(180,180,180,0.18)",
            line_width=0,
            row=1, col=_col_idx,
        )

    _fig.update_xaxes(title_text="output size (px)")
    _fig.update_yaxes(title_text="F1", row=1, col=1)
    _fig.update_yaxes(title_text="offset (pm)", row=1, col=2)
    _fig.update_layout(
        height=420, width=1000,
        margin=dict(l=8, r=8, t=54, b=8),
        title="Axis A — output resolution sweep (shaded = extrapolation beyond 384 px)",
        legend=dict(orientation="h", y=-0.18),
    )

    mo.vstack([
        _fig,
        mo.md(
            "*The baseline curve is **bilinear interpolation** of its native 128² heatmap.  "
            "INR-UNet queries the LIIF feature grid directly at the target resolution.*"
        ),
    ])
    return


@app.cell(hide_code=True)
def _(PALETTE, art, go, make_subplots, mo):
    _models_fov = art.fov_sweep["model"].unique().sort().to_list()
    _colors_fov = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(_models_fov)}

    _fig2 = make_subplots(
        rows=1, cols=2,
        subplot_titles=("F1 vs input FOV", "mean offset (pm) vs input FOV"),
        horizontal_spacing=0.10,
    )

    # training band: 12–32 Å
    _train_fov_lo, _train_fov_hi = 12, 32

    for _m in _models_fov:
        _sub = art.fov_sweep.filter(art.fov_sweep["model"] == _m).sort("fov_A")
        _col = _colors_fov[_m]
        _fig2.add_trace(
            go.Scatter(
                x=_sub["fov_A"].to_list(),
                y=_sub["f1"].to_list(),
                mode="lines+markers",
                name=_m,
                legendgroup=_m,
                line=dict(color=_col),
            ),
            row=1, col=1,
        )
        _fig2.add_trace(
            go.Scatter(
                x=_sub["fov_A"].to_list(),
                y=(_sub["mean_offset_A"] * 100).to_list(),
                mode="lines+markers",
                name=_m,
                legendgroup=_m,
                showlegend=False,
                line=dict(color=_col),
            ),
            row=1, col=2,
        )

    # shade trained band (12–32 Å) to distinguish from extrapolation
    _fov_min = float(art.fov_sweep["fov_A"].min())
    _fov_max = float(art.fov_sweep["fov_A"].max())
    for _col_idx in (1, 2):
        _fig2.add_vrect(
            x0=_train_fov_lo, x1=_train_fov_hi,
            fillcolor="rgba(42,157,143,0.12)",
            line_width=0,
            row=1, col=_col_idx,
        )

    _fig2.update_xaxes(title_text="input FOV (Å)")
    _fig2.update_yaxes(title_text="F1", row=1, col=1)
    _fig2.update_yaxes(title_text="offset (pm)", row=1, col=2)
    _fig2.update_layout(
        height=420, width=1000,
        margin=dict(l=8, r=8, t=54, b=8),
        title="Axis B — input FOV sweep (shaded = trained 12–32 Å band; 8/48/64 Å are extrapolation)",
        legend=dict(orientation="h", y=-0.18),
    )

    mo.vstack([
        _fig2,
        mo.md(
            "*Both models were trained on tiles drawn from the **12–32 Å** FOV band (shaded green).  "
            "8, 48, and 64 Å are out-of-distribution extrapolation points.*"
        ),
    ])
    return


@app.cell(hide_code=True)
def _(art, mo):
    mo.stop(
        not art.panels,
        mo.callout(
            mo.md("No panel images in this bundle (`panels.npz` absent).  "
                  "Re-run the driver with `make_panels=True` to generate the gallery."),
            kind="warn",
        ),
    )

    _sizes = sorted(
        int(k.split("_")[1])
        for k in art.panels
        if k.startswith("inr_") or k.startswith("base_")
    )
    _n_tiles = int(art.panels["input"].shape[0])

    tile_slider = mo.ui.slider(0, _n_tiles - 1, value=0, label="tile")
    size_slider = mo.ui.slider(
        0, len(_sizes) - 1, value=0,
        label="output size",
        show_value=True,
    )
    mo.hstack([tile_slider, size_slider], justify="start", gap=2)
    return size_slider, tile_slider


@app.cell(hide_code=True)
def _(art, mo, size_slider, tile_slider):
    import plotly.express as px

    _sizes_ordered = sorted(
        int(k.split("_")[1])
        for k in art.panels
        if k.startswith("inr_")
    )
    _sz = _sizes_ordered[int(size_slider.value)]
    _i = int(tile_slider.value)

    _inp = art.panels["input"][_i]
    _inr_key = f"inr_{_sz}"
    _base_key = f"base_{_sz}"

    _has_inr = _inr_key in art.panels
    _has_base = _base_key in art.panels


    def _hm(z, title, cmap="Viridis"):
        f = px.imshow(z, color_continuous_scale=cmap, origin="upper")
        f.update_layout(
            title=title, height=240, width=240,
            margin=dict(l=4, r=4, t=28, b=4),
            coloraxis_showscale=False,
        )
        f.update_xaxes(visible=False)
        f.update_yaxes(visible=False)
        return f


    _panels_out = [_hm(_inp, "input 128²", cmap="Gray")]
    if _has_inr:
        _panels_out.append(_hm(art.panels[_inr_key][_i], f"INR-UNet {_sz}²"))
    if _has_base:
        _panels_out.append(_hm(art.panels[_base_key][_i], f"baseline (bilinear) {_sz}²"))

    mo.vstack([
        mo.md(f"### Gallery — tile {_i}, output size {_sz}²"),
        mo.hstack(_panels_out, justify="start", gap=0.5),
    ])
    return (px,)


if __name__ == "__main__":
    app.run()
