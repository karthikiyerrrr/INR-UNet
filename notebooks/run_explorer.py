import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from runs import list_runs, load_run, load_training_run, run_kind

    RUNS_ROOT = Path(__file__).parent.parent / "runs"
    return (
        RUNS_ROOT,
        go,
        list_runs,
        load_run,
        load_training_run,
        make_subplots,
        mo,
        pl,
        px,
        run_kind,
    )


@app.cell(hide_code=True)
def _(RUNS_ROOT, list_runs, mo, run_kind):
    _runs = list_runs(RUNS_ROOT)
    _opts = {f"{p.name}  \u00b7  {run_kind(p)}": str(p) for p in _runs}
    run_picker = mo.ui.dropdown(
        options=_opts,
        value=(next(reversed(_opts)) if _opts else None),
        label="run",
    )
    run_picker if _runs else mo.callout(
        mo.md(f"No runs found under `{RUNS_ROOT}`. Pull a run from Drive into `runs/` first."),
        kind="warn",
    )
    return (run_picker,)


@app.cell
def _(load_run, load_training_run, mo, run_kind, run_picker):
    mo.stop(not run_picker.value, mo.md("Select a run above."))
    kind = run_kind(run_picker.value)
    art = load_run(run_picker.value) if kind == "overfit" else load_training_run(run_picker.value)
    art.meta
    return art, kind


@app.cell(hide_code=True)
def _(art, mo):
    _m = art.meta
    _enc = art.config.model.encoder
    _dec = art.config.model.decoder
    _syn = art.config.data.synthetic
    _loss = art.config.train.get("loss", "?")
    _nsc = _syn.get("n_scenes", "?")
    _dps = _syn.get("draws_per_scene", "?")
    mo.md(
        f"""
    ### Run `{_m['run_id']}` \u00b7 {_m['gpu_name']} \u00b7 git `{_m['git_sha']}` \u00b7 torch {_m['torch_version']}
    *{_m['purpose']} \u00b7 {_m['created']}*

    | encoder | decoder | data |
    |---|---|---|
    | base={_enc.base_channels}, depth={_enc.depth}, feat={_enc.feature_dim} | hidden={_dec.hidden_dim}, layers={_dec.num_layers}, LE={_dec.local_ensemble}, unfold={_dec.feature_unfold}, cell={_dec.cell_decode} | crop={_syn.crop_size}, sample_q={_syn.sample_q}, label={_syn.label_kind}, loss={_loss}, scenes={_nsc}\u00d7{_dps} |
    """
    )
    return


@app.cell(hide_code=True)
def _(art, go, kind, make_subplots, mo, pl):
    mo.stop(kind != "training")
    _h = art.history
    _best = art.meta["best_epoch"]
    _best_f1 = _h.filter(pl.col("epoch") == _best)["val_f1"][0]
    _cur = make_subplots(
        rows=1, cols=3,
        subplot_titles=("val F1", "loss (log-y)", "localization offset (pm)"),
        horizontal_spacing=0.07,
    )
    _cur.add_trace(go.Scatter(x=_h["epoch"], y=_h["val_f1"], mode="lines", name="val F1",
                              line=dict(color="#2a9d8f")), row=1, col=1)
    _cur.add_trace(go.Scatter(x=[_best], y=[_best_f1], mode="markers", name=f"best ep {_best}",
                              marker=dict(color="#e76f51", size=11, symbol="star")), row=1, col=1)
    _cur.add_trace(go.Scatter(x=_h["epoch"], y=_h["train_loss"], mode="lines", name="train_loss",
                              line=dict(color="#264653")), row=1, col=2)
    _cur.add_trace(go.Scatter(x=_h["epoch"], y=_h["val_loss"], mode="lines", name="val_loss",
                              line=dict(color="#e9c46a")), row=1, col=2)
    _cur.update_yaxes(type="log", row=1, col=2)
    _cur.add_trace(go.Scatter(x=_h["epoch"], y=_h["val_mean_offset_A"] * 100, mode="lines", name="mean",
                              line=dict(color="#8338ec")), row=1, col=3)
    _cur.add_trace(go.Scatter(x=_h["epoch"], y=_h["val_median_offset_A"] * 100, mode="lines", name="median",
                              line=dict(color="#3a86ff")), row=1, col=3)
    _cur.add_vline(x=_best, line=dict(color="#e76f51", dash="dot", width=1))
    _cur.update_xaxes(title_text="epoch")
    _cur.update_layout(height=380, width=1020, margin=dict(l=8, r=8, t=52, b=8),
                       title=f"Training curves \u2014 best epoch {_best} (val F1 {_best_f1:.3f})",
                       legend=dict(orientation="h", y=-0.22))
    _cur
    return


@app.cell(hide_code=True)
def _(art, go, kind, make_subplots, mo):
    mo.stop(kind != "training" or not art.val_panels)
    _p = art.val_panels
    _n = _p["input"].shape[0]
    _grid = make_subplots(
        rows=_n, cols=3,
        subplot_titles=["input", "gt", "pred"] + [""] * (3 * _n - 3),
        horizontal_spacing=0.02, vertical_spacing=0.02,
    )
    for _r in range(_n):
        _vmax = float(max(_p["gt"][_r].max(), _p["pred"][_r].max()))
        _vmax = _vmax if _vmax > 0 else 1.0
        _grid.add_trace(go.Heatmap(z=_p["input"][_r], colorscale="gray", showscale=False),
                        row=_r + 1, col=1)
        _grid.add_trace(go.Heatmap(z=_p["gt"][_r], colorscale="viridis", zmin=0, zmax=_vmax, showscale=False),
                        row=_r + 1, col=2)
        _grid.add_trace(go.Heatmap(z=_p["pred"][_r], colorscale="viridis", zmin=0, zmax=_vmax, showscale=False),
                        row=_r + 1, col=3)
    _grid.update_xaxes(showticklabels=False, ticks="")
    _grid.update_yaxes(showticklabels=False, ticks="", autorange="reversed")
    _grid.update_layout(height=190 * _n, width=720, margin=dict(l=8, r=8, t=30, b=8),
                        title="Validation tiles \u2014 input | gt | pred (gt & pred share a per-row color scale)")
    _grid
    return


@app.cell(hide_code=True)
def _(art, kind, mo, pl):
    mo.stop(kind != "training" or art.test_metrics is None)
    _tm = art.test_metrics
    _p = _tm["precision"]
    _r = _tm["recall"]
    _f1 = _tm["f1"]
    _mp = _tm["micro_precision"]
    _mr = _tm["micro_recall"]
    _ma = _tm["mean_offset_A"]
    _me = _tm["median_offset_A"]
    _nt = _tm["n_tiles"]
    _ne = _tm["n_empty"]
    _tdf = pl.DataFrame({
        "metric": ["precision", "recall", "F1", "micro-precision", "micro-recall",
                   "mean offset", "median offset", "tiles"],
        "value": [
            f"{_p:.4f}", f"{_r:.4f}", f"{_f1:.4f}", f"{_mp:.4f}", f"{_mr:.4f}",
            f"{_ma:.4f} Å  ({_ma * 100:.1f} pm)",
            f"{_me:.4f} Å  ({_me * 100:.1f} pm)",
            f"{_nt} ({_ne} empty)",
        ],
    })
    mo.vstack([mo.md("### Held-out test (best.pt)"), mo.ui.table(_tdf, selection=None, pagination=False)])
    return


@app.cell(hide_code=True)
def _(art, kind, mo, px):
    mo.stop(kind != "overfit")
    # plot every per-step metric column present (handles dice-era and regression-era runs)
    _terms = [c for c in art.losses.columns if c != "step"]
    _long = art.losses.unpivot(index="step", on=_terms, variable_name="term", value_name="value")
    _loss_fig = px.line(_long, x="step", y="value", color="term", title="Overfit metrics (per training step)")
    _loss_fig.update_layout(height=360, width=760, margin=dict(l=8, r=8, t=40, b=8))
    _loss_fig
    return


@app.cell(hide_code=True)
def _(art, go, kind, make_subplots, mo):
    mo.stop(kind != "overfit")
    _panels = [("input", art.sample["input"]), ("gt", art.sample["gt"]), ("pred", art.sample["pred"])]
    _pvg = make_subplots(rows=1, cols=3, subplot_titles=[t for t, _ in _panels], horizontal_spacing=0.04)
    for _c, (_t, _a) in enumerate(_panels, start=1):
        _pvg.add_trace(
            go.Heatmap(z=_a, colorscale=("gray" if _t == "input" else "viridis"), showscale=False),
            row=1, col=_c,
        )
    _pvg.update_xaxes(showticklabels=False, ticks="")
    _pvg.update_yaxes(showticklabels=False, ticks="", autorange="reversed")
    _pvg.update_layout(height=300, width=860, margin=dict(l=8, r=8, t=34, b=8), title="Overfit scene \u2014 input | gt | pred")
    _pvg
    return


@app.cell(hide_code=True)
def _(art, kind, mo, pl, px):
    mo.stop(kind != "overfit")
    _ok = art.profile.filter(~pl.col("oom"))
    _thr = px.bar(_ok, x="value", y="steps_per_s", color="knob", barmode="group", title="Throughput (steps/s) by sweep point")
    _thr.update_layout(height=340, width=760, margin=dict(l=8, r=8, t=40, b=8))
    mo.vstack([mo.ui.table(art.profile, selection=None), _thr])
    return


if __name__ == "__main__":
    app.run()
