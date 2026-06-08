import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Run comparison — INR-UNet vs. plain UNet baseline

    Overlays training-run bundles from `runs/` so the resolution-agnostic LIIF query head can be
    weighed against a fixed-resolution control. The two headline runs share a byte-identical data
    pipeline (same render cache, splits, encoder, `GaussianField` label, loss, and schedule) and
    differ only in `model.name` — so any gap is the decoder, not the data. Pick any subset of
    training runs below.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from runs import list_runs, load_training_run, run_kind

    RUNS_ROOT = Path(__file__).parent.parent / "runs"

    # stable qualitative palette, assigned per selected run
    PALETTE = ["#2a9d8f", "#e76f51", "#264653", "#8338ec", "#3a86ff", "#e9c46a", "#ff006e"]
    return (
        PALETTE,
        RUNS_ROOT,
        go,
        list_runs,
        load_training_run,
        make_subplots,
        mo,
        pl,
        px,
        run_kind,
    )


@app.cell(hide_code=True)
def _(RUNS_ROOT, list_runs, mo, run_kind):
    _train_runs = [p for p in list_runs(RUNS_ROOT) if run_kind(p) == "training"]
    _opts = {p.name: str(p) for p in _train_runs}
    run_picker = mo.ui.multiselect(
        options=_opts,
        value=list(_opts.keys()),
        label="training runs to compare",
    )
    run_picker if _train_runs else mo.callout(
        mo.md(f"No training runs under `{RUNS_ROOT}`. Pull one from Drive into `runs/` first."),
        kind="warn",
    )
    return (run_picker,)


@app.cell
def _(PALETTE, load_training_run, mo, run_picker):
    mo.stop(not run_picker.value, mo.md("Select at least one training run above."))

    arts = [load_training_run(p) for p in run_picker.value]

    # unique display label per run (model name; disambiguated by date prefix when names collide)
    _name_counts = {}
    for _a in arts:
        _nm = str(_a.config.model.get("name", _a.meta["run_id"]))
        _name_counts[_nm] = _name_counts.get(_nm, 0) + 1


    def _label(a):
        nm = str(a.config.model.get("name", a.meta["run_id"]))
        if _name_counts[nm] > 1:
            date = a.meta["run_id"].split("-")[0]
            return f"{nm} ({date})"
        return nm


    labels = {a.meta["run_id"]: _label(a) for a in arts}
    colors = {labels[a.meta["run_id"]]: PALETTE[i % len(PALETTE)] for i, a in enumerate(arts)}

    mo.md("Comparing **" + str(len(arts)) + "** run(s): " + ", ".join(f"`{labels[a.meta['run_id']]}`" for a in arts))

    return arts, colors, labels


@app.cell(hide_code=True)
def _(arts, labels, mo, pl):
    def _summary_row(a):
        m = a.meta
        tm = a.test_metrics or {}
        h = a.history
        return {
            "run": labels[m["run_id"]],
            "model": str(a.config.model.get("name", "?")),
            "git": m["git_sha"],
            "epochs": m["epochs_run"],
            "best epoch": m["best_epoch"],
            "best val F1": round(m["best_val_f1"], 4),
            "test F1": round(tm["f1"], 4) if "f1" in tm else None,
            "test mean off (pm)": round(tm["mean_offset_A"] * 100, 1) if "mean_offset_A" in tm else None,
            "test median off (pm)": round(tm["median_offset_A"] * 100, 1) if "median_offset_A" in tm else None,
            "mean epoch (s)": round(h["epoch_time_s"].mean(), 1),
        }


    summary = pl.DataFrame([_summary_row(_a) for _a in arts])
    mo.vstack([mo.md("### Runs at a glance"), mo.ui.table(summary, selection=None, pagination=False)])

    return


@app.cell(hide_code=True)
def _(arts, colors, go, labels, make_subplots, mo, pl):
    _fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("validation F1", "validation localization offset (pm)"),
        horizontal_spacing=0.09,
    )
    for _a in arts:
        _rid = _a.meta["run_id"]
        _lbl = labels[_rid]
        _col = colors[_lbl]
        _h = _a.history
        _be = _a.meta["best_epoch"]
        _bf = _h.filter(pl.col("epoch") == _be)["val_f1"][0]
        _fig.add_trace(
            go.Scatter(x=_h["epoch"], y=_h["val_f1"], mode="lines", name=_lbl,
                       legendgroup=_rid, line=dict(color=_col)),
            row=1, col=1,
        )
        _fig.add_trace(
            go.Scatter(x=[_be], y=[_bf], mode="markers", showlegend=False, legendgroup=_rid,
                       marker=dict(color=_col, size=11, symbol="star")),
            row=1, col=1,
        )
        _fig.add_trace(
            go.Scatter(x=_h["epoch"], y=_h["val_mean_offset_A"] * 100, mode="lines",
                       showlegend=False, legendgroup=_rid, line=dict(color=_col)),
            row=1, col=2,
        )
        _fig.add_trace(
            go.Scatter(x=_h["epoch"], y=_h["val_median_offset_A"] * 100, mode="lines",
                       showlegend=False, legendgroup=_rid, line=dict(color=_col, dash="dot")),
            row=1, col=2,
        )
    _fig.update_xaxes(title_text="epoch")
    _fig.update_yaxes(title_text="F1", row=1, col=1)
    _fig.update_yaxes(title_text="offset (pm)", row=1, col=2)
    _fig.update_layout(
        height=400, width=1000, margin=dict(l=8, r=8, t=54, b=8),
        title="Training curves — stars = best epoch; offset solid = mean, dotted = median",
        legend=dict(orientation="h", y=-0.2),
    )
    mo.vstack([
        _fig,
        mo.md(
            "*Both curves are the **dense** localization eval (`peak_localization` on the sigmoid "
            "heatmap) — identical and directly comparable across both models. `train_loss`/`val_loss` "
            "are **not** plotted here: INR-UNet scores MSE on random query points while the baseline "
            "scores it on the dense grid, so their loss magnitudes are not comparable. Model selection "
            "uses `val_f1`, which is.*"
        ),
    ])

    return


@app.cell(hide_code=True)
def _(arts, colors, labels, mo, pl, px):
    mo.stop(
        not any(a.test_metrics for a in arts),
        mo.md("No held-out test metrics in the selected runs."),
    )

    _rate_metrics = ["precision", "recall", "f1", "micro_precision", "micro_recall"]
    _rate_rows, _off_rows = [], []
    for _a in arts:
        _tm = _a.test_metrics
        if not _tm:
            continue
        _lbl = labels[_a.meta["run_id"]]
        for _m in _rate_metrics:
            _rate_rows.append({"run": _lbl, "metric": _m, "value": _tm[_m]})
        _off_rows.append({"run": _lbl, "metric": "mean", "value": _tm["mean_offset_A"] * 100})
        _off_rows.append({"run": _lbl, "metric": "median", "value": _tm["median_offset_A"] * 100})

    _rate_bar = px.bar(
        pl.DataFrame(_rate_rows), x="metric", y="value", color="run", barmode="group",
        color_discrete_map=colors, title="Held-out test — detection rates (higher is better)",
    )
    _rate_bar.update_yaxes(range=[0, 1])
    _rate_bar.update_layout(height=360, width=620, margin=dict(l=8, r=8, t=44, b=8))

    _off_bar = px.bar(
        pl.DataFrame(_off_rows), x="metric", y="value", color="run", barmode="group",
        color_discrete_map=colors, title="Held-out test — localization offset (pm, lower is better)",
    )
    _off_bar.update_yaxes(title_text="offset (pm)")
    _off_bar.update_layout(height=360, width=420, margin=dict(l=8, r=8, t=44, b=8))

    # tidy side-by-side table: one column per run, one row per metric
    _tbl = {"metric": ["precision", "recall", "F1", "micro-precision", "micro-recall",
                       "mean offset (pm)", "median offset (pm)", "tiles (empty)"]}
    for _a in arts:
        _tm = _a.test_metrics
        if not _tm:
            continue
        _tbl[labels[_a.meta["run_id"]]] = [
            f"{_tm['precision']:.4f}", f"{_tm['recall']:.4f}", f"{_tm['f1']:.4f}",
            f"{_tm['micro_precision']:.4f}", f"{_tm['micro_recall']:.4f}",
            f"{_tm['mean_offset_A'] * 100:.1f}", f"{_tm['median_offset_A'] * 100:.1f}",
            f"{_tm['n_tiles']} ({_tm['n_empty']})",
        ]

    mo.vstack([
        mo.md("### Held-out test (best.pt)"),
        mo.hstack([_rate_bar, _off_bar], justify="start", gap=1),
        mo.ui.table(pl.DataFrame(_tbl), selection=None, pagination=False),
    ])

    return


if __name__ == "__main__":
    app.run()
