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

    from runs import list_runs, load_run

    RUNS_ROOT = Path(__file__).parent.parent / "runs"
    return RUNS_ROOT, go, list_runs, load_run, make_subplots, mo, pl, px


@app.cell(hide_code=True)
def _(RUNS_ROOT, list_runs, mo):
    _runs = list_runs(RUNS_ROOT)
    run_picker = mo.ui.dropdown(
        options={p.name: str(p) for p in _runs},
        value=(_runs[-1].name if _runs else None),
        label="run",
    )
    run_picker if _runs else mo.callout(
        mo.md(f"No runs found under `{RUNS_ROOT}`. Pull a run from Drive into `runs/` first."),
        kind="warn",
    )
    return (run_picker,)


@app.cell
def _(load_run, mo, run_picker):
    mo.stop(not run_picker.value, mo.md("Select a run above."))
    art = load_run(run_picker.value)
    art.meta
    return (art,)


@app.cell(hide_code=True)
def _(art, mo):
    _m = art.meta
    _enc = art.config.model.encoder
    _dec = art.config.model.decoder
    _syn = art.config.data.synthetic
    mo.md(
        f"""
    ### Run `{_m['run_id']}` · {_m['gpu_name']} · git `{_m['git_sha']}` · torch {_m['torch_version']}
    *{_m['purpose']} · {_m['created']}*

    | encoder | decoder | data |
    |---|---|---|
    | base={_enc.base_channels}, depth={_enc.depth}, feat={_enc.feature_dim} | hidden={_dec.hidden_dim}, layers={_dec.num_layers}, LE={_dec.local_ensemble}, unfold={_dec.feature_unfold}, cell={_dec.cell_decode} | crop={_syn.crop_size}, sample_q={_syn.sample_q}, label={_syn.label_kind} |
    """
    )
    return


@app.cell(hide_code=True)
def _(art, px):
    _long = art.losses.unpivot(index="step", on=["loss", "bce", "dice"], variable_name="term", value_name="value")
    _loss_fig = px.line(_long, x="step", y="value", color="term", title="Overfit loss (per training step)")
    _loss_fig.update_layout(height=360, width=760, margin=dict(l=8, r=8, t=40, b=8))
    _loss_fig
    return


@app.cell(hide_code=True)
def _(art, go, make_subplots):
    _panels = [("input", art.sample["input"]), ("gt", art.sample["gt"]), ("pred", art.sample["pred"])]
    _pvg = make_subplots(rows=1, cols=3, subplot_titles=[t for t, _ in _panels], horizontal_spacing=0.04)
    for _c, (_t, _a) in enumerate(_panels, start=1):
        _pvg.add_trace(
            go.Heatmap(z=_a, colorscale=("gray" if _t == "input" else "viridis"), showscale=False),
            row=1, col=_c,
        )
    _pvg.update_xaxes(showticklabels=False, ticks="")
    _pvg.update_yaxes(showticklabels=False, ticks="", autorange="reversed")
    _pvg.update_layout(height=300, width=860, margin=dict(l=8, r=8, t=34, b=8), title="Overfit scene — input | gt | pred")
    _pvg
    return


@app.cell(hide_code=True)
def _(art, mo, pl, px):
    _ok = art.profile.filter(~pl.col("oom"))
    _thr = px.bar(_ok, x="value", y="steps_per_s", color="knob", barmode="group", title="Throughput (steps/s) by sweep point")
    _thr.update_layout(height=340, width=760, margin=dict(l=8, r=8, t=40, b=8))
    mo.vstack([mo.ui.table(art.profile, selection=None), _thr])
    return


if __name__ == "__main__":
    app.run()
