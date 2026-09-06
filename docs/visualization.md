# Visualization

[English](visualization.md) | [日本語](visualization.ja.md)

The Python `detert-tools` package reads a runvault run directory and renders PNGs. Install with `uv sync`, then run any subcommand.

Every subcommand resolves its run with `runvault path` when `--results-dir` is omitted, and writes into `<results-root>/detert/figures/<run_slug>/` — outside the run, because `manifest.csv` is settled by `finish()` and anything added afterwards would carry no hash. Directories written before the migration are still readable.

## `visualize`

Single-run plots from `metrics.csv` and `artifacts/agents.csv`:

- **silence_timeseries.png** — upward silence rate, silence rate, and climate of silence per step, with the HiCo .50 anchor line.
- **rule_firing_heatmap.png** — the five IVT rules' activation share `a_r` over time as a heatmap, from the `rule_*` metrics (the same numbers the old `rule_activation.csv` held).
- **silence_voice_scatter.png** — final-step expression (VOICE / SILENCE / NEUTRAL) plotted over IVT strength ι × private concern b.

```bash
uv run detert-tools visualize            # or --results-dir <run>
```

## `visualize-sweep`

Sweep plots from the sweep parent's children (runvault keeps no `sweep_summary.csv`; the table is rebuilt):

- **sweep_phase_diagram.png** — mean upward silence rate across the β_ι × ψ̄ grid (heatmap).
- **sweep_beta_ivt_curve.png** — upward silence vs β_ι, one line per ψ̄, with the HiCo .50 anchor.

```bash
uv run detert-tools visualize-sweep      # or --results-dir <sweep run>
```

## `show-experiment-settings`

Pretty-prints the run's `parameters`, the `llm` block of `run.json` and the run-scope metrics; `--json` emits machine-readable JSON. `--subcommand` picks which kind of run to resolve (`run` / `sweep` / `ablation` / `reproduce`).

```bash
uv run detert-tools show-experiment-settings   # or --results-dir <run>
```

## `reproduce`

See [Reproduction](reproduction.md).
