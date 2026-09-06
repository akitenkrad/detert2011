# CLI reference

[English](cli.md) | [日本語](cli.ja.md)

The `detert` binary has four subcommands. One invocation is one [runvault](https://github.com/akitenkrad/rs-runvault) run: the run directory under `--output-dir` (default `results`) is created and named by runvault, which also writes `run.json`, `config.json`, `metrics.csv`, `status.json` and `manifest.csv`. There is no timestamped directory of our own and no `results/latest` symlink.

`metrics.csv` is long and fixed — `run_uid,step,step_unit,scope,name,value`. The old per-step columns are metric names at `scope=run` with `step_unit=step`; values that describe the whole run (`final_round`, `converged`, `silence_voice_corr_timeavg`, …) carry no step. Resolve a run with `runvault path --experiment detert --latest`, never by globbing `results/`.

## `run`

Single configuration.

| Flag | Default | Meaning |
|------|---------|---------|
| `--llm-mode` | `rule` | `llm` / `rule` / `rule_no_ivt` (mutually exclusive). |
| `--n` | — | Total employees (overrides `n_teams × team_size`). |
| `--n-teams` | `8` | Number of teams. |
| `--team-size` | `25` | Employees per team. |
| `--n-levels` | `3` | Hierarchical levels. |
| `--network-model` | `watts-strogatz` | `watts-strogatz` / `erdos-renyi` / `barabasi-albert`. |
| `--network-k` | `6` | Watts–Strogatz k. |
| `--network-beta` | `0.1` | Watts–Strogatz β / Erdős–Rényi p. |
| `--ivt-mean` | `0.55` | Mean IVT strength ι̅. |
| `--ivt-sd` | `0.20` | Std-dev of IVT strength. |
| `--beta-psafety` | `1.2` | β_ψ (VOICE logit). |
| `--beta-fear` | `1.5` | β_f. |
| `--beta-ivt` | `2.0` | β_ι (the IVT main effect; calibrated HiCo point). |
| `--p-retaliate` | `0.05` | Per-agent per-step retaliation probability. |
| `--shock-t` | `24` | Exogenous σ-shock step. |
| `--t-max` | `60` | Maximum step. |
| `--runs` | `1` | Independent runs (output reflects the last). |
| `--seed` | `42` | Core-layer seed. |
| `--llm-temperature` | `0.0` | LLM temperature. |
| `--llm-seed` | `0` | LLM seed offset. |
| `--cache-path` | `.llm_cache/cache.json` | Prompt→response cache (LLM mode). |

Outputs: `config.json` (the condition, under `parameters`), `metrics.csv` (per-step metrics plus the run-scope ones), `artifacts/agents.csv` (final-step per-agent state — a table with categorical columns, so it stays a table), and, in `llm` mode, the `llm` block of `run.json` (provider / model / temperature).

With `--runs N` the recorded metrics are those of the last trial, as before; `runs` is in `parameters`.

## `sweep`

Cartesian product over `β_ι × ψ̄ × seeds`.

| Flag | Default | Meaning |
|------|---------|---------|
| `--llm-mode` | `rule` | Decision mode. |
| `--beta-ivt-min/max/step` | `0.0 / 1.6 / 0.2` | β_ι grid. |
| `--psafety-mean-values` | `0.3,0.5,0.7` | ψ̄ axis values. |
| `--runs` | `5` | Runs per cell. |
| `--t-max` | `60` | Maximum step. |
| `--seed` | `42` | Base seed. |

Outputs: a parent run (`subcommand=sweep`) holding the grid definition, plus one child run (`subcommand=run`) per cell × trial. The parent has no `master_seed` — it is driven by a list of seeds, and each child records the derived seed it actually used, with `replicate_index` separating repeats of one cell. `ψ̄` is not a `Config` field, so each child carries it as `parameters.psafety_mean`.

There is no `sweep_summary.csv`: the same table is rebuilt from the children by `detert_tools.run_io.sweep_table`.

## `ablation`

Contrast decision modes across a seed range.

| Flag | Default | Meaning |
|------|---------|---------|
| `--modes` | `rule,rule_no_ivt` | Comma-separated modes. |
| `--seed-start` / `--seed-end` | `0` / `30` | Inclusive seed range. |
| `--t-max` | `60` | Maximum step. |

Outputs: **one** run holding every arm. The IVT main effect is a difference between two arms measured in a single execution, so splitting the arms into child runs would put the claim outside every run.

Each trial is an `x.detert2011.trial` line in `events.jsonl` (mode, seed, and the three observables — they have no time axis, so they cannot be rows of `metrics.csv`). `metrics.csv` carries the per-mode means (`<mode>_mean_upward_silence_rate`, `<mode>_n_trials`) and the comparison itself (`ivt_effect_delta`, `ivt_effect_cohens_d`).

## `reproduce`

Per-mode steady-state report against the design anchors (HiCo ≈ .50, silence–voice r ≈ −.55, discriminant co-occurrence < .50). For the full Table-4-style report **plus** CFA-style fit indices reproduced from the ABM rule-firing matrix, use the Python tool `detert-tools reproduce`.

Outputs: the three observed means and `checks_passed` / `checks_total` as run-scope metrics, and one `x.detert2011.check` event per anchor carrying the band and the verdict (a verdict is a category, not a number). The bands are anchors this replication chose, not values the paper reports, so they are **not** written to `reference.csv`.
