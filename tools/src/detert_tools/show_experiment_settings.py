#!/usr/bin/env python3
"""show_experiment_settings.py — print a results directory's settings.

Reads `config.json` (run) or `sweep_config.json` (sweep) plus `llm_meta.json`
and renders them as a readable table, or as JSON with `--json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path) -> dict | None:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return None


def _find_config_file(results_dir: Path) -> tuple[Path, str]:
    run_cfg = results_dir / "config.json"
    sweep_cfg = results_dir / "sweep_config.json"
    if run_cfg.exists():
        return run_cfg, "run"
    if sweep_cfg.exists():
        return sweep_cfg, "sweep"
    raise FileNotFoundError(
        f"no settings file in: {results_dir}\n"
        f"  expected: config.json (run) or sweep_config.json (sweep / ablation)"
    )


def render_run_config(cfg: dict, source: Path) -> str:
    beta = cfg.get("beta", {})
    lines = [
        "=" * 70,
        "experiment settings (run)",
        "=" * 70,
        f"settings file: {source}",
        "-" * 70,
        f"llm_mode          : {cfg.get('llm_mode', '-')}",
        f"n_employees       : {cfg.get('n_employees', '-')} "
        f"({cfg.get('n_teams', '-')} teams × {cfg.get('team_size', '-')})",
        f"n_levels          : {cfg.get('n_levels', '-')}",
        f"network           : {cfg.get('network_kind', '-')} "
        f"(k={cfg.get('network_k', '-')}, β={cfg.get('network_beta', '-')})",
        f"ivt_mean ι̅        : {cfg.get('ivt_mean', '-')}",
        f"ivt_sd            : {cfg.get('ivt_sd', '-')}",
        f"ivt_weights       : {cfg.get('ivt_weights', '-')}",
        f"β_ι (IVT effect)  : {beta.get('beta_ivt', '-')}",
        f"β_ψ / β_f         : {beta.get('beta_psafety', '-')} / {beta.get('beta_fear', '-')}",
        f"p_retaliate       : {cfg.get('p_retaliate', '-')}",
        f"shock_t           : {cfg.get('shock_t', '-')}",
        f"t_max / runs      : {cfg.get('t_max', '-')} / {cfg.get('runs', '-')}",
        f"seed (core)       : {cfg.get('seed', '-')}",
        f"LLM temp / seed   : {cfg.get('llm_temperature', '-')} / {cfg.get('llm_seed', '-')}",
        f"output_dir        : {cfg.get('output_dir', '-')}",
        "=" * 70,
    ]
    return "\n".join(lines)


def render_sweep_config(cfg: dict, source: Path) -> str:
    lines = [
        "=" * 70,
        f"experiment settings ({cfg.get('command', 'sweep')})",
        "=" * 70,
        f"settings file: {source}",
        "-" * 70,
        f"llm_mode          : {cfg.get('llm_mode', cfg.get('modes', '-'))}",
        f"n_teams × team    : {cfg.get('n_teams', '-')} × {cfg.get('team_size', '-')}",
        f"β_ι values        : {cfg.get('beta_ivt_values', '-')}",
        f"ψ̄ values          : {cfg.get('psafety_mean_values', '-')}",
        f"seeds             : {cfg.get('seed_start', cfg.get('seed', '-'))}"
        f"..{cfg.get('seed_end', '')}",
        f"runs/cell         : {cfg.get('runs', '-')}",
        f"t_max             : {cfg.get('t_max', '-')}",
        "=" * 70,
    ]
    return "\n".join(lines)


def render_llm_meta(meta: dict) -> str:
    lines = [
        "LLM / determinism metadata",
        "-" * 70,
        f"llm_mode          : {meta.get('llm_mode', '-')}",
        f"model / endpoint  : {meta.get('llm_model', '-')} @ {meta.get('llm_endpoint', '-')}",
        f"temperature / seed: {meta.get('llm_temperature', '-')} / {meta.get('llm_seed', '-')}",
        f"LLM calls         : {meta.get('total_calls', '-')} "
        f"(cache-hit {meta.get('cache_hits', '-')}, "
        f"{100 * meta.get('cache_hit_rate', 0):.1f}%)",
        f"final_round       : {meta.get('final_round', '-')}",
        f"convergence_step  : {meta.get('convergence_step', '-')}",
        f"ever_silent_frac  : {meta.get('ever_silent_fraction', '-')}",
        "=" * 70,
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="detert-tools show-experiment-settings",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", "--results_dir", default="results/latest")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table.")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"error: directory does not exist: {results_dir}", file=sys.stderr)
        return 1

    try:
        cfg_path, kind = _find_config_file(results_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    cfg = _load(cfg_path)
    meta = _load(results_dir / "llm_meta.json")

    if args.json:
        payload = {"source": str(cfg_path), "kind": kind, "config": cfg, "llm_meta": meta}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if kind == "run":
            print(render_run_config(cfg, cfg_path))
        else:
            print(render_sweep_config(cfg, cfg_path))
        if meta is not None:
            print(render_llm_meta(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
