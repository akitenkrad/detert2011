#!/usr/bin/env python3
"""show_experiment_settings.py — print a run directory's settings.

`config.json` の `parameters`（この run が回した条件）と，run 全体を 1 つの値で表す
指標・`run.json` の `llm` ブロックを読んで表に整える．`--json` で機械可読 JSON．

移行前の `llm_meta.json` は無い．モデル名・endpoint・温度は `run.json` の `llm`
ブロックが，呼び出し回数と cache-hit は `metrics.csv` の run スコープ指標が持つ．
旧レイアウトのディレクトリを渡した場合は `llm_meta.json` から同じ値を読む．
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runvault.read import load_run_meta

from detert_tools.run_io import latest_run, run_metrics, run_parameters, subcommand


def _fmt(value: object, default: str = "-") -> str:
    return default if value is None else str(value)


def render_run_config(cfg: dict, source: Path) -> str:
    beta = cfg.get("beta", {})
    lines = [
        "=" * 70,
        "experiment settings (run)",
        "=" * 70,
        f"settings file: {source}",
        "-" * 70,
        f"llm_mode          : {_fmt(cfg.get('llm_mode'))}",
        f"n_employees       : {_fmt(cfg.get('n_employees'))} "
        f"({_fmt(cfg.get('n_teams'))} teams × {_fmt(cfg.get('team_size'))})",
        f"n_levels          : {_fmt(cfg.get('n_levels'))}",
        f"network           : {_fmt(cfg.get('network_kind'))} "
        f"(k={_fmt(cfg.get('network_k'))}, β={_fmt(cfg.get('network_beta'))})",
        f"ivt_mean ι̅        : {_fmt(cfg.get('ivt_mean'))}",
        f"ivt_sd            : {_fmt(cfg.get('ivt_sd'))}",
        f"ivt_weights       : {_fmt(cfg.get('ivt_weights'))}",
        f"β_ι (IVT effect)  : {_fmt(beta.get('beta_ivt'))}",
        f"β_ψ / β_f         : {_fmt(beta.get('beta_psafety'))} / {_fmt(beta.get('beta_fear'))}",
        f"p_retaliate       : {_fmt(cfg.get('p_retaliate'))}",
        f"shock_t           : {_fmt(cfg.get('shock_t'))}",
        f"t_max / runs      : {_fmt(cfg.get('t_max'))} / {_fmt(cfg.get('runs'))}",
        f"seed (core)       : {_fmt(cfg.get('seed'))}",
        f"LLM temp / seed   : {_fmt(cfg.get('llm_temperature'))} / {_fmt(cfg.get('llm_seed'))}",
        "=" * 70,
    ]
    if "psafety_mean" in cfg:
        lines.insert(-1, f"ψ̄ (sweep のセル)  : {cfg['psafety_mean']}")
    return "\n".join(lines)


def render_grid_config(cfg: dict, kind: str, source: Path) -> str:
    lines = [
        "=" * 70,
        f"experiment settings ({kind})",
        "=" * 70,
        f"settings file: {source}",
        "-" * 70,
        f"llm_mode          : {_fmt(cfg.get('llm_mode', cfg.get('modes')))}",
        f"n_teams × team    : {_fmt(cfg.get('n_teams'))} × {_fmt(cfg.get('team_size'))}",
        f"β_ι values        : {_fmt(cfg.get('beta_ivt_values'))}",
        f"ψ̄ values          : {_fmt(cfg.get('psafety_mean_values'))}",
        f"seeds             : {_fmt(cfg.get('seed_start', cfg.get('seed')))}"
        f"..{cfg.get('seed_end', '')}",
        f"runs/cell         : {_fmt(cfg.get('runs'))}",
        f"t_max             : {_fmt(cfg.get('t_max'))}",
        "=" * 70,
    ]
    return "\n".join(lines)


def render_provenance(cfg: dict, llm: dict | None, scoped: dict) -> str:
    calls = scoped.get("llm_calls")
    hits = scoped.get("llm_cache_hits")
    rate = scoped.get("llm_cache_hit_rate")
    lines = [
        "LLM / determinism metadata",
        "-" * 70,
        f"llm_mode          : {_fmt(cfg.get('llm_mode'))}",
    ]
    if llm:
        lines.append(
            f"provider / model  : {_fmt(llm.get('provider'))} / {_fmt(llm.get('model_snapshot'))}"
        )
        lines.append(f"temperature       : {_fmt(llm.get('temperature'))}")
    else:
        # 規則モードは LLM を 1 度も呼ばないので `llm` ブロックを持たない．
        lines.append("provider / model  : - (規則モードは LLM を呼ばない)")
    if calls is not None:
        rate_txt = "-" if rate is None else f"{100 * rate:.1f}%"
        lines.append(f"LLM calls         : {calls:.0f} (cache-hit {hits:.0f}, {rate_txt})")
    for key, label in (
        ("final_round", "final_round       "),
        ("converged", "converged         "),
        ("convergence_step", "convergence_step  "),
        ("ever_silent_fraction", "ever_silent_frac  "),
        ("silence_voice_corr_timeavg", "silence_voice_corr"),
    ):
        if key in scoped:
            lines.append(f"{label}: {scoped[key]}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="detert-tools show-experiment-settings",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", "--results_dir", default=None)
    parser.add_argument("--results-root", "--results_root", default="results")
    parser.add_argument("--subcommand", default="run", help="--results-dir 省略時に探す run の種別")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table.")
    args = parser.parse_args(argv)

    results_dir = args.results_dir or latest_run(
        results_root=args.results_root,
        subcommand=args.subcommand,
        standalone=args.subcommand == "run",
    )
    path = Path(results_dir)
    if not path.exists():
        print(f"error: directory does not exist: {path}", file=sys.stderr)
        return 1

    cfg = run_parameters(path)
    if not cfg:
        print(f"error: no settings in: {path}", file=sys.stderr)
        return 1
    kind = subcommand(path)
    meta = load_run_meta(path, required=False) or {}
    llm = meta.get("llm")
    scoped = run_metrics(path)

    if args.json:
        payload = {
            "run_dir": str(path),
            "subcommand": kind,
            "parameters": cfg,
            "llm": llm,
            "run_metrics": scoped,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    source = path / "config.json"
    if kind == "run":
        print(render_run_config(cfg, source))
    else:
        print(render_grid_config(cfg, kind, source))
    print(render_provenance(cfg, llm, scoped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
