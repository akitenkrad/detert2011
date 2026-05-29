#!/usr/bin/env python3
"""visualize.py — single-run visualization for the Detert 2011 IVT silence model.

Reads `results/latest` (or `--results-dir`) and produces:
  - silence_timeseries.png    : upward silence rate + climate of silence per step
  - rule_firing_heatmap.png   : IVT 5-rule activation share over time (heatmap)
  - silence_voice_scatter.png : per-agent IVT strength vs silence/voice expression

Usage:
    uv run detert-tools visualize
    uv run detert-tools visualize --results-dir results/latest --output-dir out
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_BG = "#FAFAF8"
RULES = ["target_id", "need_data", "no_bypass", "no_embarrass", "career_consq"]
RULE_COLS = [f"rule_{r}" for r in RULES]


def load_config(results_dir: str) -> dict | None:
    path = os.path.join(results_dir, "config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def plot_silence_timeseries(results_dir: str, output_dir: str, cfg: dict | None) -> None:
    path = os.path.join(results_dir, "metrics.csv")
    if not os.path.exists(path):
        print(f"[visualize] no metrics.csv at {results_dir}; skipping time series")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.plot(df["t"], df["upward_silence_rate"], color="#534AB7", lw=2.2, label="upward silence rate")
    if "silence_rate" in df.columns:
        ax.plot(df["t"], df["silence_rate"], color="#4C97C9", lw=1.6, ls="--", label="silence rate")
    if "climate_of_silence" in df.columns:
        ax.plot(df["t"], df["climate_of_silence"], color="#0F6E56", lw=1.6, ls=":", label="climate of silence")
    ax.axhline(0.50, color="gray", lw=1, ls="-.", label="HiCo anchor .50")
    ax.set_xlabel("step t")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    title = "Upward silence over time"
    if cfg:
        title += f"  (llm_mode={cfg.get('llm_mode')}, β_ι={cfg.get('beta', {}).get('beta_ivt')})"
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(output_dir, "silence_timeseries.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[visualize] wrote {out}")


def plot_rule_heatmap(results_dir: str, output_dir: str) -> None:
    path = os.path.join(results_dir, "rule_activation.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        pivot = df.pivot_table(index="rule", columns="t", values="share", aggfunc="mean")
        pivot = pivot.reindex(RULES)
    else:
        path = os.path.join(results_dir, "metrics.csv")
        if not os.path.exists(path):
            return
        df = pd.read_csv(path)
        if not all(c in df.columns for c in RULE_COLS):
            return
        pivot = pd.DataFrame({r: df[c] for r, c in zip(RULES, RULE_COLS)}, index=df["t"]).T
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(COLOR_BG)
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma", origin="lower", vmin=0.0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("step t")
    ax.set_title("IVT rule activation share over time")
    fig.colorbar(im, ax=ax, label="activation share a_r")
    fig.tight_layout()
    out = os.path.join(output_dir, "rule_firing_heatmap.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[visualize] wrote {out}")


def plot_silence_voice_scatter(results_dir: str, output_dir: str) -> None:
    path = os.path.join(results_dir, "agents.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    colors = {"silence": "#534AB7", "voice": "#F4A259", "neutral": "#999999"}
    for expr, color in colors.items():
        sub = df[df["expression"] == expr]
        if sub.empty:
            continue
        jitter = np.random.default_rng(0).normal(0, 0.01, len(sub))
        ax.scatter(
            sub["ivt"],
            sub["private_concern"] + jitter,
            s=24,
            alpha=0.7,
            color=color,
            label=expr,
            edgecolors="none",
        )
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xlabel("IVT strength ι")
    ax.set_ylabel("private concern b")
    ax.set_title("Final-step expression by IVT strength × private concern")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(output_dir, "silence_voice_scatter.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[visualize] wrote {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="detert-tools visualize")
    parser.add_argument("--results-dir", default="results/latest")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)
    cfg = load_config(results_dir)
    plot_silence_timeseries(results_dir, output_dir, cfg)
    plot_rule_heatmap(results_dir, output_dir)
    plot_silence_voice_scatter(results_dir, output_dir)


if __name__ == "__main__":
    main()
