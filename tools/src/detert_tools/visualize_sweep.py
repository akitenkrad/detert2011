#!/usr/bin/env python3
"""visualize_sweep.py — sweep visualization for the Detert 2011 IVT silence model.

Reads a sweep parent run and produces:
  - sweep_phase_diagram.png  : β_ι × ψ̄ heatmap of mean upward silence rate
  - sweep_beta_ivt_curve.png : upward silence vs β_ι (one line per ψ̄), with the
                               HiCo .50 anchor marked

セルごとの値は子 run にあり，runvault は `sweep_summary.csv` をディスクに持たない．
表は `run_io.sweep_table` が子から組み直す．--results-dir を省略すると
`runvault path --experiment detert --latest --subcommand sweep` が返す run を使う．

Usage:
    uv run detert-tools visualize-sweep
    uv run detert-tools visualize-sweep --results-dir <sweep run>
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from detert_tools.run_io import latest_run, output_dir as default_output_dir, sweep_table

COLOR_BG = "#FAFAF8"
LINE_COLORS = ["#534AB7", "#4C97C9", "#0F6E56", "#F4A259", "#B5546A"]


def plot_phase_diagram(df: pd.DataFrame, output_dir: str) -> None:
    pivot = df.pivot_table(
        index="psafety_mean", columns="beta_ivt", values="upward_silence_rate", aggfunc="mean"
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(COLOR_BG)
    im = ax.imshow(pivot.values, cmap="magma", origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
    ax.set_xlabel("β_ι (IVT main-effect coefficient)")
    ax.set_ylabel("ψ̄ (psychological-safety axis)")
    ax.set_title("Phase diagram: mean upward silence rate over β_ι × ψ̄")
    fig.colorbar(im, ax=ax, label="upward silence rate")
    fig.tight_layout()
    out = os.path.join(output_dir, "sweep_phase_diagram.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[visualize-sweep] wrote {out}")


def plot_beta_ivt_curve(df: pd.DataFrame, output_dir: str) -> None:
    grouped = df.groupby(["psafety_mean", "beta_ivt"])["upward_silence_rate"].mean().reset_index()
    if grouped.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    for i, (psi, sub) in enumerate(grouped.groupby("psafety_mean")):
        sub = sub.sort_values("beta_ivt")
        ax.plot(
            sub["beta_ivt"],
            sub["upward_silence_rate"],
            marker="o",
            color=LINE_COLORS[i % len(LINE_COLORS)],
            label=f"ψ̄={psi:.2f}",
        )
    ax.axhline(0.50, color="gray", ls="--", lw=1, label="HiCo anchor .50")
    ax.set_xlabel("β_ι (IVT main-effect coefficient)")
    ax.set_ylabel("mean upward silence rate")
    ax.set_ylim(0, 1)
    ax.set_title("IVT main effect: upward silence rises with β_ι")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(output_dir, "sweep_beta_ivt_curve.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[visualize-sweep] wrote {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="detert-tools visualize-sweep")
    parser.add_argument("--results-dir", "--results_dir", default=None)
    parser.add_argument("--results-root", "--results_root", default="results")
    parser.add_argument("--output-dir", "--output_dir", default=None)
    args = parser.parse_args(argv)

    results_dir = args.results_dir or latest_run(
        results_root=args.results_root, subcommand="sweep", standalone=False
    )
    output_dir = args.output_dir or default_output_dir(results_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[visualize-sweep] sweep: {results_dir}")
    df = sweep_table(results_dir)
    plot_phase_diagram(df, output_dir)
    plot_beta_ivt_curve(df, output_dir)


if __name__ == "__main__":
    main()
