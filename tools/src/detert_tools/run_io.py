"""run ディレクトリの読み方をここ 1 箇所に集める．

置き場と同一性は runvault が持つので，どのスクリプトも `results/` を自分で
glob しない．対象の run は `runvault path` が返す．

移行後のレイアウトで変わったのは 3 点：

- `metrics.csv` が wide から long になった．ステップごとの値は
  [`metrics_table`] が wide に戻し，run 全体を 1 つの値で表すもの
  (`final_round` / `converged` / `silence_voice_corr_timeavg` など) は
  [`run_metrics`] が返す．
- 従業員 1 人 1 行の表は `artifacts/agents.csv` に移った (数とカテゴリが同居する
  表なので，指標に割らず表のまま置いてある)．
- `sweep_summary.csv` は無くなった．同じ表は子 run から組み直せる
  ([`sweep_table`])．`rule_activation.csv` も無い — 5 本の share は
  `metrics.csv` の `rule_*` と同じ数なので，1 つの数を 2 箇所に置かない．

runvault より前のレイアウト (フラットな `results/<stamp>/` と `results/latest`)
はディスクにまだ残っており，書き換えない．そちらも読めるようにしておく．
"""
from __future__ import annotations

import os

import pandas as pd
from runvault.read import (
    artifacts_dir,
    config_parameters,
    figures_dir,
    load_run_meta,
    metrics_wide,
    run_scope_metrics,
    run_subcommand,
    runvault_path,
    sweep_children,
)

__all__ = [
    "CHECK_EVENT",
    "EXPERIMENT",
    "TRIAL_EVENT",
    "agents_table",
    "is_runvault_run",
    "latest_run",
    "metrics_table",
    "output_dir",
    "run_metrics",
    "run_parameters",
    "subcommand",
    "sweep_table",
]

#: runvault 上の実験名．
EXPERIMENT = "detert"
#: `ablation` の試行 1 本を表す実験固有のイベント種別．
TRIAL_EVENT = "x.detert2011.trial"
#: `reproduce` の帯照合 1 件を表す実験固有のイベント種別．
CHECK_EVENT = "x.detert2011.check"

#: 掃引表の列．旧 `sweep_summary.csv` と同じ順にする．
SWEEP_COLUMNS = [
    "llm_mode",
    "beta_ivt",
    "psafety_mean",
    "run",
    "seed",
    "final_round",
    "upward_silence_rate",
    "silence_voice_corr",
    "max_rule_cooccurrence",
    "convergence_step",
]


def is_runvault_run(run_dir: str | os.PathLike) -> bool:
    """runvault が書いた run か（旧レイアウトなら False）．"""
    return load_run_meta(run_dir, required=False) is not None


def latest_run(
    results_root: str = "results",
    subcommand: str = "run",
    standalone: bool = True,
) -> str:
    """直近に完了した run のディレクトリ．

    `standalone` は sweep の子を除く．子は親と同じ `subcommand` (`run`) で走るので，
    これを外すと «最後に走った子» が返る．
    """
    return runvault_path(
        EXPERIMENT,
        results_root=results_root,
        subcommand=subcommand,
        standalone=standalone,
    )


def subcommand(run_dir: str | os.PathLike) -> str:
    """この run がどのサブコマンドの実行か．

    旧レイアウトは `run.json` を持たないので，どの設定ファイルがあるかで見分ける．
    """
    if is_runvault_run(run_dir):
        return run_subcommand(run_dir)
    if os.path.exists(os.path.join(str(run_dir), "sweep_config.json")):
        return "sweep"
    return "run"


def run_parameters(run_dir: str | os.PathLike) -> dict:
    """この run が回した条件（`config.json` の `parameters`）．

    旧レイアウトの sweep / ablation は `sweep_config.json` に持っている．
    """
    params = config_parameters(run_dir, required=False)
    if params is not None:
        return params
    legacy = os.path.join(str(run_dir), "sweep_config.json")
    if os.path.exists(legacy):
        import json

        with open(legacy, encoding="utf-8") as f:
            return json.load(f)
    return {}


def metrics_table(run_dir: str | os.PathLike) -> pd.DataFrame:
    """ステップごとの指標，1 ステップ 1 行．

    long 形式の `metrics.csv` を wide に戻す．旧レイアウトの時間軸は `t` だったので
    `step` に揃える（列名を 1 つにしないと，呼び出し側が経路ごとに分岐する）．
    """
    df = metrics_wide(os.path.join(str(run_dir), "metrics.csv"))
    if "t" in df.columns and "step" not in df.columns:
        df = df.rename(columns={"t": "step"})
    return df


def run_metrics(run_dir: str | os.PathLike) -> dict[str, float]:
    """run 全体を 1 つの値で表す指標．

    旧レイアウトでは `llm_meta.json` が持っていた値にあたる．収束しなかった run に
    `convergence_step` は無い（0 で埋めない）ので，`.get` で受けること．
    """
    if is_runvault_run(run_dir):
        return run_scope_metrics(run_dir)
    import json

    path = os.path.join(str(run_dir), "llm_meta.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    out = {
        "final_round": meta.get("final_round"),
        "ever_silent_fraction": meta.get("ever_silent_fraction"),
        "silence_voice_corr_timeavg": meta.get("silence_voice_corr"),
        "llm_calls": meta.get("total_calls"),
        "llm_cache_hits": meta.get("cache_hits"),
        "llm_cache_hit_rate": meta.get("cache_hit_rate"),
    }
    if meta.get("convergence_step") is not None:
        out["convergence_step"] = meta["convergence_step"]
        out["converged"] = 1.0
    else:
        out["converged"] = 0.0
    return {k: float(v) for k, v in out.items() if v is not None}


def agents_table(run_dir: str | os.PathLike) -> pd.DataFrame:
    """最終ステップの従業員 1 人 1 行の表．"""
    path = os.path.join(artifacts_dir(run_dir), "agents.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"agents.csv not found: {path}")
    return pd.read_csv(path)


def output_dir(run_dir: str | os.PathLike) -> str:
    """図と後処理の表の置き場．run が終わった後に作るものは run の外に置く．

    `manifest.csv` は `finish()` が確定させるので，後から `artifacts/` に足しても
    hash が付かない．
    """
    out = figures_dir(run_dir)
    os.makedirs(out, exist_ok=True)
    return out


def sweep_table(sweep_dir: str | os.PathLike) -> pd.DataFrame:
    """掃引 1 セル × 試行 1 行の表．

    runvault はこの表をディスクに持たない．親の `parameters` が格子の定義で，
    各子の `parameters` がそのセルの条件なので，子から組み直す．旧レイアウトでは
    `sweep_summary.csv` がそれにあたる．
    """
    if not is_runvault_run(sweep_dir):
        legacy = os.path.join(str(sweep_dir), "sweep_summary.csv")
        if not os.path.exists(legacy):
            raise FileNotFoundError(f"sweep_summary.csv not found: {legacy}")
        return pd.read_csv(legacy)

    children = sweep_children(sweep_dir)
    if not children:
        raise SystemExit(
            f"error: この sweep 親に属する子 run がありません: {sweep_dir}\n"
            "  子は lineage.parent_run_uid で親を指す．親子が同じ results ルート"
            "にあるか確認すること．"
        )
    rows = []
    for child in children:
        params = run_parameters(child)
        meta = load_run_meta(child) or {}
        rng = meta.get("rng") or {}
        scoped = run_metrics(child)
        steps = metrics_table(child)
        last = steps.iloc[-1]
        rows.append(
            {
                "llm_mode": params.get("llm_mode"),
                "beta_ivt": params.get("beta", {}).get("beta_ivt"),
                "psafety_mean": params.get("psafety_mean"),
                "run": rng.get("replicate_index"),
                "seed": rng.get("master_seed"),
                "final_round": scoped.get("final_round"),
                "upward_silence_rate": last["upward_silence_rate"],
                "silence_voice_corr": scoped.get("silence_voice_corr_timeavg"),
                "max_rule_cooccurrence": last["max_rule_cooccurrence"],
                # 収束しなかった run に収束ステップは無い．旧 CSV の欠測表現に揃える．
                "convergence_step": scoped.get("convergence_step", -1),
                "run_dir": child,
            }
        )
    return pd.DataFrame(rows, columns=SWEEP_COLUMNS + ["run_dir"])
