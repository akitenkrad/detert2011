//! runvault への記録の共通部分．
//!
//! 論文メタデータ (research) は `run` / `sweep` / `ablation` / `reproduce` の
//! どのサブコマンドでも同一なので，ここ 1 箇所で組み立てる．集団指標の long 形式へ
//! の落とし方，`agents.csv` の置き場，`ablation` の試行行と `reproduce` の帯照合の
//! 書き方もここに集める．

use std::path::Path;

use runvault::{Llm, Replication, Run, Target, Work};
use serde::Serialize;

use crate::simulation::{MetricsRow, SimulationResult};

/// runvault 上の実験名．`runvault path --experiment` に渡す値でもある．
pub const EXPERIMENT: &str = "detert";
/// リポジトリの安定 id．git remote の名前とは独立に固定する．
pub const REPO_ID: &str = "detert2011";
/// 分野．`simulation` を名乗ると `master_seed` が必須になる．
///
/// `--llm-mode llm` では LLM が発話/沈黙を決めるが，測っているのはモデルの安全性では
/// なく組織網上の沈黙の伝播なので `llm-safety` ではない．LLM 側の同一性は `llm`
/// ブロック ([`llm_block`]) が持つ．
pub const DOMAIN: &str = "simulation";

/// 時間軸の単位．モデルの刻みは socsim の 6 相ループ 1 周で，論文モデルの $t$ その
/// ものである．runvault の語彙では `step`．
const T_UNIT: &str = "step";

/// 指標の粒度．集団指標はどれも従業員全体の集約なので `run`．
const SCOPE: &str = "run";

/// `ablation` の試行 1 本を表す実験固有のイベント種別．
pub const TRIAL_EVENT: &str = "x.detert2011.trial";
/// `reproduce` の帯照合 1 件を表す実験固有のイベント種別．
pub const CHECK_EVENT: &str = "x.detert2011.check";

/// この再現実験が対象としている論文．
///
/// AMJ 掲載論文なので同定は DOI で行う．論文が持つのは調査 (Study 1–4) であって
/// シミュレーションの図表ではないため，`Target::table` / `Target::figure` ではなく
/// 主張そのもの (`Target::claim`) を対象にする．3 つの claim は `reproduce` が
/// 照合する 3 指標に対応する．
pub fn replication() -> Replication {
    Work::doi("10.5465/AMJ.2011.61967925")
        .title("Implicit Voice Theories: Taken-for-Granted Rules of Self-Censorship at Work")
        .year(2011)
        .source_version("published")
        .target(Target::claim(
            "upward-silence",
            "Employees who hold an upward concern withhold it at a high rate",
        ))
        .target(Target::claim(
            "silence-voice-distinct",
            "Silence and voice are distinct constructs, not two ends of one scale",
        ))
        .target(Target::claim(
            "ivt-rule-discriminance",
            "The five implicit voice theories fire as discriminant rules rather than one factor",
        ))
        .obsidian_note("研究/98_論文レポート/80-再現実験/実装完了/detert2011/設計書.md")
}

// ---------------------------------------------------------------------------
// LLM ブロック
// ---------------------------------------------------------------------------

/// 実際に応答したバックエンドを `llm` ブロックに落とす．
///
/// `model` / `endpoint` はクライアントが名乗った値をそのまま使う．`provider` は
/// runvault の語彙ではなく自由記述なので，endpoint から «どのゲートウェイが答えたか»
/// を決める．推測しているのは分類だけで，値そのものは記録から採る．
///
/// 規則モード (`rule` / `rule_no_ivt`) は LLM を 1 度も呼ばない．そのときこの関数は
/// 呼ばれず，`llm` ブロックごと書かれない — 呼んでいないモデルの名前を書かないため．
pub fn llm_block(model: &str, endpoint: &str, temperature: f32) -> Llm {
    let provider = if endpoint.contains("openai") {
        "openai"
    } else {
        "ollama"
    };
    Llm {
        provider: provider.to_string(),
        model_snapshot: model.to_string(),
        temperature: Some(temperature as f64),
        // プロンプトは従業員のペルソナと局所文脈から毎回組み立てられ，固定の
        // system prompt を持たない．無いものを hash しない．
        system_prompt_hash: None,
    }
}

// ---------------------------------------------------------------------------
// シミュレーション 1 本
// ---------------------------------------------------------------------------

/// シミュレーション 1 本ぶんの記録．
///
/// ステップごとの 11 指標 (`t` は時間軸なので値としては書かない) と，run 全体を
/// 1 つの値で表す収束・LLM 呼び出しの内訳を書く．実行時間は `status.json` の
/// `duration_sec` が正本なので指標にはしない．
pub fn log_simulation(run: &mut Run, result: &SimulationResult) {
    for m in &result.metrics_rows {
        log_step(run, m);
    }

    let mut aggregates: Vec<(&str, f64)> = vec![
        ("final_round", result.final_round as f64),
        (
            "converged",
            if result.convergence_step.is_some() {
                1.0
            } else {
                0.0
            },
        ),
        ("ever_silent_fraction", result.ever_silent_fraction),
        // ステップごとの `silence_voice_corr` は «その時刻の» VOICE/SILENCE 二分の
        // 相関で，run レベルの時間平均値とは別物なので名前を分ける．同じ名前だと
        // step の有無だけが両者の違いになり，後から取り違える．
        (
            "silence_voice_corr_timeavg",
            result.silence_voice_corr_timeavg,
        ),
        ("llm_calls", result.metadata.total() as f64),
        ("llm_cache_hits", result.metadata.cache_hits() as f64),
    ];
    // 収束しなかった run に収束ステップは無い．0 を書くと «0 歩目で収束した» と
    // 区別できなくなるので行ごと書かない (`converged` が 0 であることが答え)．
    if let Some(step) = result.convergence_step {
        aggregates.push(("convergence_step", step as f64));
    }
    // 呼び出し 0 回のときのヒット率は定義されない．規則モードの 0 回を 0.0 と
    // 書くと «1 度も当たらなかった» と読めてしまう．
    if result.metadata.total() > 0 {
        aggregates.push(("llm_cache_hit_rate", result.metadata.cache_hit_rate()));
    }
    run.log_metrics(SCOPE, &aggregates)
        .expect("run スコープの指標の記録に失敗");
}

/// [`MetricsRow`] の 11 フィールドを 1 ステップぶんまとめて書く．
///
/// `rule_*` の 5 本は IVT 5 ルールそれぞれの発火率という «ステップごとの数» が 5 つ
/// あるだけで，ルールというカテゴリに番号を振ったものではない．
fn log_step(run: &mut Run, m: &MetricsRow) {
    run.log_metrics_at(
        m.t,
        T_UNIT,
        SCOPE,
        &[
            ("silence_rate", m.silence_rate),
            ("upward_silence_rate", m.upward_silence_rate),
            ("climate_of_silence", m.climate_of_silence),
            ("silence_voice_corr", m.silence_voice_corr),
            ("issue_salience", m.issue_salience),
            ("rule_target_id", m.rule_target_id),
            ("rule_need_data", m.rule_need_data),
            ("rule_no_bypass", m.rule_no_bypass),
            ("rule_no_embarrass", m.rule_no_embarrass),
            ("rule_career_consq", m.rule_career_consq),
            ("max_rule_cooccurrence", m.max_rule_cooccurrence),
        ],
    )
    .unwrap_or_else(|e| panic!("step {} の指標の記録に失敗: {e}", m.t));
}

/// 最終ステップの従業員 1 人 1 行の表を `artifacts/agents.csv` に書く．
///
/// 発現 (`expression`) ・動機 (`motive`) ・発火ルール列 (`active_rules`) はカテゴリで
/// あって数ではないので指標にしない．数の列だけを `metrics.csv` に移すと 1 つの表が
/// 2 ファイルに割れ，`active_rules` から相関行列を組む Python 側が組み直せなくなる．
/// 表は表のまま artifacts に置く (`finish()` が manifest に hash を残す)．
pub fn save_agents(run: &Run, result: &SimulationResult) {
    let dir = run.dir().join("artifacts");
    std::fs::create_dir_all(&dir).expect("artifacts ディレクトリの作成に失敗");
    write_csv(&result.agent_rows, dir.join("agents.csv"));
}

/// `serde` で直列化できる行の並びを CSV に書く．
fn write_csv<T: Serialize>(rows: &[T], path: impl AsRef<Path>) {
    let path = path.as_ref();
    let mut writer = csv::Writer::from_path(path)
        .unwrap_or_else(|e| panic!("{} を開けません: {e}", path.display()));
    for row in rows {
        writer
            .serialize(row)
            .unwrap_or_else(|e| panic!("{} への書き込みに失敗: {e}", path.display()));
    }
    writer
        .flush()
        .unwrap_or_else(|e| panic!("{} の flush に失敗: {e}", path.display()));
}

// ---------------------------------------------------------------------------
// ablation — 1 つの run の中で決定モードを対比する
// ---------------------------------------------------------------------------

/// `ablation` の試行 1 本 (モード × シード)．
///
/// この 3 指標には時間軸が無く，62 本の試行を `metrics.csv` に並べると全行が同じ
/// 主キー (`step`, `step_unit`, `scope`, `name`) を名乗ってしまう．試行は
/// `events.jsonl` に置き，`metrics.csv` にはモード別の集約だけを書く．
#[derive(Debug, Clone, Serialize)]
pub struct AblationTrial {
    /// 決定モード (`rule` / `rule_no_ivt` / `llm`)．
    pub mode: String,
    /// この試行のシード．
    pub seed: u64,
    /// 最終ステップの upward silence 率．
    pub upward_silence_rate: f64,
    /// 時間平均 silence↔voice 相関．
    pub silence_voice_corr: f64,
    /// 最終ステップの最大非対角同時発火率．
    pub max_rule_cooccurrence: f64,
}

/// 試行 1 本を `events.jsonl` に書く．
pub fn log_trial(run: &mut Run, trial: &AblationTrial) {
    run.log_event(TRIAL_EVENT, trial).unwrap_or_else(|e| {
        panic!(
            "試行 ({}, seed={}) の記録に失敗: {e}",
            trial.mode, trial.seed
        )
    });
}

/// モード別の平均と試行数．名前でモードを分ける (1 つの run に同居するため)．
pub fn log_mode_summary(run: &mut Run, mode: &str, upward: &[f64]) {
    let mean = upward.iter().sum::<f64>() / upward.len().max(1) as f64;
    run.log_metrics(
        SCOPE,
        &[
            (format!("{mode}_mean_upward_silence_rate").as_str(), mean),
            (format!("{mode}_n_trials").as_str(), upward.len() as f64),
        ],
    )
    .unwrap_or_else(|e| panic!("モード {mode} の集約の記録に失敗: {e}"));
}

/// IVT 主効果 — `rule` と `rule_no_ivt` の差．
///
/// これは «1 回の実行の中で 2 つの腕を測った差» なので，腕を子 run に割ると主張が
/// どの run にも属さなくなる．同じ run の中に置く．
pub fn log_ivt_effect(run: &mut Run, delta: f64, cohens_d: f64) {
    run.log_metrics(
        SCOPE,
        &[
            ("ivt_effect_delta", delta),
            ("ivt_effect_cohens_d", cohens_d),
        ],
    )
    .expect("IVT 主効果の記録に失敗");
}

// ---------------------------------------------------------------------------
// reproduce の帯照合
// ---------------------------------------------------------------------------

/// 帯照合 1 件．
#[derive(Debug, Clone, Serialize)]
pub struct Check {
    /// 指標名．
    pub indicator: String,
    /// 観測値．
    pub observed: f64,
    /// 照合先の帯 (人が読む形．`|Δ| < 0.07` など)．
    pub band: String,
    /// 判定．移行前に標準出力へ出していた語をそのまま使う．
    pub verdict: String,
}

impl Check {
    /// PASS したか．
    pub fn passed(&self) -> bool {
        self.verdict == "PASS"
    }
}

/// 観測量そのものは run 全体を 1 つの値で表す数なので指標に書く．
pub fn log_observations(run: &mut Run, checks: &[Check]) {
    let values: Vec<(&str, f64)> = checks
        .iter()
        .map(|c| (c.indicator.as_str(), c.observed))
        .collect();
    run.log_metrics(SCOPE, &values)
        .expect("headline 観測量の記録に失敗");

    let passed = checks.iter().filter(|c| c.passed()).count();
    run.log_metrics(
        SCOPE,
        &[
            ("checks_passed", passed as f64),
            ("checks_total", checks.len() as f64),
        ],
    )
    .expect("帯照合の集計の記録に失敗");
}

/// 帯照合の判定は数ではないので `events.jsonl` へ書く．
///
/// 照合先の帯 (`0.50 ± 0.07`，`[-0.65, -0.45]`，`< 0.50`) は設計書がこの再現実装の
/// ために置いた定性的なアンカーであって，論文が報告した数値そのものではない．出典を
/// 要求する `reference.csv` には書かない — 書くと論文の報告値と自前のアンカーが後から
/// 見分けられなくなる．
pub fn log_checks(run: &mut Run, checks: &[Check]) {
    for c in checks {
        run.log_event(CHECK_EVENT, c)
            .unwrap_or_else(|e| panic!("帯照合 {} の記録に失敗: {e}", c.indicator));
    }
}
