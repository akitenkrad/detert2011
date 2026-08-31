//! Detert & Edmondson (2011) — Implicit Voice Theories silence CLI.
//!
//! `run`       : single configuration; `--llm-mode {llm|rule|rule_no_ivt}`.
//! `sweep`     : Cartesian product over `β_ι × ψ̄ × seeds`; a parent run plus one
//!               child run per cell × trial.
//! `ablation`  : contrast decision modes (e.g. `rule,rule_no_ivt`) across seeds.
//! `reproduce` : per-mode steady-state report against the design's anchors.
//!
//! サブコマンド 1 回が runvault の run 1 本になる．出力の置き場と同一性 (run ディレ
//! クトリ・`config.json`・`metrics.csv`・`events.jsonl`) は runvault が持つので，
//! ここではタイムスタンプ付きディレクトリも `latest` symlink も作らない．

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use runvault::{Lineage, Run, RunOptions};
use serde::Serialize;

use detert_silence::config::{
    parse_llm_mode, parse_network_kind, BetaGroup, Config, LlmMode, LlmSettings, NetworkKind,
    RunConfigJson,
};
use detert_silence::llm::{build_live_client, SilenceClient};
use detert_silence::record::{self, AblationTrial, Check, DOMAIN, EXPERIMENT, REPO_ID};
use detert_silence::simulation::{cohens_d, run_with_client, SimulationResult};

use socsim_core::derive_seed;

// --------------------------------------------------------------------------- //
// CLI
// --------------------------------------------------------------------------- //

#[derive(Parser, Debug)]
#[command(
    name = "detert",
    about = "Detert & Edmondson (2011) — Implicit Voice Theories (LLM vs rule vs rule_no_ivt)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    /// Ollama 接続先 URL（指定時は環境変数 OLLAMA_HOST を上書きする）．
    #[arg(long, global = true)]
    ollama_host: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run a single configuration.
    Run(RunArgs),
    /// Sweep β_ι × ψ̄ across seeds; a parent run plus one child run per cell × trial.
    Sweep(SweepArgs),
    /// Contrast decision modes across a seed range.
    Ablation(AblationArgs),
    /// Per-mode steady-state report against the design anchors.
    Reproduce(ReproduceArgs),
}

#[derive(Parser, Debug)]
struct RunArgs {
    /// Decision mechanism (llm / rule / rule_no_ivt).
    #[arg(long, default_value = "rule")]
    llm_mode: String,
    /// Total number of employees (overrides n_teams × team_size if set).
    #[arg(long)]
    n: Option<usize>,
    /// Number of teams.
    #[arg(long, default_value_t = 8)]
    n_teams: usize,
    /// Employees per team.
    #[arg(long, default_value_t = 25)]
    team_size: usize,
    /// Number of hierarchical levels.
    #[arg(long, default_value_t = 3)]
    n_levels: u8,
    /// Network family.
    #[arg(long, default_value = "watts-strogatz")]
    network_model: String,
    /// Watts–Strogatz `k`.
    #[arg(long, default_value_t = 6)]
    network_k: usize,
    /// Watts–Strogatz β / Erdős–Rényi p.
    #[arg(long, default_value_t = 0.1)]
    network_beta: f64,
    /// Mean IVT strength ι̅.
    #[arg(long, default_value_t = 0.55)]
    ivt_mean: f64,
    /// Std-dev of IVT strength.
    #[arg(long, default_value_t = 0.20)]
    ivt_sd: f64,
    /// β_ψ — psychological-safety coefficient.
    #[arg(long, default_value_t = 1.2)]
    beta_psafety: f64,
    /// β_f — fear coefficient.
    #[arg(long, default_value_t = 1.5)]
    beta_fear: f64,
    /// β_ι — IVT main-effect coefficient (calibrated HiCo point).
    #[arg(long, default_value_t = 2.0)]
    beta_ivt: f64,
    /// Per-agent per-step retaliation probability.
    #[arg(long, default_value_t = 0.05)]
    p_retaliate: f64,
    /// Optional exogenous σ-shock time step.
    #[arg(long, default_value_t = 24)]
    shock_t: u64,
    /// Maximum simulation step.
    #[arg(long, default_value_t = 60)]
    t_max: u64,
    /// Number of independent runs (outputs reflect the *last* run).
    #[arg(long, default_value_t = 1)]
    runs: usize,
    /// Random seed (governs the socsim core layer).
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// LLM generation temperature.
    #[arg(long, default_value_t = 0.0)]
    llm_temperature: f32,
    /// LLM generation seed (offset; per-(agent, t) seed derived from it).
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,
    /// Prompt → response cache path (LLM mode only).
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,
    /// Output base directory.
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct SweepArgs {
    /// Decision mode (β_ι sweep is meaningful only for rule).
    #[arg(long, default_value = "rule")]
    llm_mode: String,
    /// Number of teams.
    #[arg(long, default_value_t = 8)]
    n_teams: usize,
    /// Employees per team.
    #[arg(long, default_value_t = 25)]
    team_size: usize,
    /// β_ι sweep minimum.
    #[arg(long, default_value_t = 0.0)]
    beta_ivt_min: f64,
    /// β_ι sweep maximum.
    #[arg(long, default_value_t = 1.6)]
    beta_ivt_max: f64,
    /// β_ι sweep step.
    #[arg(long, default_value_t = 0.2)]
    beta_ivt_step: f64,
    /// ψ̄ mean values (comma-separated).
    #[arg(long, default_value = "0.3,0.5,0.7")]
    psafety_mean_values: String,
    /// Runs (seeds) per cell.
    #[arg(long, default_value_t = 5)]
    runs: usize,
    /// Maximum simulation step.
    #[arg(long, default_value_t = 60)]
    t_max: u64,
    /// Base seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Output base directory.
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct AblationArgs {
    /// Comma-separated decision modes to contrast.
    #[arg(long, default_value = "rule,rule_no_ivt")]
    modes: String,
    /// Number of teams.
    #[arg(long, default_value_t = 8)]
    n_teams: usize,
    /// Employees per team.
    #[arg(long, default_value_t = 25)]
    team_size: usize,
    /// First seed (inclusive).
    #[arg(long, default_value_t = 0)]
    seed_start: u64,
    /// Last seed (inclusive).
    #[arg(long, default_value_t = 30)]
    seed_end: u64,
    /// Maximum simulation step.
    #[arg(long, default_value_t = 60)]
    t_max: u64,
    /// Output base directory.
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct ReproduceArgs {
    /// Decision mode to report.
    #[arg(long, default_value = "rule")]
    llm_mode: String,
    /// Number of teams.
    #[arg(long, default_value_t = 8)]
    n_teams: usize,
    /// Employees per team.
    #[arg(long, default_value_t = 25)]
    team_size: usize,
    /// Maximum simulation step.
    #[arg(long, default_value_t = 60)]
    t_max: u64,
    /// Base seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Runs.
    #[arg(long, default_value_t = 5)]
    runs: usize,
    /// Output base directory.
    #[arg(long, default_value = "results")]
    output_dir: String,
}

// --------------------------------------------------------------------------- //
// run ごとの parameters
// --------------------------------------------------------------------------- //

/// `sweep` の子 run の条件．
///
/// ψ̄ は [`Config`] に無い — β_ψ の倍率としてしか効かないためである．しかし掃引の
/// 軸そのものなので子の条件として残す．無いと «どの ψ̄ の run だったか» が後から
/// 辿れず，旧 `sweep_summary.csv` の `psafety_mean` 列が失われる．
#[derive(Serialize)]
struct SweepPointConfigJson {
    #[serde(flatten)]
    base: RunConfigJson,
    psafety_mean: f64,
}

/// `sweep` 親 run の条件 — 格子の定義そのもの．
#[derive(Serialize)]
struct SweepConfigJson {
    llm_mode: String,
    n_teams: usize,
    team_size: usize,
    beta_ivt_values: Vec<f64>,
    psafety_mean_values: Vec<f64>,
    runs: usize,
    t_max: u64,
    seed: u64,
}

/// `ablation` run の条件．
#[derive(Serialize)]
struct AblationConfigJson {
    modes: Vec<String>,
    n_teams: usize,
    team_size: usize,
    seed_start: u64,
    seed_end: u64,
    t_max: u64,
}

/// `reproduce` run の条件．
#[derive(Serialize)]
struct ReproduceConfigJson {
    llm_mode: String,
    n_teams: usize,
    team_size: usize,
    t_max: u64,
    runs: usize,
    seed: u64,
}

// --------------------------------------------------------------------------- //
// helpers
// --------------------------------------------------------------------------- //

fn parse_f64_list(s: &str) -> Vec<f64> {
    s.split([',', ' '])
        .filter(|t| !t.is_empty())
        .filter_map(|t| t.trim().parse::<f64>().ok())
        .collect()
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

/// LLM モードなら 1 試行ぶんのクライアントを組む．規則モードは LLM を 1 度も
/// 呼ばないので `None`．
fn build_client(cfg: &Config) -> Option<SilenceClient> {
    if cfg.llm_mode.is_llm() {
        Some(
            build_live_client(&cfg.llm)
                .unwrap_or_else(|e| panic!("LLM クライアント構築に失敗: {e}")),
        )
    } else {
        None
    }
}

/// 実際に応答するバックエンドから `llm` ブロックを組む．
fn llm_block_of(client: Option<&SilenceClient>, temperature: f32) -> Option<runvault::Llm> {
    client.map(|c| record::llm_block(c.inner().model(), c.inner().endpoint(), temperature))
}

fn cfg_from_run_args(args: &RunArgs) -> Config {
    let (n_teams, team_size) = match args.n {
        Some(n) if args.team_size > 0 => {
            let teams = n.div_ceil(args.team_size);
            (teams.max(1), args.team_size)
        }
        _ => (args.n_teams, args.team_size),
    };
    Config {
        n_teams,
        team_size,
        n_levels: args.n_levels,
        network_kind: parse_network_kind(&args.network_model).unwrap_or(NetworkKind::WattsStrogatz),
        network_k: args.network_k,
        network_beta: args.network_beta,
        llm_mode: parse_llm_mode(&args.llm_mode).unwrap_or_else(|e| panic!("{e}")),
        ivt_mean: args.ivt_mean,
        ivt_sd: args.ivt_sd,
        beta: BetaGroup {
            beta_psafety: args.beta_psafety,
            beta_fear: args.beta_fear,
            beta_ivt: args.beta_ivt,
            ..BetaGroup::default()
        },
        p_retaliate: args.p_retaliate,
        shock_t: Some(args.shock_t),
        shock_magnitude: 0.3,
        t_max: args.t_max,
        runs: args.runs,
        seed: args.seed,
        llm: LlmSettings {
            temperature: args.llm_temperature,
            seed: args.llm_seed,
            cache_path: Some(args.cache_path.clone()),
        },
        ..Config::default()
    }
}

// --------------------------------------------------------------------------- //
// run
// --------------------------------------------------------------------------- //

fn cmd_run(args: RunArgs) {
    let base_cfg = cfg_from_run_args(&args);
    if base_cfg.llm_mode.is_llm() {
        if let Some(parent) = Path::new(&args.cache_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
    }

    // LLM クライアントは run を開始する前に組む．`llm` ブロックに書くモデル名と
    // endpoint は，実際に応答するバックエンドから採らないと意味を持たない．組んだ
    // ものは 1 本目の試行がそのまま使う．
    let mut pending = build_client(&base_cfg);
    let llm = llm_block_of(pending.as_ref(), base_cfg.llm.temperature);

    let parameters = base_cfg.to_run_config_json();
    let mut options = RunOptions::new(EXPERIMENT, "run")
        .repo_id(REPO_ID)
        .domain(DOMAIN)
        .results_root(&args.output_dir)
        .parameters(&parameters)
        .expect("runvault: parameters の組み立てに失敗")
        .seed_pointers(["/seed"])
        .master_seed(base_cfg.seed)
        .replication(record::replication());
    if let Some(llm) = llm {
        options = options.llm(llm);
    }
    let mut rv = Run::start(options).expect("runvault: run の開始に失敗");

    println!("=== Detert & Edmondson (2011) — Implicit Voice Theories ===");
    println!(
        "llm-mode: {} | teams: {}×{} (={}) | network: {:?} k={} β={:.2}",
        base_cfg.llm_mode.label(),
        base_cfg.n_teams,
        base_cfg.team_size,
        base_cfg.n_employees(),
        base_cfg.network_kind,
        base_cfg.network_k,
        base_cfg.network_beta,
    );
    println!(
        "ι̅={:.2} ι_sd={:.2} | β_ι={:.2} β_ψ={:.2} β_f={:.2} | t_max={} runs={} seed={}",
        base_cfg.ivt_mean,
        base_cfg.ivt_sd,
        base_cfg.beta.beta_ivt,
        base_cfg.beta.beta_psafety,
        base_cfg.beta.beta_fear,
        base_cfg.t_max,
        base_cfg.runs,
        base_cfg.seed,
    );
    println!("出力先: {}", rv.dir().display());
    println!("----------------------------------------------------------------------");

    let mut last_result: Option<SimulationResult> = None;
    let runs = base_cfg.runs.max(1);
    for run_idx in 0..runs {
        let seed = derive_seed(base_cfg.seed, &[run_idx as u64]);
        let cfg = Config {
            seed,
            ..base_cfg.clone()
        };
        let client = pending.take().or_else(|| build_client(&cfg));
        let result = run_with_client(&cfg, client).unwrap_or_else(|e| panic!("run failed: {e}"));
        println!(
            "[{}/{}] seed={} upward_silence={:.3} silence_voice_r={:.3} max_cooc={:.3} conv={:?}",
            run_idx + 1,
            runs,
            seed,
            result.final_upward_silence(),
            result.final_silence_voice_corr(),
            result
                .metrics_rows
                .last()
                .map(|r| r.max_rule_cooccurrence)
                .unwrap_or(0.0),
            result.convergence_step,
        );
        last_result = Some(result);
    }

    // 記録するのは最後の試行 — 移行前も `metrics.csv` / `agents.csv` は最後の試行
    // のものだった．`runs` は parameters にあるので，何本目を記録したかは辿れる．
    let result = last_result.expect("at least one run");
    record::log_simulation(&mut rv, &result);
    record::save_agents(&rv, &result);

    println!("----------------------------------------------------------------------");
    println!(
        "LLM calls: {} | cache-hit: {} ({:.1}%) | model: {}",
        result.metadata.total(),
        result.metadata.cache_hits(),
        result.metadata.cache_hit_rate() * 100.0,
        result.llm_model,
    );
    let dir = rv.finish().expect("runvault: run の完了に失敗");
    println!("指標   → {}/metrics.csv", dir.display());
    println!("従業員 → {}/artifacts/agents.csv", dir.display());
    println!("設定   → {}/config.json", dir.display());
}

// --------------------------------------------------------------------------- //
// sweep
// --------------------------------------------------------------------------- //

fn cmd_sweep(args: SweepArgs) {
    let llm_mode = parse_llm_mode(&args.llm_mode).unwrap_or_else(|e| panic!("{e}"));

    let mut beta_ivt_vals: Vec<f64> = Vec::new();
    let mut b = args.beta_ivt_min;
    while b <= args.beta_ivt_max + 1e-9 {
        beta_ivt_vals.push((b * 1000.0).round() / 1000.0);
        b += args.beta_ivt_step.max(1e-6);
    }
    let psafety_vals = parse_f64_list(&args.psafety_mean_values);

    let n_cells = beta_ivt_vals.len() * psafety_vals.len();
    let n_total = n_cells * args.runs;

    // 親 run: 格子の定義そのものを parameters に持つ．個別条件の指標は書かない．
    // 親は単一の master_seed を持たない (条件ごとの子が派生シードをそれぞれ持つ)．
    // base seed は /parameters.seed と seed_pointers 経由で execution_hash に残る．
    // sweep_id は runvault が親の run_slug で埋める．
    let sweep_parameters = SweepConfigJson {
        llm_mode: llm_mode.label().to_string(),
        n_teams: args.n_teams,
        team_size: args.team_size,
        beta_ivt_values: beta_ivt_vals.clone(),
        psafety_mean_values: psafety_vals.clone(),
        runs: args.runs,
        t_max: args.t_max,
        seed: args.seed,
    };
    let parent = Run::start(
        RunOptions::new(EXPERIMENT, "sweep")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&sweep_parameters)
            .expect("runvault: sweep の parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .sweep_parent()
            .replication(record::replication()),
    )
    .expect("runvault: sweep 親 run の開始に失敗");

    let sweep_id = parent
        .sweep_id()
        .expect("runvault: sweep 親に sweep_id がありません")
        .to_string();
    let parent_run_uid = parent.run_uid().to_string();

    println!("=== detert-sweep ===");
    println!(
        "mode: {} | β_ι={:?} | ψ̄={:?} | runs/cell={} | total {} runs",
        llm_mode.label(),
        beta_ivt_vals,
        psafety_vals,
        args.runs,
        n_total,
    );
    println!("出力先: {}", parent.dir().display());
    println!("------------------------------------------------------------");

    let mut idx = 0usize;
    for &bivt in &beta_ivt_vals {
        for &psi in &psafety_vals {
            for run_idx in 0..args.runs {
                idx += 1;
                let seed = derive_seed(
                    args.seed,
                    &[
                        (bivt * 1000.0) as u64,
                        (psi * 1000.0) as u64,
                        run_idx as u64,
                    ],
                );
                // ψ̄ axis: scale the psafety VOICE coefficient so higher target
                // ψ̄ raises VOICE — a monotone proxy for the climate mean.
                let psi_scale = (psi / 0.5).clamp(0.2, 2.0);
                let cfg = Config {
                    n_teams: args.n_teams,
                    team_size: args.team_size,
                    llm_mode,
                    beta: BetaGroup {
                        beta_ivt: bivt,
                        beta_psafety: BetaGroup::default().beta_psafety * psi_scale,
                        ..BetaGroup::default()
                    },
                    t_max: args.t_max,
                    runs: 1,
                    seed,
                    ..Config::default()
                };

                let client = build_client(&cfg);
                let llm = llm_block_of(client.as_ref(), cfg.llm.temperature);

                // 子は «その条件の run» そのもの．master_seed は base から派生した
                // 実際に使われるシードで，同一条件の繰り返しは replicate_index で分ける．
                let parameters = SweepPointConfigJson {
                    base: cfg.to_run_config_json(),
                    psafety_mean: psi,
                };
                let mut options = RunOptions::new(EXPERIMENT, "run")
                    .repo_id(REPO_ID)
                    .domain(DOMAIN)
                    .results_root(&args.output_dir)
                    .parameters(&parameters)
                    .expect("runvault: 子 run の parameters の組み立てに失敗")
                    .seed_pointers(["/seed"])
                    .master_seed(seed)
                    .replicate_index(run_idx as u64)
                    .lineage(Lineage {
                        sweep_id: Some(sweep_id.clone()),
                        parent_run_uid: Some(parent_run_uid.clone()),
                        ..Default::default()
                    })
                    .replication(record::replication());
                if let Some(llm) = llm {
                    options = options.llm(llm);
                }
                let mut child = Run::start(options).expect("runvault: 子 run の開始に失敗");

                let result = run_with_client(&cfg, client)
                    .unwrap_or_else(|e| panic!("sweep run failed: {e}"));
                record::log_simulation(&mut child, &result);
                record::save_agents(&child, &result);

                let last = result
                    .metrics_rows
                    .last()
                    .expect("metrics_rows must not be empty");
                if idx.is_multiple_of(10) || idx == n_total {
                    println!(
                        "[{}/{}] β_ι={:.2} ψ̄={:.2} run={} upward_silence={:.3}",
                        idx, n_total, bivt, psi, run_idx, last.upward_silence_rate
                    );
                }
                child.finish().expect("runvault: 子 run の完了に失敗");
            }
        }
    }

    let dir = parent
        .finish()
        .expect("runvault: sweep 親 run の完了に失敗");
    println!("------------------------------------------------------------");
    println!("sweep done.");
    println!("掃引の定義 → {}/config.json", dir.display());
    println!("各セルの指標は子 run (subcommand=run) の metrics.csv にあります");
}

// --------------------------------------------------------------------------- //
// ablation
// --------------------------------------------------------------------------- //

fn cmd_ablation(args: AblationArgs) {
    let modes: Vec<LlmMode> = args
        .modes
        .split([',', ' '])
        .filter(|s| !s.is_empty())
        .map(|s| parse_llm_mode(s).unwrap_or_else(|e| panic!("{e}")))
        .collect();
    assert!(!modes.is_empty(), "no modes given");

    // 決定モードごとの腕を «1 回の実行の中で» 対比するのが ablation の主張なので，
    // 試行を子 run に割らない — 割ると IVT 主効果 (rule − rule_no_ivt) がどの run に
    // も属さなくなる．一方この run は 1 つの master_seed から派生するのではなく
    // seed_start..=seed_end という «シードの列» で駆動されるので，master_seed は
    // 名乗らない (runvault が sweep 親に用意している免除がこの形にあたる)．
    // 列そのものは /parameters と seed_pointers 経由で execution_hash に残る．
    let ablation_parameters = AblationConfigJson {
        modes: modes.iter().map(|m| m.label().to_string()).collect(),
        n_teams: args.n_teams,
        team_size: args.team_size,
        seed_start: args.seed_start,
        seed_end: args.seed_end,
        t_max: args.t_max,
    };

    // 最初に LLM を使う試行のクライアントだけ run の開始前に組む (`llm` ブロックの
    // モデル名と endpoint は実際に応答するバックエンドからしか採れない)．
    let probe_cfg = Config {
        llm_mode: *modes.iter().find(|m| m.is_llm()).unwrap_or(&modes[0]),
        ..Config::default()
    };
    let mut pending = build_client(&probe_cfg);
    let llm = llm_block_of(pending.as_ref(), probe_cfg.llm.temperature);

    let mut options = RunOptions::new(EXPERIMENT, "ablation")
        .repo_id(REPO_ID)
        .domain(DOMAIN)
        .results_root(&args.output_dir)
        .parameters(&ablation_parameters)
        .expect("runvault: ablation の parameters の組み立てに失敗")
        .seed_pointers(["/seed_start", "/seed_end"])
        .sweep_parent()
        .replication(record::replication());
    if let Some(llm) = llm {
        options = options.llm(llm);
    }
    let mut rv = Run::start(options).expect("runvault: ablation run の開始に失敗");

    println!("=== detert-ablation ===");
    println!(
        "modes: {:?} | seeds: {}..={} | teams {}×{} | t_max {}",
        modes.iter().map(|m| m.label()).collect::<Vec<_>>(),
        args.seed_start,
        args.seed_end,
        args.n_teams,
        args.team_size,
        args.t_max,
    );
    println!("出力先: {}", rv.dir().display());

    let mut per_mode: std::collections::BTreeMap<String, Vec<f64>> =
        std::collections::BTreeMap::new();
    for &mode in &modes {
        for seed in args.seed_start..=args.seed_end {
            let cfg = Config {
                n_teams: args.n_teams,
                team_size: args.team_size,
                llm_mode: mode,
                t_max: args.t_max,
                runs: 1,
                seed,
                ..Config::default()
            };
            let client = if cfg.llm_mode.is_llm() {
                pending.take().or_else(|| build_client(&cfg))
            } else {
                None
            };
            let result = run_with_client(&cfg, client)
                .unwrap_or_else(|e| panic!("ablation run failed: {e}"));
            let last = result.metrics_rows.last().expect("metrics");
            per_mode
                .entry(mode.label().to_string())
                .or_default()
                .push(last.upward_silence_rate);
            record::log_trial(
                &mut rv,
                &AblationTrial {
                    mode: mode.label().to_string(),
                    seed,
                    upward_silence_rate: last.upward_silence_rate,
                    silence_voice_corr: result.final_silence_voice_corr(),
                    max_rule_cooccurrence: last.max_rule_cooccurrence,
                },
            );
        }
    }

    println!("------------------------------------------------------------");
    for (m, v) in &per_mode {
        record::log_mode_summary(&mut rv, m, v);
        println!(
            "  {:<12} mean upward_silence_rate = {:.3} (n={})",
            m,
            mean(v),
            v.len()
        );
    }
    // IVT necessity: rule vs rule_no_ivt Cohen's d.
    if let (Some(r), Some(n)) = (per_mode.get("rule"), per_mode.get("rule_no_ivt")) {
        let delta = mean(r) - mean(n);
        let d = cohens_d(r, n);
        record::log_ivt_effect(&mut rv, delta, d);
        println!("  IVT main effect (rule − rule_no_ivt): Δ={delta:.3}, Cohen's d={d:.2}");
    }

    let dir = rv.finish().expect("runvault: ablation run の完了に失敗");
    println!("試行   → {}/events.jsonl", dir.display());
    println!("集約   → {}/metrics.csv", dir.display());
}

// --------------------------------------------------------------------------- //
// reproduce
// --------------------------------------------------------------------------- //

/// 設計書 §5 のアンカー．論文が報告した数値そのものではなく，この再現実装が置いた
/// 定性的な帯なので `reference.csv` には書かない (`record::log_checks` を参照)．
const HICO_ANCHOR: f64 = 0.50;
const HICO_TOL: f64 = 0.07;
const SILENCE_VOICE_BAND: (f64, f64) = (-0.65, -0.45);
const COOCCURRENCE_MAX: f64 = 0.50;

fn cmd_reproduce(args: ReproduceArgs) {
    let mode = parse_llm_mode(&args.llm_mode).unwrap_or_else(|e| panic!("{e}"));

    let parameters = ReproduceConfigJson {
        llm_mode: mode.label().to_string(),
        n_teams: args.n_teams,
        team_size: args.team_size,
        t_max: args.t_max,
        runs: args.runs,
        seed: args.seed,
    };

    let probe_cfg = Config {
        llm_mode: mode,
        ..Config::default()
    };
    let mut pending = build_client(&probe_cfg);
    let llm = llm_block_of(pending.as_ref(), probe_cfg.llm.temperature);

    let mut options = RunOptions::new(EXPERIMENT, "reproduce")
        .repo_id(REPO_ID)
        .domain(DOMAIN)
        .results_root(&args.output_dir)
        .parameters(&parameters)
        .expect("runvault: reproduce の parameters の組み立てに失敗")
        .seed_pointers(["/seed"])
        .master_seed(args.seed)
        .replication(record::replication());
    if let Some(llm) = llm {
        options = options.llm(llm);
    }
    let mut rv = Run::start(options).expect("runvault: reproduce run の開始に失敗");

    println!("=== detert-reproduce ({} mode) ===", mode.label());
    println!("出力先: {}", rv.dir().display());
    let mut up = Vec::new();
    let mut sv = Vec::new();
    let mut cooc = Vec::new();
    for run_idx in 0..args.runs.max(1) {
        let seed = derive_seed(args.seed, &[run_idx as u64]);
        let cfg = Config {
            n_teams: args.n_teams,
            team_size: args.team_size,
            llm_mode: mode,
            t_max: args.t_max,
            runs: 1,
            seed,
            ..Config::default()
        };
        let client = if cfg.llm_mode.is_llm() {
            pending.take().or_else(|| build_client(&cfg))
        } else {
            None
        };
        let result =
            run_with_client(&cfg, client).unwrap_or_else(|e| panic!("reproduce run failed: {e}"));
        let last = result.metrics_rows.last().expect("metrics");
        up.push(last.upward_silence_rate);
        sv.push(result.final_silence_voice_corr());
        cooc.push(last.max_rule_cooccurrence);
    }

    let mu = mean(&up);
    let msv = mean(&sv);
    let mc = mean(&cooc);
    let checks = vec![
        Check {
            indicator: "upward_silence_rate".to_string(),
            observed: mu,
            band: format!("|Δ| < {HICO_TOL:.2} around {HICO_ANCHOR:.2}"),
            verdict: if (mu - HICO_ANCHOR).abs() < HICO_TOL {
                "PASS"
            } else {
                "off-anchor"
            }
            .to_string(),
        },
        Check {
            indicator: "silence_voice_corr".to_string(),
            observed: msv,
            band: format!(
                "within [{:.2}, {:.2}]",
                SILENCE_VOICE_BAND.0, SILENCE_VOICE_BAND.1
            ),
            verdict: if (SILENCE_VOICE_BAND.0..=SILENCE_VOICE_BAND.1).contains(&msv) {
                "PASS"
            } else {
                "review"
            }
            .to_string(),
        },
        Check {
            indicator: "max_rule_cooccurrence".to_string(),
            observed: mc,
            band: format!("< {COOCCURRENCE_MAX:.2}"),
            verdict: if mc < COOCCURRENCE_MAX {
                "PASS"
            } else {
                "non-discriminant"
            }
            .to_string(),
        },
    ];
    record::log_observations(&mut rv, &checks);
    record::log_checks(&mut rv, &checks);

    println!("steady-state over {} runs:", up.len());
    println!(
        "  upward_silence_rate  = {:.3}   (HiCo anchor ≈ {:.2}; PASS if |Δ|<{:.2}: {})",
        mu, HICO_ANCHOR, HICO_TOL, checks[0].verdict,
    );
    println!(
        "  silence_voice_corr   = {:.3}   (Study 4 r=-.55; PASS if ∈[{:.2},{:.2}]: {})",
        msv, SILENCE_VOICE_BAND.0, SILENCE_VOICE_BAND.1, checks[1].verdict,
    );
    println!(
        "  max_rule_cooccurrence= {:.3}   (discriminant <{:.2}: {})",
        mc, COOCCURRENCE_MAX, checks[2].verdict,
    );
    println!();
    println!("For the full Table-4-style report + CFA-style fit indices (RMSEA / CFI)");
    println!("reproduced from the ABM rule-firing matrix, run the Python tool:");
    println!("  uv run detert-tools reproduce");

    let dir = rv.finish().expect("runvault: reproduce run の完了に失敗");
    println!("観測量 → {}/metrics.csv", dir.display());
    println!("帯照合 → {}/events.jsonl", dir.display());
}

// --------------------------------------------------------------------------- //
// main
// --------------------------------------------------------------------------- //

fn main() {
    let cli = Cli::parse();
    if let Some(host) = cli.ollama_host.as_deref() {
        std::env::set_var("OLLAMA_HOST", host);
    }
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Ablation(args) => cmd_ablation(args),
        Commands::Reproduce(args) => cmd_reproduce(args),
    }
}
