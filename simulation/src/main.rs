//! Detert & Edmondson (2011) — Implicit Voice Theories silence CLI.
//!
//! `run`       : single configuration; `--llm-mode {llm|rule|rule_no_ivt}`.
//! `sweep`     : Cartesian product over `β_ι × ψ̄ × seeds`; one row per cell.
//! `ablation`  : contrast decision modes (e.g. `rule,rule_no_ivt`) across seeds.
//! `reproduce` : per-mode steady-state report against the design's anchors.

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};

use detert_silence::config::{
    parse_llm_mode, parse_network_kind, BetaGroup, Config, LlmMode, LlmSettings, NetworkKind,
};
use detert_silence::simulation::{
    cohens_d, ensure_output_dir, run, save_agents, save_llm_meta, save_metrics,
    save_rule_activation, SimulationResult,
};

use socsim_core::derive_seed;
use socsim_results::{refresh_latest_symlink, timestamp, write_csv, write_json};

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
    /// Sweep β_ι × ψ̄ across seeds; aggregate into `sweep_summary.csv`.
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
// CSV rows
// --------------------------------------------------------------------------- //

#[derive(serde::Serialize)]
struct SweepRow {
    llm_mode: String,
    beta_ivt: f64,
    psafety_mean: f64,
    run: usize,
    seed: u64,
    final_round: u64,
    upward_silence_rate: f64,
    silence_voice_corr: f64,
    max_rule_cooccurrence: f64,
    convergence_step: i64,
}

#[derive(serde::Serialize)]
struct AblationRow {
    mode: String,
    seed: u64,
    upward_silence_rate: f64,
    silence_voice_corr: f64,
    max_rule_cooccurrence: f64,
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
        output_dir: args.output_dir.clone(),
        ..Config::default()
    }
}

// --------------------------------------------------------------------------- //
// run
// --------------------------------------------------------------------------- //

fn cmd_run(args: RunArgs) {
    let timestamp = timestamp();
    let output_dir = format!("{}/{}", args.output_dir, timestamp);
    ensure_output_dir(&output_dir);

    let mut base_cfg = cfg_from_run_args(&args);
    base_cfg.output_dir = output_dir.clone();
    if base_cfg.llm_mode.is_llm() {
        if let Some(parent) = Path::new(&args.cache_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
    }

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
    println!("output: {output_dir}");
    println!("----------------------------------------------------------------------");

    {
        let path = format!("{output_dir}/config.json");
        write_json(&base_cfg.to_run_config_json(), &path).expect("failed to write config.json");
    }

    let mut last_result: Option<SimulationResult> = None;
    let runs = base_cfg.runs.max(1);
    for run_idx in 0..runs {
        let seed = derive_seed(base_cfg.seed, &[run_idx as u64]);
        let cfg = Config {
            seed,
            ..base_cfg.clone()
        };
        let result = run(&cfg).unwrap_or_else(|e| panic!("run failed: {e}"));
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

    let result = last_result.expect("at least one run");
    save_metrics(&result, &output_dir);
    save_agents(&result, &output_dir);
    save_rule_activation(&result, &output_dir);
    save_llm_meta(&result, &base_cfg, &output_dir);

    let _ = refresh_latest_symlink(&args.output_dir, &timestamp);

    println!("----------------------------------------------------------------------");
    println!(
        "LLM calls: {} | cache-hit: {} ({:.1}%) | model: {}",
        result.metadata.total(),
        result.metadata.cache_hits(),
        result.metadata.cache_hit_rate() * 100.0,
        result.llm_model,
    );
    println!("metrics        → {output_dir}/metrics.csv");
    println!("agents         → {output_dir}/agents.csv");
    println!("rule_activation→ {output_dir}/rule_activation.csv");
    println!("llm_meta       → {output_dir}/llm_meta.json");
    println!("config         → {output_dir}/config.json");
}

// --------------------------------------------------------------------------- //
// sweep
// --------------------------------------------------------------------------- //

fn cmd_sweep(args: SweepArgs) {
    let llm_mode = parse_llm_mode(&args.llm_mode).unwrap_or_else(|e| panic!("{e}"));
    let timestamp = timestamp();
    let dir_name = format!("{timestamp}_sweep");
    let sweep_dir = format!("{}/{}", args.output_dir, dir_name);
    fs::create_dir_all(&sweep_dir).expect("failed to create sweep dir");

    let mut beta_ivt_vals: Vec<f64> = Vec::new();
    let mut b = args.beta_ivt_min;
    while b <= args.beta_ivt_max + 1e-9 {
        beta_ivt_vals.push((b * 1000.0).round() / 1000.0);
        b += args.beta_ivt_step.max(1e-6);
    }
    let psafety_vals = parse_f64_list(&args.psafety_mean_values);

    let n_cells = beta_ivt_vals.len() * psafety_vals.len();
    let n_total = n_cells * args.runs;
    println!("=== detert-sweep ===");
    println!(
        "mode: {} | β_ι={:?} | ψ̄={:?} | runs/cell={} | total {} runs",
        llm_mode.label(),
        beta_ivt_vals,
        psafety_vals,
        args.runs,
        n_total,
    );
    println!("output: {sweep_dir}");
    println!("------------------------------------------------------------");

    {
        let config_json = serde_json::json!({
            "command": "sweep",
            "llm_mode": llm_mode.label(),
            "n_teams": args.n_teams,
            "team_size": args.team_size,
            "beta_ivt_values": beta_ivt_vals,
            "psafety_mean_values": psafety_vals,
            "runs": args.runs,
            "t_max": args.t_max,
            "seed": args.seed,
        });
        let path = format!("{sweep_dir}/sweep_config.json");
        write_json(&config_json, &path).expect("failed to write sweep_config.json");
    }

    let mut rows: Vec<SweepRow> = Vec::with_capacity(n_total);
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
                let result = run(&cfg).unwrap_or_else(|e| panic!("sweep run failed: {e}"));
                let last = result
                    .metrics_rows
                    .last()
                    .expect("metrics_rows must not be empty");
                rows.push(SweepRow {
                    llm_mode: llm_mode.label().to_string(),
                    beta_ivt: bivt,
                    psafety_mean: psi,
                    run: run_idx,
                    seed,
                    final_round: result.final_round,
                    upward_silence_rate: last.upward_silence_rate,
                    silence_voice_corr: result.final_silence_voice_corr(),
                    max_rule_cooccurrence: last.max_rule_cooccurrence,
                    convergence_step: result.convergence_step.map(|x| x as i64).unwrap_or(-1),
                });
                if idx.is_multiple_of(10) || idx == n_total {
                    println!(
                        "[{}/{}] β_ι={:.2} ψ̄={:.2} run={} upward_silence={:.3}",
                        idx, n_total, bivt, psi, run_idx, last.upward_silence_rate
                    );
                }
            }
        }
    }

    let path = format!("{sweep_dir}/sweep_summary.csv");
    write_csv(&rows, &path).expect("failed to write sweep_summary.csv");

    let _ = refresh_latest_symlink(&args.output_dir, &dir_name);
    println!("------------------------------------------------------------");
    println!("sweep done.");
    println!("summary → {sweep_dir}/sweep_summary.csv");
    println!("config  → {sweep_dir}/sweep_config.json");
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

    let timestamp = timestamp();
    let dir_name = format!("{timestamp}_ablation");
    let abl_dir = format!("{}/{}", args.output_dir, dir_name);
    fs::create_dir_all(&abl_dir).expect("failed to create ablation dir");

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

    {
        let config_json = serde_json::json!({
            "command": "ablation",
            "modes": modes.iter().map(|m| m.label()).collect::<Vec<_>>(),
            "n_teams": args.n_teams,
            "team_size": args.team_size,
            "seed_start": args.seed_start,
            "seed_end": args.seed_end,
            "t_max": args.t_max,
        });
        write_json(&config_json, format!("{abl_dir}/sweep_config.json"))
            .expect("failed to write sweep_config.json");
    }

    let mut rows: Vec<AblationRow> = Vec::new();
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
            let result = run(&cfg).unwrap_or_else(|e| panic!("ablation run failed: {e}"));
            let last = result.metrics_rows.last().expect("metrics");
            per_mode
                .entry(mode.label().to_string())
                .or_default()
                .push(last.upward_silence_rate);
            rows.push(AblationRow {
                mode: mode.label().to_string(),
                seed,
                upward_silence_rate: last.upward_silence_rate,
                silence_voice_corr: result.final_silence_voice_corr(),
                max_rule_cooccurrence: last.max_rule_cooccurrence,
            });
        }
    }

    write_csv(&rows, format!("{abl_dir}/ablation_summary.csv"))
        .expect("failed to write ablation_summary.csv");

    println!("------------------------------------------------------------");
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    for (m, v) in &per_mode {
        println!(
            "  {:<12} mean upward_silence_rate = {:.3} (n={})",
            m,
            mean(v),
            v.len()
        );
    }
    // IVT necessity: rule vs rule_no_ivt Cohen's d.
    if let (Some(r), Some(n)) = (per_mode.get("rule"), per_mode.get("rule_no_ivt")) {
        let d = cohens_d(r, n);
        println!(
            "  IVT main effect (rule − rule_no_ivt): Δ={:.3}, Cohen's d={:.2}",
            mean(r) - mean(n),
            d
        );
    }
    let _ = refresh_latest_symlink(&args.output_dir, &dir_name);
    println!("summary → {abl_dir}/ablation_summary.csv");
}

// --------------------------------------------------------------------------- //
// reproduce
// --------------------------------------------------------------------------- //

fn cmd_reproduce(args: ReproduceArgs) {
    let mode = parse_llm_mode(&args.llm_mode).unwrap_or_else(|e| panic!("{e}"));
    println!("=== detert-reproduce ({} mode) ===", mode.label());
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
        let result = run(&cfg).unwrap_or_else(|e| panic!("reproduce run failed: {e}"));
        let last = result.metrics_rows.last().expect("metrics");
        up.push(last.upward_silence_rate);
        sv.push(result.final_silence_voice_corr());
        cooc.push(last.max_rule_cooccurrence);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    println!("steady-state over {} runs:", up.len());
    let mu = mean(&up);
    println!(
        "  upward_silence_rate  = {:.3}   (HiCo anchor ≈ 0.50; PASS if |Δ|<.07: {})",
        mu,
        if (mu - 0.50).abs() < 0.07 {
            "PASS"
        } else {
            "off-anchor"
        }
    );
    let msv = mean(&sv);
    println!(
        "  silence_voice_corr   = {:.3}   (Study 4 r=-.55; PASS if ∈[-.65,-.45]: {})",
        msv,
        if (-0.65..=-0.45).contains(&msv) {
            "PASS"
        } else {
            "review"
        }
    );
    let mc = mean(&cooc);
    println!(
        "  max_rule_cooccurrence= {:.3}   (discriminant <.50: {})",
        mc,
        if mc < 0.50 {
            "PASS"
        } else {
            "non-discriminant"
        }
    );
    println!();
    println!("For the full Table-4-style report + CFA-style fit indices (RMSEA / CFI)");
    println!("reproduced from the ABM rule-firing matrix, run the Python tool:");
    println!("  uv run detert-tools reproduce --results-dir results/latest");
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
