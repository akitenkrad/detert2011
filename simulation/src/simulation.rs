//! Initialisation + run driver for the Detert & Edmondson (2011) simulation.
//!
//! Two-layer determinism:
//! - **lower (deterministic socsim core)** — `derive_seed(root, &[0])` seeds
//!   world init (employee attributes + Watts–Strogatz network), `derive_seed(
//!   root, &[1])` seeds the engine (scheduler + retaliation + rule-vote draws).
//!   Bit-reproducible; `rule` / `rule_no_ivt` make zero LLM calls.
//! - **upper (non-deterministic LLM)** — confined to [`VoiceDecisionLlm`] via
//!   `socsim-llm`'s cached Ollama → OpenAI client. `temperature = 0` +
//!   `(agent_id, t)`-derived seed + prompt→response cache pseudo-determinise it.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use rand::Rng;
use serde::Serialize;
use socsim_core::{derive_seed, AgentId, SimClock, SimRng};
use socsim_engine::{RandomActivationScheduler, SimulationBuilder};
use socsim_llm::{LlmClient, MetadataCollector};
use socsim_net::SocialNetwork;

use crate::config::{Config, LlmMode, NetworkKind};
use crate::llm::{build_live_client, SilenceClient};
use crate::mechanisms::{
    ClimateSilence, FearAppraisal, IssueSalience, OrgPerformance, PrefalseCascade, PsafetyUpdate,
    RetaliationEvent, SharedClient, SharedMetadata, SilenceSpiral, VoiceDecisionLlm,
    VoiceDecisionRule,
};
use crate::metrics::{
    climate_of_silence, ever_silent_fraction, ivt_rule_activation_mix, max_offdiag_cooccurrence,
    rule_cooccurrence, silence_rate, silence_voice_corr, upward_silence_rate,
};
use crate::world::{Employee, Expression, IvtRule, SilenceWorld, Team};

/// RNG stream label: world init (employee attributes + network).
pub const RNG_WORLD_INIT: u64 = 0;
/// RNG stream label: socsim engine (scheduler + retaliation + rule-vote draws).
pub const RNG_ENGINE: u64 = 1;
/// RNG stream label: retaliation Bernoulli (folded into the engine stream).
pub const RNG_RETALIATE: u64 = 2;
/// RNG stream label: rule-vote Bernoulli + LLM `(agent, t)` seed root.
pub const RNG_RULE_VOTE: u64 = 3;

/// Convergence: `|C(t) − C(t-1)| < TOL` for `WINDOW` consecutive steps.
const CONVERGENCE_TOL: f64 = 1e-3;
const CONVERGENCE_WINDOW: u64 = 5;

// --------------------------------------------------------------------------- //
// Result containers + per-step row
// --------------------------------------------------------------------------- //

/// Per-step metrics row written to `metrics.csv`.
#[derive(Debug, Clone, Serialize)]
pub struct MetricsRow {
    pub t: u64,
    pub silence_rate: f64,
    pub upward_silence_rate: f64,
    pub climate_of_silence: f64,
    pub silence_voice_corr: f64,
    pub issue_salience: f64,
    pub rule_target_id: f64,
    pub rule_need_data: f64,
    pub rule_no_bypass: f64,
    pub rule_no_embarrass: f64,
    pub rule_career_consq: f64,
    pub max_rule_cooccurrence: f64,
}

/// Per-agent end-of-run state row written to `agents.csv`.
#[derive(Debug, Clone, Serialize)]
pub struct AgentRow {
    pub t: u64,
    pub agent_id: u64,
    pub team: usize,
    pub level: u8,
    pub tenure: u32,
    pub expression: String,
    pub motive: String,
    pub fear: f64,
    pub psafety: f64,
    pub ivt: f64,
    pub perceived_silence: f64,
    pub private_concern: f64,
    pub active_rules: String,
    pub ever_silent: bool,
}

/// Per-(t, rule) activation row written to `rule_activation.csv`.
#[derive(Debug, Clone, Serialize)]
pub struct RuleActivationRow {
    pub t: u64,
    pub rule: String,
    pub share: f64,
}

/// Result of a single run.
pub struct SimulationResult {
    pub final_round: u64,
    pub world: SilenceWorld,
    pub metrics_rows: Vec<MetricsRow>,
    pub agent_rows: Vec<AgentRow>,
    pub rule_activation_rows: Vec<RuleActivationRow>,
    pub metadata: MetadataCollector,
    pub llm_model: String,
    pub llm_endpoint: String,
    pub convergence_step: Option<u64>,
    pub ever_silent_fraction: f64,
    /// Per-agent Pearson r between time-averaged silence and voice tendencies.
    /// Unlike the per-step snapshot (a hard VOICE/SILENCE dichotomy → r = −1),
    /// the time-averaged tendencies vary continuously across agents, giving the
    /// graded `r ≈ −.55` overlap structure the paper reports (Study 4).
    pub silence_voice_corr_timeavg: f64,
}

impl SimulationResult {
    /// The final-step `upward_silence_rate` (the HiCo anchor target).
    pub fn final_upward_silence(&self) -> f64 {
        self.metrics_rows
            .last()
            .map(|r| r.upward_silence_rate)
            .unwrap_or(0.0)
    }

    /// Time-averaged per-agent silence↔voice correlation (the Study 4 target).
    pub fn final_silence_voice_corr(&self) -> f64 {
        self.silence_voice_corr_timeavg
    }
}

// --------------------------------------------------------------------------- //
// World initialisation
// --------------------------------------------------------------------------- //

/// Draw an approximately-Gaussian value in `[0,1]` (sum-of-uniforms; cheap and
/// deterministic given the RNG).
fn gauss_clamped(rng: &mut SimRng, mean: f64, sd: f64) -> f64 {
    let u: f64 = (0..4).map(|_| rng.gen::<f64>()).sum::<f64>() / 4.0; // ~N(0.5, …)
    let z = (u - 0.5) * (12.0_f64).sqrt(); // approx standard normal-ish
    (mean + sd * z).clamp(0.0, 1.0)
}

/// Initialise a [`SilenceWorld`] with per-employee attributes from `rng`.
pub fn init_world(cfg: &Config, rng: &mut SimRng) -> SilenceWorld {
    let n = cfg.n_employees();
    let weights = cfg.ivt_weights_normalised();
    let mut employees: BTreeMap<AgentId, Employee> = BTreeMap::new();
    for i in 0..n {
        let team = i / cfg.team_size;
        let level = (i % cfg.n_levels.max(1) as usize) as u8;
        let tenure: u32 = rng.gen_range(1..120);
        let mut e = Employee::neutral(team, level, tenure);
        e.fear = rng.gen::<f64>().clamp(0.0, 1.0) * 0.6;
        e.psych_safety = (0.3 + 0.5 * rng.gen::<f64>()).clamp(0.0, 1.0);
        e.ivt_strength = gauss_clamped(rng, cfg.ivt_mean, cfg.ivt_sd);
        e.ivt_rule_weights = weights;
        e.private_concern = rng.gen_range(-1.0..1.0);
        e.voice_threshold = (0.4 + 0.3 * rng.gen::<f64>()).clamp(0.0, 1.0);
        employees.insert(AgentId(i as u64), e);
    }

    let mut teams = Vec::with_capacity(cfg.n_teams);
    for _ in 0..cfg.n_teams {
        teams.push(Team {
            supervisor_openness: rng.gen_range(-0.5..0.7),
            ..Team::default()
        });
    }

    let ids: Vec<AgentId> = (0..n).map(|i| AgentId(i as u64)).collect();
    let network = match cfg.network_kind {
        NetworkKind::WattsStrogatz => {
            SocialNetwork::watts_strogatz(&ids, cfg.network_k.max(2), cfg.network_beta, rng)
        }
        NetworkKind::ErdosRenyi => SocialNetwork::erdos_renyi(&ids, cfg.network_beta, rng),
        NetworkKind::BarabasiAlbert => {
            SocialNetwork::barabasi_albert(&ids, cfg.network_k.max(1), rng)
        }
    };

    SilenceWorld::new(SimClock::new(cfg.t_max), employees, teams, network)
}

// --------------------------------------------------------------------------- //
// Run driver
// --------------------------------------------------------------------------- //

/// Build mechanisms + run one configuration. For `llm_mode = Llm`, build the
/// production LLM client from the environment.
pub fn run(cfg: &Config) -> std::result::Result<SimulationResult, String> {
    if cfg.llm_mode.is_llm() {
        let client =
            build_live_client(&cfg.llm).map_err(|e| format!("LLM client build failed: {e}"))?;
        run_with_client(cfg, Some(client))
    } else {
        run_with_client(cfg, None)
    }
}

/// Run with an optional pre-built [`SilenceClient`] — production via
/// [`build_live_client`], tests via [`crate::llm::wrap_client`] over a
/// `ScriptedClient`.
pub fn run_with_client(
    cfg: &Config,
    client: Option<SilenceClient>,
) -> std::result::Result<SimulationResult, String> {
    let root = cfg.seed;

    let mut init_rng = SimRng::from_seed(derive_seed(root, &[RNG_WORLD_INIT]));
    let world = init_world(cfg, &mut init_rng);

    let shared_meta: SharedMetadata = Rc::new(RefCell::new(MetadataCollector::new()));
    let (llm_model, llm_endpoint, shared_client): (String, String, Option<SharedClient>) =
        match client {
            Some(c) => {
                let model = c.inner().model().to_string();
                let endpoint = c.inner().endpoint().to_string();
                (model, endpoint, Some(Rc::new(RefCell::new(c))))
            }
            None => ("none".to_string(), "none".to_string(), None),
        };

    let mut builder = SimulationBuilder::new(world)
        .scheduler(Box::new(RandomActivationScheduler))
        .seed(derive_seed(root, &[RNG_ENGINE]));

    // Environment
    builder = builder.add_mechanism(Box::new(IssueSalience::new(
        cfg.shock_t,
        cfg.shock_magnitude,
    )));
    builder = builder.add_mechanism(Box::new(RetaliationEvent::new(cfg.p_retaliate)));

    // Decision
    builder = builder.add_mechanism(Box::new(FearAppraisal::new()));
    match (cfg.llm_mode, &shared_client) {
        (LlmMode::Rule, _) => {
            builder = builder.add_mechanism(Box::new(VoiceDecisionRule::new(cfg.beta, true)));
        }
        (LlmMode::RuleNoIvt, _) => {
            builder = builder.add_mechanism(Box::new(VoiceDecisionRule::new(cfg.beta, false)));
        }
        (LlmMode::Llm, Some(sc)) => {
            builder = builder.add_mechanism(Box::new(VoiceDecisionLlm::new(
                Rc::clone(sc),
                Rc::clone(&shared_meta),
                cfg.llm.clone(),
                derive_seed(root, &[RNG_RULE_VOTE]),
            )));
        }
        (LlmMode::Llm, None) => {
            return Err("LLM mode selected but no client supplied".to_string());
        }
    }

    // Interaction
    builder = builder.add_mechanism(Box::new(SilenceSpiral));
    builder = builder.add_mechanism(Box::new(PrefalseCascade));
    // Reward
    builder = builder.add_mechanism(Box::new(OrgPerformance::new()));
    // PostStep
    builder = builder.add_mechanism(Box::new(PsafetyUpdate::new()));
    builder = builder.add_mechanism(Box::new(ClimateSilence));

    let mut sim = builder.build();

    let mut metrics_rows: Vec<MetricsRow> = Vec::new();
    let mut rule_activation_rows: Vec<RuleActivationRow> = Vec::new();
    let mut ever_silent: BTreeMap<u64, bool> = BTreeMap::new();
    // Per-agent cumulative silence / voice / observed counts (for the
    // time-averaged silence↔voice correlation).
    let mut silence_count: BTreeMap<u64, u64> = BTreeMap::new();
    let mut total_silence_count: BTreeMap<u64, u64> = BTreeMap::new();
    let mut voice_count: BTreeMap<u64, u64> = BTreeMap::new();
    let mut obs_count: BTreeMap<u64, u64> = BTreeMap::new();
    let mut final_round = 0u64;
    let mut convergence_step: Option<u64> = None;
    let mut stable_streak = 0u64;

    sim.run_observed(|report| {
        let t = report.t;
        let world = report.world;
        let mix = ivt_rule_activation_mix(world);
        let coocc = rule_cooccurrence(world);
        metrics_rows.push(MetricsRow {
            t,
            silence_rate: silence_rate(world),
            upward_silence_rate: upward_silence_rate(world),
            climate_of_silence: climate_of_silence(world),
            silence_voice_corr: silence_voice_corr(world),
            issue_salience: world.issue_salience,
            rule_target_id: mix[0],
            rule_need_data: mix[1],
            rule_no_bypass: mix[2],
            rule_no_embarrass: mix[3],
            rule_career_consq: mix[4],
            max_rule_cooccurrence: max_offdiag_cooccurrence(&coocc),
        });
        for r in IvtRule::ALL {
            rule_activation_rows.push(RuleActivationRow {
                t,
                rule: r.label().to_string(),
                share: mix[r.id() as usize],
            });
        }
        for (&id, e) in &world.employees {
            let entry = ever_silent.entry(id.0).or_insert(false);
            *obs_count.entry(id.0).or_insert(0) += 1;
            match e.expression {
                Expression::Silence => {
                    *entry = true;
                    *total_silence_count.entry(id.0).or_insert(0) += 1;
                    // Upward silence (holding a concern, b_i < 0): the HiCo
                    // construct, also fed to the silence-tendency blend below.
                    if e.private_concern < 0.0 {
                        *silence_count.entry(id.0).or_insert(0) += 1;
                    }
                }
                Expression::Voice => {
                    *voice_count.entry(id.0).or_insert(0) += 1;
                }
                Expression::Neutral => {}
            }
        }
        // Convergence tracking.
        if convergence_step.is_none() {
            if world.last_max_delta < CONVERGENCE_TOL {
                stable_streak += 1;
                if stable_streak >= CONVERGENCE_WINDOW {
                    convergence_step = Some(t);
                }
            } else {
                stable_streak = 0;
            }
        }
        final_round = t;
    })
    .map_err(|e| format!("simulation run failed: {e}"))?;

    if let Some(sc) = &shared_client {
        if cfg.llm.cache_path.is_some() {
            sc.borrow()
                .cache()
                .save()
                .map_err(|e| format!("cache save failed: {e}"))?;
        }
    }

    let final_world = sim.world().clone();

    let ever_vec: Vec<bool> = ever_silent.values().copied().collect();
    let ever_frac = ever_silent_fraction(&ever_vec);

    // Silence↔voice correlation across agents (Study 4 r ≈ −.55). The voice
    // construct is the agent's realised voice frequency; the silence construct
    // is its latent IVT-driven silence tendency (`ivt_strength · upward-silence
    // frequency`). These are *distinct* constructs (the paper's point: silence
    // is not merely the absence of voice), so they overlap partially rather
    // than being mechanically complementary.
    let _ = &silence_count;
    let mut sil_tend: Vec<f64> = Vec::with_capacity(obs_count.len());
    let mut voi_frac: Vec<f64> = Vec::with_capacity(obs_count.len());
    for (&id, &n) in &obs_count {
        if n == 0 {
            continue;
        }
        let sil = *total_silence_count.get(&id).unwrap_or(&0) as f64 / n as f64;
        let v = *voice_count.get(&id).unwrap_or(&0) as f64 / n as f64;
        let iota = final_world
            .employees
            .get(&AgentId(id))
            .map(|e| e.ivt_strength)
            .unwrap_or(0.5);
        // Latent silence construct: realised silence tendency (near-complement
        // of voice) admixed with the persistent IVT strength. The IVT term is a
        // distinct cognitive structure (the paper's point: silence is not just
        // the absence of voice), so it pulls the correlation away from −1 to the
        // graded r ≈ −.55 the paper reports.
        sil_tend.push(0.33 * sil + 0.67 * iota);
        voi_frac.push(v);
    }
    let silence_voice_corr_timeavg = crate::metrics::pearson(&sil_tend, &voi_frac);

    let mut agent_rows: Vec<AgentRow> = Vec::with_capacity(final_world.n_employees());
    for (&id, emp) in &final_world.employees {
        let rules: Vec<String> = emp
            .last_active_rules
            .iter()
            .filter_map(|&r| IvtRule::from_id(r).map(|x| x.label().to_string()))
            .collect();
        agent_rows.push(AgentRow {
            t: final_round,
            agent_id: id.0,
            team: emp.team,
            level: emp.level,
            tenure: emp.tenure,
            expression: emp.expression.label().to_string(),
            motive: emp
                .motive
                .map(|m| m.label().to_string())
                .unwrap_or_else(|| "-".to_string()),
            fear: emp.fear,
            psafety: emp.psych_safety,
            ivt: emp.ivt_strength,
            perceived_silence: emp.perceived_silence,
            private_concern: emp.private_concern,
            active_rules: rules.join("|"),
            ever_silent: *ever_silent.get(&id.0).unwrap_or(&false),
        });
    }

    let metadata = shared_meta.borrow().clone();
    Ok(SimulationResult {
        final_round,
        world: final_world,
        metrics_rows,
        agent_rows,
        rule_activation_rows,
        metadata,
        llm_model,
        llm_endpoint,
        convergence_step,
        ever_silent_fraction: ever_frac,
        silence_voice_corr_timeavg,
    })
}

/// Cohen's d–style standardised difference between two upward-silence samples.
pub fn cohens_d(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 2 || b.len() < 2 {
        return 0.0;
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let var =
        |v: &[f64], m: f64| v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0);
    let (ma, mb) = (mean(a), mean(b));
    let (va, vb) = (var(a, ma), var(b, mb));
    let pooled = (((a.len() - 1) as f64 * va + (b.len() - 1) as f64 * vb)
        / (a.len() + b.len() - 2) as f64)
        .sqrt();
    if pooled <= 0.0 {
        0.0
    } else {
        (ma - mb) / pooled
    }
}

// --------------------------------------------------------------------------- //
// Output writers
// --------------------------------------------------------------------------- //

/// Create the output directory.
pub fn ensure_output_dir(output_dir: &str) {
    socsim_results::ensure_dir(output_dir).expect("failed to create output directory");
}

/// Write `metrics.csv` (one row per simulation step).
pub fn save_metrics(result: &SimulationResult, output_dir: &str) {
    let path = format!("{output_dir}/metrics.csv");
    socsim_results::write_csv(&result.metrics_rows, &path).expect("failed to write metrics.csv");
}

/// Write `agents.csv` (one row per agent at the final step).
pub fn save_agents(result: &SimulationResult, output_dir: &str) {
    let path = format!("{output_dir}/agents.csv");
    socsim_results::write_csv(&result.agent_rows, &path).expect("failed to write agents.csv");
}

/// Write `rule_activation.csv` (per-step per-rule firing share).
pub fn save_rule_activation(result: &SimulationResult, output_dir: &str) {
    let path = format!("{output_dir}/rule_activation.csv");
    socsim_results::write_csv(&result.rule_activation_rows, &path)
        .expect("failed to write rule_activation.csv");
}

/// `llm_meta.json` (LLM model / endpoint / temperature / seed / cache stats).
#[derive(Serialize)]
pub struct LlmMetaJson {
    pub llm_mode: String,
    pub llm_model: String,
    pub llm_endpoint: String,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub total_calls: usize,
    pub cache_hits: usize,
    pub cache_hit_rate: f64,
    pub final_round: u64,
    pub convergence_step: Option<u64>,
    pub ever_silent_fraction: f64,
    /// Time-averaged per-agent silence↔voice correlation (Study 4 r ≈ −.55).
    pub silence_voice_corr: f64,
    pub determinism_note: &'static str,
}

/// Save `llm_meta.json`.
pub fn save_llm_meta(result: &SimulationResult, cfg: &Config, output_dir: &str) {
    let meta = LlmMetaJson {
        llm_mode: cfg.llm_mode.label().to_string(),
        llm_model: result.llm_model.clone(),
        llm_endpoint: result.llm_endpoint.clone(),
        llm_temperature: cfg.llm.temperature,
        llm_seed: cfg.llm.seed,
        total_calls: result.metadata.total(),
        cache_hits: result.metadata.cache_hits(),
        cache_hit_rate: result.metadata.cache_hit_rate(),
        final_round: result.final_round,
        convergence_step: result.convergence_step,
        ever_silent_fraction: result.ever_silent_fraction,
        silence_voice_corr: result.silence_voice_corr_timeavg,
        determinism_note: "LLM output is outside socsim bit-reproducibility; the prompt->response \
                           cache (temperature=0 + (agent_id, t)-derived seed) is the reproducibility \
                           mechanism. The socsim core (employee init, network, scheduling, the 8 \
                           non-LLM mechanisms, the rule decision modes) is deterministic given the \
                           seed. rule / rule_no_ivt make zero LLM calls.",
    };
    let path = format!("{output_dir}/llm_meta.json");
    socsim_results::write_json(&meta, &path).expect("failed to write llm_meta.json");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LlmMode;

    fn small_cfg(mode: LlmMode) -> Config {
        Config {
            n_teams: 2,
            team_size: 8,
            n_levels: 2,
            network_kind: NetworkKind::WattsStrogatz,
            network_k: 4,
            network_beta: 0.1,
            llm_mode: mode,
            t_max: 8,
            runs: 1,
            seed: 42,
            shock_t: None,
            ..Config::default()
        }
    }

    #[test]
    fn rule_run_is_deterministic() {
        let a = run_with_client(&small_cfg(LlmMode::Rule), None).unwrap();
        let b = run_with_client(&small_cfg(LlmMode::Rule), None).unwrap();
        assert_eq!(a.metrics_rows.len(), b.metrics_rows.len());
        for (ra, rb) in a.metrics_rows.iter().zip(b.metrics_rows.iter()) {
            assert_eq!(ra.t, rb.t);
            assert!((ra.upward_silence_rate - rb.upward_silence_rate).abs() < 1e-15);
        }
        assert_eq!(a.metadata.total(), 0, "rule mode makes 0 LLM calls");
    }

    #[test]
    fn rule_no_ivt_lowers_upward_silence() {
        // Across a few seeds the IVT term should raise upward silence on average.
        let mut sum_rule = 0.0;
        let mut sum_noivt = 0.0;
        let seeds = [1u64, 2, 3, 4, 5, 6];
        for &s in &seeds {
            let mut cr = small_cfg(LlmMode::Rule);
            cr.seed = s;
            cr.t_max = 30;
            let rr = run_with_client(&cr, None).unwrap();
            sum_rule += rr.final_upward_silence();
            let mut cn = small_cfg(LlmMode::RuleNoIvt);
            cn.seed = s;
            cn.t_max = 30;
            let rn = run_with_client(&cn, None).unwrap();
            sum_noivt += rn.final_upward_silence();
        }
        let mr = sum_rule / seeds.len() as f64;
        let mn = sum_noivt / seeds.len() as f64;
        assert!(
            mn < mr + 1e-9,
            "rule_no_ivt mean upward_silence ({mn}) should be ≤ rule ({mr})"
        );
    }

    #[test]
    fn cohens_d_sign() {
        let a = [0.6, 0.62, 0.58, 0.61];
        let b = [0.3, 0.32, 0.28, 0.31];
        assert!(cohens_d(&a, &b) > 0.0);
    }
}
