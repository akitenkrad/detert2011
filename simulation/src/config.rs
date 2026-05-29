//! Simulation configuration for the Detert & Edmondson (2011) IVT model.
//!
//! Holds all knobs surfaced by the `run` / `sweep` / `ablation` CLI: org /
//! network shape, the rule-mode logit `β` group (including the IVT term
//! `β_ι`), the per-employee IVT mean / sd / rule-weight vector, retaliation /
//! shock parameters, and the LLM settings used when `llm_mode == Llm`.

use serde::Serialize;

// --------------------------------------------------------------------------- //
// LlmMode — LLM-driven vs rule-based ablation vs IVT-ablated rule
// --------------------------------------------------------------------------- //

/// Decision-mechanism selector (mutually exclusive, like mou2024's mode switch).
///
/// The driver wires **exactly one** decision mechanism:
/// - `Llm`       → `voice_decision` (LLM with the 5-rule self-reflection layer)
/// - `Rule`      → `voice_decision_rule` logit with the full `−β_ι·ι_i` term
/// - `RuleNoIvt` → same logit with `β_ι` forced to 0 (the IVT ablation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmMode {
    /// `voice_decision` — LLM-driven (5-rule self-reflection).
    Llm,
    /// `voice_decision_rule` — logit ablation with the IVT term.
    Rule,
    /// `voice_decision_rule` — logit ablation with `β_ι = 0` (IVT ablated).
    RuleNoIvt,
}

impl LlmMode {
    /// Stable snake_case label (CSV / JSON / directory friendly).
    pub fn label(&self) -> &'static str {
        match self {
            LlmMode::Llm => "llm",
            LlmMode::Rule => "rule",
            LlmMode::RuleNoIvt => "rule_no_ivt",
        }
    }

    /// Whether this mode reaches the LLM layer.
    pub fn is_llm(&self) -> bool {
        matches!(self, LlmMode::Llm)
    }

    /// Whether the IVT logit term `β_ι` is active (false for the ablation).
    pub fn ivt_active(&self) -> bool {
        !matches!(self, LlmMode::RuleNoIvt)
    }
}

/// Parse an [`LlmMode`] from a CLI string.
pub fn parse_llm_mode(s: &str) -> Result<LlmMode, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "llm" | "ollama" | "openai" => Ok(LlmMode::Llm),
        "rule" | "rules" | "logit" => Ok(LlmMode::Rule),
        "rule_no_ivt" | "rule-no-ivt" | "no_ivt" | "ablation" => Ok(LlmMode::RuleNoIvt),
        _ => Err(format!(
            "invalid llm-mode: \"{s}\" (llm / rule / rule_no_ivt)"
        )),
    }
}

// --------------------------------------------------------------------------- //
// LLM settings (re-exported from socsim-llm)
// --------------------------------------------------------------------------- //

/// LLM-layer settings (`temperature`, `seed`, `cache_path`) — re-exported from
/// `socsim-llm::harness` so every replication shares one struct.
pub use socsim_llm::LlmSettings;

// --------------------------------------------------------------------------- //
// BetaGroup — logit coefficients for voice_decision_rule
// --------------------------------------------------------------------------- //

/// Coefficient group for the rule-mode VOICE logit (§4.3):
///
/// `P(VOICE) = σ(β0 + β_ψ ψ + β_u u + β_σ σ − β_f f − β_ι ι − β_C ρ + β_θ 1[ρ^V>θ])`.
///
/// `β_ι` is forced to 0 by `LlmMode::RuleNoIvt`, isolating the causal
/// contribution of the Implicit Voice Theory term.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct BetaGroup {
    /// Intercept `β0`.
    pub intercept: f64,
    /// `β_ψ` — psychological safety (positive for VOICE).
    pub beta_psafety: f64,
    /// `β_u` — supervisor openness (positive for VOICE).
    pub beta_supervisor: f64,
    /// `β_σ` — issue salience (positive for VOICE).
    pub beta_salience: f64,
    /// `β_f` — fear (subtracted in the VOICE logit).
    pub beta_fear: f64,
    /// `β_ι` — IVT strength (subtracted; the IVT main effect).
    pub beta_ivt: f64,
    /// `β_C` — perceived peer silence ρ (subtracted; spiral-of-silence).
    pub beta_climate: f64,
    /// `β_θ` — preference-falsification cascade bonus when ρ^V > θ.
    pub beta_threshold: f64,
}

impl Default for BetaGroup {
    fn default() -> Self {
        // Calibrated so that ι̅=0.55 with β_ι=0.8 lands upward_silence_rate near
        // the HiCo 50% anchor, while β_ι=0 drops it below ~0.40.
        BetaGroup {
            intercept: -0.7,
            beta_psafety: 1.2,
            beta_supervisor: 0.6,
            beta_salience: 0.4,
            beta_fear: 1.5,
            beta_ivt: 2.0,
            beta_climate: 1.0,
            beta_threshold: 0.4,
        }
    }
}

// --------------------------------------------------------------------------- //
// NetworkKind
// --------------------------------------------------------------------------- //

/// Inter-employee network family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NetworkKind {
    /// Watts–Strogatz small-world (default — design §4.3).
    WattsStrogatz,
    /// Erdős–Rényi G(n,p) — sensitivity.
    ErdosRenyi,
    /// Barabási–Albert preferential attachment — sensitivity.
    BarabasiAlbert,
}

/// Parse a [`NetworkKind`] from a CLI string.
pub fn parse_network_kind(s: &str) -> Result<NetworkKind, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "ws" | "watts-strogatz" | "small-world" => Ok(NetworkKind::WattsStrogatz),
        "er" | "erdos-renyi" | "erdos_renyi" => Ok(NetworkKind::ErdosRenyi),
        "ba" | "barabasi-albert" | "scale-free" => Ok(NetworkKind::BarabasiAlbert),
        _ => Err(format!(
            "invalid network kind: \"{s}\" (watts-strogatz / erdos-renyi / barabasi-albert)"
        )),
    }
}

// --------------------------------------------------------------------------- //
// Config
// --------------------------------------------------------------------------- //

/// Configuration for a single run.
#[derive(Debug, Clone)]
pub struct Config {
    // ── organisation shape ─────────────────────────────────────────────────
    pub n_teams: usize,
    pub team_size: usize,
    /// Number of hierarchical levels (`level = i % n_levels`).
    pub n_levels: u8,

    // ── network ────────────────────────────────────────────────────────────
    pub network_kind: NetworkKind,
    pub network_k: usize,
    pub network_beta: f64,

    // ── decision-mode switch ───────────────────────────────────────────────
    pub llm_mode: LlmMode,

    // ── IVT distribution ───────────────────────────────────────────────────
    /// Mean of the per-employee IVT strength ι_i draw.
    pub ivt_mean: f64,
    /// Std-dev of the per-employee IVT strength ι_i draw.
    pub ivt_sd: f64,
    /// Per-rule IVT weights `w` (the simplex applied to every employee at init).
    pub ivt_weights: [f64; 5],

    // ── logit β group ──────────────────────────────────────────────────────
    pub beta: BetaGroup,

    // ── retaliation / shocks ───────────────────────────────────────────────
    pub p_retaliate: f64,
    pub shock_t: Option<u64>,
    pub shock_magnitude: f64,

    // ── horizon / repeats ──────────────────────────────────────────────────
    pub t_max: u64,
    pub runs: usize,
    pub seed: u64,

    // ── LLM settings (used iff `llm_mode == Llm`) ──────────────────────────
    pub llm: LlmSettings,

    // ── output ─────────────────────────────────────────────────────────────
    pub output_dir: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            n_teams: 8,
            team_size: 25,
            n_levels: 3,
            network_kind: NetworkKind::WattsStrogatz,
            network_k: 6,
            network_beta: 0.1,
            llm_mode: LlmMode::Rule,
            ivt_mean: 0.55,
            ivt_sd: 0.20,
            ivt_weights: [0.20, 0.22, 0.18, 0.20, 0.20],
            beta: BetaGroup::default(),
            p_retaliate: 0.05,
            shock_t: Some(24),
            shock_magnitude: 0.3,
            t_max: 60,
            runs: 1,
            seed: 42,
            llm: LlmSettings::default(),
            output_dir: "results".to_string(),
        }
    }
}

impl Config {
    /// Total number of employees.
    pub fn n_employees(&self) -> usize {
        self.n_teams.saturating_mul(self.team_size)
    }

    /// L1-normalised IVT weight simplex (falls back to uniform if degenerate).
    pub fn ivt_weights_normalised(&self) -> [f64; 5] {
        let s: f64 = self.ivt_weights.iter().map(|w| w.max(0.0)).sum();
        if s <= 0.0 {
            return [0.2; 5];
        }
        let mut out = [0.0; 5];
        for (o, w) in out.iter_mut().zip(self.ivt_weights.iter()) {
            *o = w.max(0.0) / s;
        }
        out
    }
}

/// JSON representation of a `run`'s `config.json`.
#[derive(Serialize)]
pub struct RunConfigJson {
    pub command: &'static str,
    pub n_teams: usize,
    pub team_size: usize,
    pub n_levels: u8,
    pub n_employees: usize,
    pub network_kind: NetworkKind,
    pub network_k: usize,
    pub network_beta: f64,
    pub llm_mode: LlmMode,
    pub ivt_mean: f64,
    pub ivt_sd: f64,
    pub ivt_weights: [f64; 5],
    pub beta: BetaGroup,
    pub p_retaliate: f64,
    pub shock_t: Option<u64>,
    pub shock_magnitude: f64,
    pub t_max: u64,
    pub runs: usize,
    pub seed: u64,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub llm_cache_path: Option<String>,
    pub output_dir: String,
}

impl Config {
    /// Build the `config.json` representation.
    pub fn to_run_config_json(&self) -> RunConfigJson {
        RunConfigJson {
            command: "run",
            n_teams: self.n_teams,
            team_size: self.team_size,
            n_levels: self.n_levels,
            n_employees: self.n_employees(),
            network_kind: self.network_kind,
            network_k: self.network_k,
            network_beta: self.network_beta,
            llm_mode: self.llm_mode,
            ivt_mean: self.ivt_mean,
            ivt_sd: self.ivt_sd,
            ivt_weights: self.ivt_weights,
            beta: self.beta,
            p_retaliate: self.p_retaliate,
            shock_t: self.shock_t,
            shock_magnitude: self.shock_magnitude,
            t_max: self.t_max,
            runs: self.runs,
            seed: self.seed,
            llm_temperature: self.llm.temperature,
            llm_seed: self.llm.seed,
            llm_cache_path: self.llm.cache_path.clone(),
            output_dir: self.output_dir.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_llm_mode_variants() {
        assert_eq!(parse_llm_mode("llm").unwrap(), LlmMode::Llm);
        assert_eq!(parse_llm_mode("rule").unwrap(), LlmMode::Rule);
        assert_eq!(parse_llm_mode("RULE_NO_IVT").unwrap(), LlmMode::RuleNoIvt);
        assert!(parse_llm_mode("bogus").is_err());
    }

    #[test]
    fn ivt_active_flags() {
        assert!(LlmMode::Rule.ivt_active());
        assert!(LlmMode::Llm.ivt_active());
        assert!(!LlmMode::RuleNoIvt.ivt_active());
    }

    #[test]
    fn ivt_weights_normalise() {
        let cfg = Config {
            ivt_weights: [1.0, 1.0, 1.0, 1.0, 1.0],
            ..Config::default()
        };
        let w = cfg.ivt_weights_normalised();
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        for v in w {
            assert!((v - 0.2).abs() < 1e-12);
        }
    }

    #[test]
    fn parse_network_kind_variants() {
        assert_eq!(
            parse_network_kind("watts-strogatz").unwrap(),
            NetworkKind::WattsStrogatz
        );
        assert_eq!(parse_network_kind("ER").unwrap(), NetworkKind::ErdosRenyi);
        assert_eq!(
            parse_network_kind("ba").unwrap(),
            NetworkKind::BarabasiAlbert
        );
    }
}
