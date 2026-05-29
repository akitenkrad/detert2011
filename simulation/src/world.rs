//! World state for the Detert & Edmondson (2011) Implicit Voice Theory model.
//!
//! Implements socsim's [`WorldState`] over employees living on a
//! [`SocialNetwork`] (Watts–Strogatz organisational small-world). Each employee
//! carries a persistent IVT cognitive structure (`ivt_strength` ι_i plus a
//! 5-vector of per-rule weights `ivt_rule_weights`) and records, each step,
//! which of the five IVT rules fired (`last_active_rules`).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use socsim_core::{AgentId, SimClock, WorldState};
use socsim_net::SocialNetwork;

// --------------------------------------------------------------------------- //
// IVT rules
// --------------------------------------------------------------------------- //

/// The five Implicit Voice Theory rules (Detert & Edmondson 2011).
///
/// Canonical order: `target_id`, `need_data`, `no_bypass`, `no_embarrass`,
/// `career_consq`. The `u8` `id()` (0..5) indexes `Employee::ivt_rule_weights`
/// and the `rule_activation` / `rule_cooccurrence` metrics.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IvtRule {
    /// Presumed target identification — the boss takes input as personal attack.
    TargetId,
    /// Need solid data or a finished solution before speaking.
    NeedData,
    /// Don't bypass the boss upward.
    NoBypass,
    /// Don't embarrass the boss in public.
    NoEmbarrass,
    /// Negative career consequences of speaking up.
    CareerConsq,
}

impl IvtRule {
    /// All five rules in canonical order.
    pub const ALL: [IvtRule; 5] = [
        IvtRule::TargetId,
        IvtRule::NeedData,
        IvtRule::NoBypass,
        IvtRule::NoEmbarrass,
        IvtRule::CareerConsq,
    ];

    /// Stable rule id `0..5` (indexes `ivt_rule_weights` + metrics vectors).
    pub fn id(&self) -> u8 {
        match self {
            IvtRule::TargetId => 0,
            IvtRule::NeedData => 1,
            IvtRule::NoBypass => 2,
            IvtRule::NoEmbarrass => 3,
            IvtRule::CareerConsq => 4,
        }
    }

    /// Construct from a `0..5` id (`None` out of range).
    pub fn from_id(id: u8) -> Option<IvtRule> {
        IvtRule::ALL.get(id as usize).copied()
    }

    /// Stable snake_case label (CSV / JSON friendly).
    pub fn label(&self) -> &'static str {
        match self {
            IvtRule::TargetId => "target_id",
            IvtRule::NeedData => "need_data",
            IvtRule::NoBypass => "no_bypass",
            IvtRule::NoEmbarrass => "no_embarrass",
            IvtRule::CareerConsq => "career_consq",
        }
    }
}

// --------------------------------------------------------------------------- //
// Motive / Expression
// --------------------------------------------------------------------------- //

/// Silence motive (Van Dyne et al. 2003 taxonomy; `None` when expressing VOICE).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Motive {
    /// Resigned withdrawal (gave-up).
    Acquiescent,
    /// Fear-based self-protective withholding.
    Defensive,
    /// Other-protective withholding.
    Prosocial,
}

impl Motive {
    /// Stable lowercase label.
    pub fn label(&self) -> &'static str {
        match self {
            Motive::Acquiescent => "acquiescent",
            Motive::Defensive => "defensive",
            Motive::Prosocial => "prosocial",
        }
    }

    /// Parse a lowercase motive label.
    pub fn parse(s: &str) -> Option<Motive> {
        match s.trim().to_ascii_lowercase().as_str() {
            "acquiescent" | "as" => Some(Motive::Acquiescent),
            "defensive" | "quiescent" | "qs" => Some(Motive::Defensive),
            "prosocial" | "ps" => Some(Motive::Prosocial),
            _ => None,
        }
    }
}

/// Public expression at step `t`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Expression {
    Voice,
    Silence,
    Neutral,
}

impl Expression {
    pub fn label(&self) -> &'static str {
        match self {
            Expression::Voice => "voice",
            Expression::Silence => "silence",
            Expression::Neutral => "neutral",
        }
    }
}

// --------------------------------------------------------------------------- //
// Employee / Team
// --------------------------------------------------------------------------- //

/// Per-employee state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Employee {
    /// Hierarchical level `ℓ_i` (`0` = lowest).
    pub level: u8,
    /// Tenure in months `τ_i`.
    pub tenure: u32,
    /// Team membership index.
    pub team: usize,
    /// Private concern intensity `b_i ∈ [-1, 1]` (negative = disagreeing).
    pub private_concern: f64,
    /// Current public expression `b̂_i`.
    pub expression: Expression,
    /// Silence motive when `expression == Silence`; `None` otherwise.
    pub motive: Option<Motive>,
    /// Fear `f_i ∈ [0, 1]` (Kish-Gephart 2009).
    pub fear: f64,
    /// Psychological safety `ψ_i ∈ [0, 1]` (Edmondson 1999).
    pub psych_safety: f64,
    /// IVT integrated strength `ι_i ∈ [0, 1]` (5-rule aggregate).
    pub ivt_strength: f64,
    /// Per-rule IVT weights `w_{i,r}` over the 5 rules (a simplex by convention).
    pub ivt_rule_weights: [f64; 5],
    /// Perceived neighbour silence ratio `ρ_i ∈ [0, 1]` (silence_spiral).
    pub perceived_silence: f64,
    /// VOICE threshold `θ_i ∈ [0, 1]` (Kuran 1995).
    pub voice_threshold: f64,
    /// IVT rule ids `0..5` that fired in the most recent decision.
    pub last_active_rules: Vec<u8>,
}

impl Employee {
    /// Initialise a "neutral" employee with defaults; per-attribute random draws
    /// happen at the call site (`simulation::init_world`).
    pub fn neutral(team: usize, level: u8, tenure: u32) -> Self {
        Employee {
            level,
            tenure,
            team,
            private_concern: 0.0,
            expression: Expression::Neutral,
            motive: None,
            fear: 0.3,
            psych_safety: 0.5,
            ivt_strength: 0.55,
            ivt_rule_weights: [0.20, 0.22, 0.18, 0.20, 0.20],
            perceived_silence: 0.5,
            voice_threshold: 0.5,
            last_active_rules: Vec::new(),
        }
    }
}

/// Per-team state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Team {
    /// Supervisor openness `u_k ∈ [-1, 1]`.
    pub supervisor_openness: f64,
    /// Cumulative team knowledge stock `K_k(t)` (org_performance mechanism).
    pub knowledge_stock: f64,
    /// Team-level climate-of-silence proxy `C_k(t)` (climate_silence mechanism).
    pub climate: f64,
}

impl Default for Team {
    fn default() -> Self {
        Team {
            supervisor_openness: 0.0,
            knowledge_stock: 0.0,
            climate: 0.0,
        }
    }
}

// --------------------------------------------------------------------------- //
// SilenceWorld
// --------------------------------------------------------------------------- //

/// World state for the Implicit Voice Theory silence model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SilenceWorld {
    pub clock: SimClock,
    /// Employees keyed by sorted [`AgentId`] (sorted keys = determinism).
    pub employees: BTreeMap<AgentId, Employee>,
    pub teams: Vec<Team>,
    /// Inter-employee social network (Watts–Strogatz by default).
    pub network: SocialNetwork,
    /// Issue salience `σ(t) ∈ [0, 1]` (issue_salience mechanism).
    pub issue_salience: f64,
    /// Whole-organisation climate of silence `C(t)` (climate_silence mechanism).
    pub climate_of_silence: f64,
    /// Agents affected by retaliation in the current step
    /// (cleared at the start of each step by `retaliation_event`).
    pub retaliation_this_step: Vec<AgentId>,
    /// Largest `|C(t) − C(t-1)|` observed last step (convergence tracking).
    pub last_max_delta: f64,
}

impl SilenceWorld {
    /// Build a world from teams + an inter-employee network.
    pub fn new(
        clock: SimClock,
        employees: BTreeMap<AgentId, Employee>,
        teams: Vec<Team>,
        network: SocialNetwork,
    ) -> Self {
        SilenceWorld {
            clock,
            employees,
            teams,
            network,
            issue_salience: 0.5,
            climate_of_silence: 0.0,
            retaliation_this_step: Vec::new(),
            last_max_delta: f64::INFINITY,
        }
    }

    /// Total number of employees.
    pub fn n_employees(&self) -> usize {
        self.employees.len()
    }

    /// Perceived-silence ratio `ρ_i` over network neighbours. Isolated → 0.
    pub fn neighbour_silence_ratio(&self, id: AgentId) -> f64 {
        let neighbours = self.network.neighbors(id);
        if neighbours.is_empty() {
            return 0.0;
        }
        let mut silent = 0usize;
        for nb in &neighbours {
            if let Some(e) = self.employees.get(nb) {
                if e.expression == Expression::Silence {
                    silent += 1;
                }
            }
        }
        silent as f64 / neighbours.len() as f64
    }

    /// Voice ratio over network neighbours (used by `prefalse_cascade`).
    pub fn neighbour_voice_ratio(&self, id: AgentId) -> f64 {
        let neighbours = self.network.neighbors(id);
        if neighbours.is_empty() {
            return 0.0;
        }
        let mut voice = 0usize;
        for nb in &neighbours {
            if let Some(e) = self.employees.get(nb) {
                if e.expression == Expression::Voice {
                    voice += 1;
                }
            }
        }
        voice as f64 / neighbours.len() as f64
    }
}

impl WorldState for SilenceWorld {
    fn agent_ids(&self) -> Vec<AgentId> {
        // BTreeMap keys are already sorted — canonical activation order.
        self.employees.keys().copied().collect()
    }

    fn clock(&self) -> &SimClock {
        &self.clock
    }

    fn clock_mut(&mut self) -> &mut SimClock {
        &mut self.clock
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use socsim_core::SimRng;

    #[test]
    fn ivt_rule_ids_round_trip() {
        for r in IvtRule::ALL {
            assert_eq!(IvtRule::from_id(r.id()), Some(r));
        }
        assert_eq!(IvtRule::from_id(5), None);
    }

    #[test]
    fn ivt_rule_labels_unique() {
        let mut seen = std::collections::HashSet::new();
        for r in IvtRule::ALL {
            assert!(seen.insert(r.label()));
        }
    }

    #[test]
    fn motive_parse_round_trips() {
        for m in [Motive::Acquiescent, Motive::Defensive, Motive::Prosocial] {
            assert_eq!(Motive::parse(m.label()), Some(m));
        }
    }

    #[test]
    fn neighbour_ratios_isolated_is_zero() {
        let mut rng = SimRng::from_seed(7);
        let ids: Vec<AgentId> = (0..4).map(|i| AgentId(i as u64)).collect();
        let net = SocialNetwork::erdos_renyi(&ids, 0.0, &mut rng);
        let mut emps: BTreeMap<AgentId, Employee> = BTreeMap::new();
        for &id in &ids {
            emps.insert(id, Employee::neutral(0, 0, 0));
        }
        let world = SilenceWorld::new(SimClock::new(1), emps, vec![Team::default()], net);
        assert_eq!(world.neighbour_silence_ratio(AgentId(0)), 0.0);
        assert_eq!(world.neighbour_voice_ratio(AgentId(0)), 0.0);
    }
}
