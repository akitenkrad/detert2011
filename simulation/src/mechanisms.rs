//! 9 mechanisms across the socsim 6-phase loop.
//!
//! | # | Mechanism            | Phase       | Role |
//! |---|----------------------|-------------|------|
//! | 1 | `IssueSalience`      | Environment | Update σ(t); fire optional shock at `shock_t` |
//! | 2 | `RetaliationEvent`   | Environment | With probability `p_retaliate` mark agents touched by retaliation |
//! | 3 | `FearAppraisal`      | Decision    | Threat appraisal: `f_i ← clamp(f_i + α·retaliated − γ·max(u,0))` |
//! | 4 | `VoiceDecisionRule` / `VoiceDecisionLlm` | Decision | **★ mutually exclusive**: logit (rule / rule_no_ivt) vs LLM 5-rule self-reflection |
//! | 5 | `SilenceSpiral`      | Interaction | ρ_i ← neighbour silence ratio (Noelle-Neumann 1974) |
//! | 6 | `PrefalseCascade`    | Interaction | Silent agent flips to VOICE if neighbour-VOICE ratio > θ_i (Kuran 1995) |
//! | 7 | `OrgPerformance`     | Reward      | Team `K_k(t)` ← decay + voice contribution |
//! | 8 | `PsafetyUpdate`      | PostStep    | ψ_i ← ψ_i + η(voiced) − ν(retaliated) |
//! | 9 | `ClimateSilence`     | PostStep    | Aggregate C(t) + per-team climate; track convergence delta |
//!
//! The decision mechanisms **snapshot all employees at step start** and apply
//! the new expressions/motives/active_rules from the snapshot (synchronous
//! update, design §4.3 "Update semantics").

use std::cell::RefCell;
use std::rc::Rc;

use rand::Rng;
use socsim_core::{
    derive_seed, AgentId, Mechanism, Phase, Result, SocsimError, StepContext, WorldState,
};
use socsim_llm::MetadataCollector;

use crate::config::{BetaGroup, LlmSettings};
use crate::llm::{llm_config, SilenceClient};
use crate::prompts::{build_voice_prompt, parse_voice_decision};
use crate::world::{Expression, IvtRule, Motive, SilenceWorld};

// --------------------------------------------------------------------------- //
// Shared LLM client / metadata wrappers (mirrors knoll2013 / brinsfield2013)
// --------------------------------------------------------------------------- //

/// Shared LLM client between driver + mechanism (`Rc<RefCell>` pattern).
pub type SharedClient = Rc<RefCell<SilenceClient>>;
/// Shared metadata collector for cache-hit rate / call count.
pub type SharedMetadata = Rc<RefCell<MetadataCollector>>;

// --------------------------------------------------------------------------- //
// 1. IssueSalience  (Environment)
// --------------------------------------------------------------------------- //

/// Updates `world.issue_salience` with mild mean-reversion plus an optional
/// step-`shock_t` exogenous bump (e.g. an organisational problem surfacing).
pub struct IssueSalience {
    decay: f64,
    target: f64,
    shock_t: Option<u64>,
    shock_magnitude: f64,
}

impl IssueSalience {
    pub fn new(shock_t: Option<u64>, shock_magnitude: f64) -> Self {
        IssueSalience {
            decay: 0.10,
            target: 0.5,
            shock_t,
            shock_magnitude,
        }
    }
}

impl Mechanism<SilenceWorld> for IssueSalience {
    fn name(&self) -> &str {
        "issue_salience"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Environment]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let sigma = ctx.world.issue_salience;
        let mut new_sigma = sigma + self.decay * (self.target - sigma);
        if let Some(t_shock) = self.shock_t {
            if ctx.clock.t() == t_shock {
                new_sigma = (new_sigma + self.shock_magnitude).clamp(0.0, 1.0);
            }
        }
        ctx.world.issue_salience = new_sigma.clamp(0.0, 1.0);
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 2. RetaliationEvent  (Environment)
// --------------------------------------------------------------------------- //

/// With probability `p_retaliate` per agent, mark them retaliated this step.
pub struct RetaliationEvent {
    p_retaliate: f64,
}

impl RetaliationEvent {
    pub fn new(p_retaliate: f64) -> Self {
        RetaliationEvent {
            p_retaliate: p_retaliate.clamp(0.0, 1.0),
        }
    }
}

impl Mechanism<SilenceWorld> for RetaliationEvent {
    fn name(&self) -> &str {
        "retaliation_event"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Environment]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        ctx.world.retaliation_this_step.clear();
        if self.p_retaliate <= 0.0 {
            return Ok(());
        }
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        for id in ids {
            if ctx.rng.gen::<f64>() < self.p_retaliate {
                ctx.world.retaliation_this_step.push(id);
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 3. FearAppraisal  (Decision)
// --------------------------------------------------------------------------- //

/// Threat appraisal: `f_i ← clamp(f_i + α·retaliated − γ·max(u,0), 0, 1)`.
pub struct FearAppraisal {
    alpha: f64,
    gamma: f64,
}

impl FearAppraisal {
    pub fn new() -> Self {
        FearAppraisal {
            alpha: 0.30,
            gamma: 0.10,
        }
    }
}

impl Default for FearAppraisal {
    fn default() -> Self {
        Self::new()
    }
}

impl Mechanism<SilenceWorld> for FearAppraisal {
    fn name(&self) -> &str {
        "fear_appraisal"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let retaliated: std::collections::HashSet<AgentId> =
            ctx.world.retaliation_this_step.iter().copied().collect();
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        for id in ids {
            let team_idx = ctx.world.employees[&id].team;
            let u = ctx.world.teams[team_idx].supervisor_openness;
            let emp = ctx.world.employees.get_mut(&id).expect("agent missing");
            let r = if retaliated.contains(&id) { 1.0 } else { 0.0 };
            let new_f = (emp.fear + self.alpha * r - self.gamma * u.max(0.0)).clamp(0.0, 1.0);
            emp.fear = new_f;
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 4a. VoiceDecisionRule  (Decision) — logit ablation
// --------------------------------------------------------------------------- //

/// Sigmoid (logistic) function.
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Per-rule firing probability `Pr(rule r fires) = σ(k·(ι·w_r − mid))` for a
/// silent agent. Each rule is then sampled **independently** (using a supplied
/// uniform draw per rule), so the five rules fire *discriminantly* rather than
/// in lockstep — keeping the off-diagonal `rule_cooccurrence` below 0.50 (the
/// design's discriminant-validity criterion) while preserving the mean
/// activation ranking (need_data, the heaviest rule, fires most often).
fn rule_fire_prob(ivt_strength: f64, weight: f64) -> f64 {
    const SLOPE: f64 = 28.0;
    const MID: f64 = 0.115;
    sigmoid(SLOPE * (ivt_strength * weight - MID))
}

/// Sample the firing set from the per-rule probabilities and `u` uniforms.
fn sample_rule_fire_set(ivt_strength: f64, weights: &[f64; 5], u: &[f64; 5]) -> Vec<u8> {
    let mut out = Vec::new();
    for r in IvtRule::ALL {
        let p = rule_fire_prob(ivt_strength, weights[r.id() as usize]);
        if u[r.id() as usize] < p {
            out.push(r.id());
        }
    }
    out
}

/// `voice_decision_rule` mechanism — the §4.3 logit ablation.
///
/// `ivt_active = false` corresponds to `--llm-mode rule_no_ivt` (β_ι → 0).
pub struct VoiceDecisionRule {
    beta: BetaGroup,
    ivt_active: bool,
}

impl VoiceDecisionRule {
    pub fn new(beta: BetaGroup, ivt_active: bool) -> Self {
        VoiceDecisionRule { beta, ivt_active }
    }
}

/// Snapshot of the features the VOICE logit consumes for one agent.
struct RuleFeatures {
    id: AgentId,
    psafety: f64,
    supervisor: f64,
    salience: f64,
    fear: f64,
    ivt: f64,
    rho: f64,
    rho_voice: f64,
    voice_threshold: f64,
    private_concern: f64,
    ivt_weights: [f64; 5],
}

impl Mechanism<SilenceWorld> for VoiceDecisionRule {
    fn name(&self) -> &str {
        "voice_decision_rule"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        // Synchronous update: snapshot first, then write the new states.
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        let mut snapshot: Vec<RuleFeatures> = Vec::with_capacity(ids.len());
        for id in &ids {
            let emp = &ctx.world.employees[id];
            let team = &ctx.world.teams[emp.team];
            snapshot.push(RuleFeatures {
                id: *id,
                psafety: emp.psych_safety,
                supervisor: team.supervisor_openness,
                salience: ctx.world.issue_salience,
                fear: emp.fear,
                ivt: emp.ivt_strength,
                rho: emp.perceived_silence,
                rho_voice: ctx.world.neighbour_voice_ratio(*id),
                voice_threshold: emp.voice_threshold,
                private_concern: emp.private_concern,
                ivt_weights: emp.ivt_rule_weights,
            });
        }

        let beta_ivt = if self.ivt_active {
            self.beta.beta_ivt
        } else {
            0.0
        };

        let mut updates: Vec<(AgentId, Expression, Option<Motive>, Vec<u8>)> =
            Vec::with_capacity(ids.len());
        for f in snapshot {
            let cascade = if f.rho_voice > f.voice_threshold {
                self.beta.beta_threshold
            } else {
                0.0
            };
            let voice_logit = self.beta.intercept
                + self.beta.beta_psafety * f.psafety
                + self.beta.beta_supervisor * f.supervisor
                + self.beta.beta_salience * f.salience
                - self.beta.beta_fear * f.fear
                - beta_ivt * f.ivt
                - self.beta.beta_climate * f.rho
                + cascade;
            let p_voice = sigmoid(voice_logit);
            let u_voice: f64 = ctx.rng.gen();
            // Draw the 5 per-rule uniforms (rule-vote stream) up front so the
            // RNG sequence is independent of the VOICE/SILENCE branch taken.
            let u_rules: [f64; 5] = [
                ctx.rng.gen(),
                ctx.rng.gen(),
                ctx.rng.gen(),
                ctx.rng.gen(),
                ctx.rng.gen(),
            ];
            if u_voice < p_voice {
                updates.push((f.id, Expression::Voice, None, Vec::new()));
            } else {
                // Silence motive: defensive when fearful, prosocial when
                // supervisor is open, acquiescent otherwise.
                let motive = if f.fear > 0.5 {
                    Motive::Defensive
                } else if f.supervisor > 0.2 {
                    Motive::Prosocial
                } else {
                    Motive::Acquiescent
                };
                // In the IVT-ablated mode no rules fire (the IVT layer is off).
                let rules = if self.ivt_active {
                    sample_rule_fire_set(f.ivt, &f.ivt_weights, &u_rules)
                } else {
                    Vec::new()
                };
                // private_concern is consumed by the upward-silence metric.
                let _ = f.private_concern;
                updates.push((f.id, Expression::Silence, Some(motive), rules));
            }
        }
        for (id, expr, m, rules) in updates {
            let emp = ctx.world.employees.get_mut(&id).expect("agent missing");
            emp.expression = expr;
            emp.motive = m;
            emp.last_active_rules = rules;
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 4b. VoiceDecisionLlm  (Decision) — LLM-driven 5-rule self-reflection
// --------------------------------------------------------------------------- //

/// LLM-driven voice decision with the IVT 5-rule self-reflection layer.
pub struct VoiceDecisionLlm {
    client: SharedClient,
    metadata: SharedMetadata,
    settings: LlmSettings,
    /// `derive_seed` root for the (agent_id, t) LLM seed stream.
    llm_seed_root: u64,
}

impl VoiceDecisionLlm {
    pub fn new(
        client: SharedClient,
        metadata: SharedMetadata,
        settings: LlmSettings,
        llm_seed_root: u64,
    ) -> Self {
        VoiceDecisionLlm {
            client,
            metadata,
            settings,
            llm_seed_root,
        }
    }
}

impl Mechanism<SilenceWorld> for VoiceDecisionLlm {
    fn name(&self) -> &str {
        "voice_decision"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        let t = ctx.clock.t();
        // Snapshot prompts before mutating (synchronous update).
        let mut prompts: Vec<(AgentId, String, u64)> = Vec::with_capacity(ids.len());
        for id in ids {
            let prompt = build_voice_prompt(ctx.world, id);
            // Per-(agent, t) LLM seed — design §4.3 RNG streams.
            let llm_seed = derive_seed(self.llm_seed_root, &[3, id.0, t]);
            prompts.push((id, prompt, llm_seed));
        }

        let mut updates: Vec<(AgentId, Expression, Option<Motive>, Vec<u8>)> =
            Vec::with_capacity(prompts.len());
        for (id, prompt, llm_seed) in prompts {
            let mut cfg = llm_config(&self.settings);
            cfg.seed = llm_seed;
            let text = {
                let mut client = self.client.borrow_mut();
                let resp = client.complete(&prompt, &cfg).map_err(|e| {
                    SocsimError::Mechanism(format!("voice_decision LLM call failed: {e}"))
                })?;
                self.metadata.borrow_mut().record(resp.metadata.clone());
                resp.text
            };
            let verdict = parse_voice_decision(&text);
            updates.push((id, verdict.expression, verdict.motive, verdict.active_rules));
        }
        for (id, expr, m, rules) in updates {
            let emp = ctx.world.employees.get_mut(&id).expect("agent missing");
            emp.expression = expr;
            emp.motive = m;
            emp.last_active_rules = rules;
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 5. SilenceSpiral  (Interaction)
// --------------------------------------------------------------------------- //

/// `ρ_i ← neighbour silence ratio` (Noelle-Neumann 1974).
pub struct SilenceSpiral;

impl Mechanism<SilenceWorld> for SilenceSpiral {
    fn name(&self) -> &str {
        "silence_spiral"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Interaction]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        let mut new_rho: Vec<(AgentId, f64)> = Vec::with_capacity(ids.len());
        for id in ids {
            new_rho.push((id, ctx.world.neighbour_silence_ratio(id)));
        }
        for (id, r) in new_rho {
            if let Some(e) = ctx.world.employees.get_mut(&id) {
                e.perceived_silence = r;
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 6. PrefalseCascade  (Interaction)
// --------------------------------------------------------------------------- //

/// Threshold cascade (Kuran 1995): a silent agent flips to VOICE if their
/// neighbour-VOICE ratio exceeds `θ_i`.
pub struct PrefalseCascade;

impl Mechanism<SilenceWorld> for PrefalseCascade {
    fn name(&self) -> &str {
        "prefalse_cascade"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Interaction]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        let mut flips: Vec<AgentId> = Vec::new();
        for id in ids {
            let emp = &ctx.world.employees[&id];
            // Preference-falsification cascade (Kuran 1995): only agents who are
            // *falsely* silent — i.e. they do not privately disagree
            // (`private_concern >= 0`) — flip to VOICE once enough neighbours
            // are visibly voicing. Genuinely self-censoring agents
            // (`private_concern < 0`, the upward-silence population) are driven
            // by the IVT structure, not by a peer cascade, so they are exempt.
            if emp.expression == Expression::Silence
                && emp.private_concern >= 0.0
                && ctx.world.neighbour_voice_ratio(id) > emp.voice_threshold
            {
                flips.push(id);
            }
        }
        for id in flips {
            if let Some(e) = ctx.world.employees.get_mut(&id) {
                e.expression = Expression::Voice;
                e.motive = None;
                e.last_active_rules.clear();
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 7. OrgPerformance  (Reward)
// --------------------------------------------------------------------------- //

/// Per-team knowledge-stock update: `K_k(t+1) = (1-δ)K_k(t) + voice_share_k`.
pub struct OrgPerformance {
    delta: f64,
}

impl OrgPerformance {
    pub fn new() -> Self {
        OrgPerformance { delta: 0.10 }
    }
}

impl Default for OrgPerformance {
    fn default() -> Self {
        Self::new()
    }
}

impl Mechanism<SilenceWorld> for OrgPerformance {
    fn name(&self) -> &str {
        "org_performance"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Reward]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let n_teams = ctx.world.teams.len();
        let mut sizes = vec![0u64; n_teams];
        let mut voice_cnt = vec![0u64; n_teams];
        for e in ctx.world.employees.values() {
            sizes[e.team] += 1;
            if e.expression == Expression::Voice {
                voice_cnt[e.team] += 1;
            }
        }
        for k in 0..n_teams {
            let n = sizes[k].max(1) as f64;
            let v = voice_cnt[k] as f64 / n;
            let team = &mut ctx.world.teams[k];
            team.knowledge_stock = ((1.0 - self.delta) * team.knowledge_stock + v).max(0.0);
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 8. PsafetyUpdate  (PostStep)
// --------------------------------------------------------------------------- //

/// `ψ_i ← clamp(ψ_i + η(voiced) − ν(retaliated), 0, 1)`.
pub struct PsafetyUpdate {
    eta: f64,
    nu: f64,
}

impl PsafetyUpdate {
    pub fn new() -> Self {
        PsafetyUpdate {
            eta: 0.05,
            nu: 0.15,
        }
    }
}

impl Default for PsafetyUpdate {
    fn default() -> Self {
        Self::new()
    }
}

impl Mechanism<SilenceWorld> for PsafetyUpdate {
    fn name(&self) -> &str {
        "psafety_update"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::PostStep]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let retaliated: std::collections::HashSet<AgentId> =
            ctx.world.retaliation_this_step.iter().copied().collect();
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        for id in ids {
            let emp = ctx.world.employees.get_mut(&id).expect("agent missing");
            let voiced = matches!(emp.expression, Expression::Voice);
            let was_retaliated = retaliated.contains(&id);
            let delta =
                self.eta * (voiced as i32 as f64) - self.nu * (was_retaliated as i32 as f64);
            emp.psych_safety = (emp.psych_safety + delta).clamp(0.0, 1.0);
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 9. ClimateSilence  (PostStep)
// --------------------------------------------------------------------------- //

/// Updates `world.climate_of_silence` + per-team climate and tracks the
/// convergence delta `|C(t) − C(t-1)|`.
pub struct ClimateSilence;

impl Mechanism<SilenceWorld> for ClimateSilence {
    fn name(&self) -> &str {
        "climate_silence"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::PostStep]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let prev = ctx.world.climate_of_silence;
        let now = crate::metrics::climate_of_silence(ctx.world);
        ctx.world.last_max_delta = (now - prev).abs();
        ctx.world.climate_of_silence = now;
        let per_team = crate::metrics::team_climates(ctx.world);
        for (k, c) in per_team.into_iter().enumerate() {
            if let Some(team) = ctx.world.teams.get_mut(k) {
                team.climate = c;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_at_zero_is_half() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn rule_fire_prob_ranks_by_weight() {
        // need_data (w=0.22) must have a higher firing probability than
        // no_bypass (w=0.18) at the same ι.
        let p_need = rule_fire_prob(0.55, 0.22);
        let p_bypass = rule_fire_prob(0.55, 0.18);
        assert!(p_need > p_bypass);
    }

    #[test]
    fn rule_fire_prob_zero_when_ivt_zero() {
        // With ι=0 every rule's salience is 0, far below MID → ~0 probability.
        assert!(rule_fire_prob(0.0, 0.22) < 0.05);
    }

    #[test]
    fn sample_rule_fire_set_fires_high_prob_rule() {
        let w = [0.20, 0.22, 0.18, 0.20, 0.20];
        // u = 0 for every rule → all rules with p > 0 fire.
        let fired = sample_rule_fire_set(0.55, &w, &[0.0; 5]);
        assert!(fired.contains(&IvtRule::NeedData.id()));
        // u = 1 for every rule → none fire (p < 1 always).
        assert!(sample_rule_fire_set(0.55, &w, &[1.0; 5]).is_empty());
    }
}
