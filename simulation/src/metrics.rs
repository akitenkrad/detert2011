//! Aggregate metrics for the Detert & Edmondson (2011) IVT silence model.
//!
//! - **upward_silence_rate** — `(1/N) Σ 1[b̂_i = Silence ∧ b_i < 0]` (the HiCo
//!   50% anchor): employees who hold an upward concern yet stay silent.
//! - **ivt_rule_activation_mix** — per-rule firing rate `a_r` over employees
//!   (5-vec, rules `target_id … career_consq`).
//! - **rule_cooccurrence** — `5×5` matrix of joint firing rates; discriminant
//!   validity expects off-diagonal entries `< 0.50`.
//! - **silence_voice_corr** — per-agent Pearson r between a silence indicator
//!   and a voice indicator (paper Study 4 `r = −.55`).
//! - **climate_of_silence** — `C(t) = (1/N) Σ 1[Silence ∧ b_i < 0]`.
//! - **ever_silent_fraction** — fraction of employees who were ever silent.

use crate::world::{Expression, SilenceWorld};

/// Fraction of employees in `Silence` expression.
pub fn silence_rate(world: &SilenceWorld) -> f64 {
    let n = world.n_employees();
    if n == 0 {
        return 0.0;
    }
    let silent = world
        .employees
        .values()
        .filter(|e| e.expression == Expression::Silence)
        .count();
    silent as f64 / n as f64
}

/// `upward_silence_rate = (1/N) Σ 1[Silence ∧ private_concern < 0]`
/// (the HiCo 50% anchor — Study 1).
pub fn upward_silence_rate(world: &SilenceWorld) -> f64 {
    let n = world.n_employees();
    if n == 0 {
        return 0.0;
    }
    let cnt = world
        .employees
        .values()
        .filter(|e| e.expression == Expression::Silence && e.private_concern < 0.0)
        .count();
    cnt as f64 / n as f64
}

/// Climate of silence `C(t)` — identical definition to `upward_silence_rate`
/// here, retained as a distinct PostStep aggregate the next tick observes.
pub fn climate_of_silence(world: &SilenceWorld) -> f64 {
    upward_silence_rate(world)
}

/// Per-team climate-of-silence values, one entry per team.
pub fn team_climates(world: &SilenceWorld) -> Vec<f64> {
    let n_teams = world.teams.len();
    let mut counts = vec![0u64; n_teams];
    let mut sizes = vec![0u64; n_teams];
    for e in world.employees.values() {
        sizes[e.team] += 1;
        if e.expression == Expression::Silence && e.private_concern < 0.0 {
            counts[e.team] += 1;
        }
    }
    let mut out = vec![0.0; n_teams];
    for k in 0..n_teams {
        out[k] = if sizes[k] == 0 {
            0.0
        } else {
            counts[k] as f64 / sizes[k] as f64
        };
    }
    out
}

/// Per-rule activation rate `a_r` over all employees' `last_active_rules`.
pub fn ivt_rule_activation_mix(world: &SilenceWorld) -> [f64; 5] {
    let n = world.n_employees();
    if n == 0 {
        return [0.0; 5];
    }
    let mut counts = [0u64; 5];
    for e in world.employees.values() {
        for &r in &e.last_active_rules {
            if (r as usize) < 5 {
                counts[r as usize] += 1;
            }
        }
    }
    let mut out = [0.0; 5];
    for (o, &c) in out.iter_mut().zip(counts.iter()) {
        *o = c as f64 / n as f64;
    }
    out
}

/// `5×5` joint firing-rate matrix `Pr(r, r' fire together)` over employees.
/// The diagonal is each rule's own activation rate.
pub fn rule_cooccurrence(world: &SilenceWorld) -> [[f64; 5]; 5] {
    let n = world.n_employees();
    let mut mat = [[0.0f64; 5]; 5];
    if n == 0 {
        return mat;
    }
    let mut counts = [[0u64; 5]; 5];
    for e in world.employees.values() {
        for &r in &e.last_active_rules {
            for &s in &e.last_active_rules {
                if (r as usize) < 5 && (s as usize) < 5 {
                    counts[r as usize][s as usize] += 1;
                }
            }
        }
    }
    for (r, row) in mat.iter_mut().enumerate() {
        for (s, cell) in row.iter_mut().enumerate() {
            *cell = counts[r][s] as f64 / n as f64;
        }
    }
    mat
}

/// Maximum off-diagonal co-occurrence (the discriminant-validity statistic;
/// expected `< 0.50`).
pub fn max_offdiag_cooccurrence(mat: &[[f64; 5]; 5]) -> f64 {
    let mut m = 0.0f64;
    for (r, row) in mat.iter().enumerate() {
        for (s, &cell) in row.iter().enumerate() {
            if r != s && cell > m {
                m = cell;
            }
        }
    }
    m
}

/// Per-agent Pearson r between a SILENCE indicator and a VOICE indicator.
/// With a hard VOICE/SILENCE dichotomy this is `−1` whenever the population is
/// mixed; computed generally so it degrades gracefully.
pub fn silence_voice_corr(world: &SilenceWorld) -> f64 {
    let mut silence: Vec<f64> = Vec::with_capacity(world.n_employees());
    let mut voice: Vec<f64> = Vec::with_capacity(world.n_employees());
    for e in world.employees.values() {
        match e.expression {
            Expression::Silence => {
                silence.push(1.0);
                voice.push(0.0);
            }
            Expression::Voice => {
                silence.push(0.0);
                voice.push(1.0);
            }
            Expression::Neutral => {
                silence.push(0.0);
                voice.push(0.0);
            }
        }
    }
    pearson(&silence, &voice)
}

/// Fraction of employees flagged as ever-silent (set on the agent rows by the
/// run driver). Computed here from a supplied bitset to keep the world clean.
pub fn ever_silent_fraction(ever: &[bool]) -> f64 {
    if ever.is_empty() {
        return 0.0;
    }
    ever.iter().filter(|&&b| b).count() as f64 / ever.len() as f64
}

/// Pearson correlation between paired `x`/`y`. Returns 0 on degenerate input.
pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let denom = (sxx * syy).sqrt();
    if denom <= 0.0 {
        0.0
    } else {
        sxy / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{Employee, Expression, SilenceWorld, Team};
    use socsim_core::{AgentId, SimClock, SimRng};
    use socsim_net::SocialNetwork;
    use std::collections::BTreeMap;

    fn mini_world(exprs: &[Expression], rules: &[Vec<u8>]) -> SilenceWorld {
        assert_eq!(exprs.len(), rules.len());
        let mut rng = SimRng::from_seed(0);
        let ids: Vec<AgentId> = (0..exprs.len()).map(|i| AgentId(i as u64)).collect();
        let net = SocialNetwork::erdos_renyi(&ids, 0.5, &mut rng);
        let mut emps: BTreeMap<AgentId, Employee> = BTreeMap::new();
        for (i, &id) in ids.iter().enumerate() {
            let mut e = Employee::neutral(0, 0, 0);
            e.expression = exprs[i];
            e.private_concern = -0.5;
            e.last_active_rules = rules[i].clone();
            emps.insert(id, e);
        }
        SilenceWorld::new(SimClock::new(1), emps, vec![Team::default()], net)
    }

    #[test]
    fn upward_silence_counts_disagreeing_silent() {
        let w = mini_world(
            &[Expression::Silence, Expression::Voice],
            &[vec![1], vec![]],
        );
        assert!((upward_silence_rate(&w) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn activation_mix_counts_rules() {
        let w = mini_world(
            &[Expression::Silence, Expression::Silence],
            &[vec![1, 4], vec![1]],
        );
        let mix = ivt_rule_activation_mix(&w);
        assert!((mix[1] - 1.0).abs() < 1e-12); // need_data in both
        assert!((mix[4] - 0.5).abs() < 1e-12); // career_consq in one
        assert_eq!(mix[0], 0.0);
    }

    #[test]
    fn cooccurrence_offdiag_below_diag() {
        let w = mini_world(
            &[Expression::Silence, Expression::Silence],
            &[vec![1, 4], vec![1]],
        );
        let mat = rule_cooccurrence(&w);
        // need_data fires in both → diag[1] = 1.0
        assert!((mat[1][1] - 1.0).abs() < 1e-12);
        // need_data & career_consq co-fire in one → 0.5
        assert!((mat[1][4] - 0.5).abs() < 1e-12);
        assert!(max_offdiag_cooccurrence(&mat) <= mat[1][1]);
    }

    #[test]
    fn silence_voice_corr_is_minus_one_when_mixed() {
        let w = mini_world(
            &[Expression::Silence, Expression::Voice, Expression::Silence],
            &[vec![], vec![], vec![]],
        );
        assert!((silence_voice_corr(&w) + 1.0).abs() < 1e-9);
    }

    #[test]
    fn pearson_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        assert!((pearson(&x, &y) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ever_silent_fraction_basic() {
        assert!((ever_silent_fraction(&[true, false, true, false]) - 0.5).abs() < 1e-12);
    }
}
