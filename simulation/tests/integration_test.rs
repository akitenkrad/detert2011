//! Integration tests for the Detert & Edmondson (2011) IVT silence simulation.
//!
//! **No live LLM required.** The rule / rule_no_ivt modes need no LLM at all;
//! the LLM path is driven by `socsim_llm::mock::ScriptedClient`. Tests cover:
//! rule-mode bit-determinism, the IVT-necessity contrast (rule_no_ivt lowers
//! upward silence), and the LLM path end-to-end via a scripted client.

use detert_silence::config::{BetaGroup, Config, LlmMode, LlmSettings, NetworkKind};
use detert_silence::llm::wrap_client;
use detert_silence::simulation::{run_with_client, SimulationResult};

use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

fn small_cfg(mode: LlmMode) -> Config {
    Config {
        n_teams: 2,
        team_size: 8,
        n_levels: 2,
        network_kind: NetworkKind::WattsStrogatz,
        network_k: 4,
        network_beta: 0.1,
        llm_mode: mode,
        ivt_mean: 0.55,
        ivt_sd: 0.20,
        beta: BetaGroup::default(),
        p_retaliate: 0.05,
        shock_t: None,
        shock_magnitude: 0.3,
        t_max: 10,
        runs: 1,
        seed: 1234,
        llm: LlmSettings::default(),
        output_dir: "results".to_string(),
        ..Config::default()
    }
}

/// Scripted client cycling VOICE / SILENCE-defensive(need_data,career) /
/// SILENCE-prosocial(no_embarrass).
fn scripted_client() -> detert_silence::llm::SilenceClient {
    let backend = ScriptedClient::new("mock-model", |prompt: &str| {
        let h = prompt.len() % 3;
        match h {
            0 => r#"{"decision":"voice","motive":null,"active_rules":[]}"#.to_string(),
            1 => r#"{"decision":"silence","motive":"defensive","active_rules":["need_data","career_consq"]}"#
                .to_string(),
            _ => r#"{"decision":"silence","motive":"prosocial","active_rules":["no_embarrass"]}"#
                .to_string(),
        }
    });
    wrap_client(backend, PromptCache::in_memory())
}

#[test]
fn rule_mode_smoke_run() {
    let r: SimulationResult = run_with_client(&small_cfg(LlmMode::Rule), None).unwrap();
    assert!(!r.metrics_rows.is_empty(), "must produce per-step metrics");
    assert_eq!(r.metadata.total(), 0, "rule mode makes 0 LLM calls");
    for row in &r.metrics_rows {
        assert!((0.0..=1.0).contains(&row.upward_silence_rate));
        assert!((0.0..=1.0).contains(&row.climate_of_silence));
        assert!((-1.0..=1.0).contains(&row.silence_voice_corr));
    }
    // rule mode should produce non-empty rule activation (the IVT layer is on).
    let any_rule = r.metrics_rows.iter().any(|row| {
        row.rule_target_id
            + row.rule_need_data
            + row.rule_no_bypass
            + row.rule_no_embarrass
            + row.rule_career_consq
            > 0.0
    });
    assert!(any_rule, "rule mode should fire some IVT rules");
    // 5 rules × t rows.
    assert_eq!(r.rule_activation_rows.len(), 5 * r.metrics_rows.len());
}

#[test]
fn rule_mode_is_bit_deterministic() {
    let a = run_with_client(&small_cfg(LlmMode::Rule), None).unwrap();
    let b = run_with_client(&small_cfg(LlmMode::Rule), None).unwrap();
    assert_eq!(a.metrics_rows.len(), b.metrics_rows.len());
    for (ra, rb) in a.metrics_rows.iter().zip(b.metrics_rows.iter()) {
        assert_eq!(ra.t, rb.t);
        assert!((ra.upward_silence_rate - rb.upward_silence_rate).abs() < 1e-15);
        assert!((ra.silence_voice_corr - rb.silence_voice_corr).abs() < 1e-15);
        assert!((ra.max_rule_cooccurrence - rb.max_rule_cooccurrence).abs() < 1e-15);
        assert!((ra.rule_need_data - rb.rule_need_data).abs() < 1e-15);
    }
}

#[test]
fn rule_no_ivt_fires_no_rules() {
    let r = run_with_client(&small_cfg(LlmMode::RuleNoIvt), None).unwrap();
    let any_rule = r.metrics_rows.iter().any(|row| {
        row.rule_target_id
            + row.rule_need_data
            + row.rule_no_bypass
            + row.rule_no_embarrass
            + row.rule_career_consq
            > 0.0
    });
    assert!(!any_rule, "rule_no_ivt ablates the IVT rule layer");
}

#[test]
fn rule_no_ivt_lowers_upward_silence_on_average() {
    let mut sum_rule = 0.0;
    let mut sum_noivt = 0.0;
    let seeds = [10u64, 11, 12, 13, 14, 15];
    for &s in &seeds {
        let mut cr = small_cfg(LlmMode::Rule);
        cr.seed = s;
        cr.t_max = 40;
        sum_rule += run_with_client(&cr, None).unwrap().final_upward_silence();
        let mut cn = small_cfg(LlmMode::RuleNoIvt);
        cn.seed = s;
        cn.t_max = 40;
        sum_noivt += run_with_client(&cn, None).unwrap().final_upward_silence();
    }
    let mr = sum_rule / seeds.len() as f64;
    let mn = sum_noivt / seeds.len() as f64;
    assert!(
        mn <= mr + 1e-9,
        "rule_no_ivt upward_silence ({mn}) should be ≤ rule ({mr}) — IVT necessity"
    );
}

#[test]
fn llm_mode_smoke_run_with_scripted_client() {
    let cfg = small_cfg(LlmMode::Llm);
    let client = scripted_client();
    let r = run_with_client(&cfg, Some(client)).unwrap();
    assert!(!r.metrics_rows.is_empty());
    assert!(r.metadata.total() > 0, "LLM mode must call the LLM");
    // The scripted client emits active_rules on silence, so some rule firing.
    let any_rule = r
        .metrics_rows
        .iter()
        .any(|row| row.rule_need_data + row.rule_career_consq + row.rule_no_embarrass > 0.0);
    assert!(any_rule, "scripted silence responses carry active_rules");
}
