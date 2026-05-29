//! LLM prompt construction and response parsing for `voice_decision`.
//!
//! The IVT 5-rule self-reflection layer is embedded directly in the prompt: the
//! employee is asked to reflect on each of the five Implicit Voice Theory rules
//! (weighted by `ι_i · w_{i,r}`) before deciding, and to report which rules
//! actually drove their choice. The LLM returns a JSON object of the form
//!
//! ```json
//! {
//!   "decision": "voice" | "silence",
//!   "motive":   "acquiescent" | "defensive" | "prosocial" | null,
//!   "active_rules": ["need_data", "career_consq"]
//! }
//! ```
//!
//! When `decision = silence`, `motive` is expected; when `decision = voice`,
//! `motive` is `null` and `active_rules` is typically empty. Parse failures
//! fall back to `Silence + None + []` (`parse_failed = true`).

use serde::Deserialize;
use serde_json::Value;

use crate::world::{Employee, Expression, IvtRule, Motive, SilenceWorld};

// --------------------------------------------------------------------------- //
// IVT rule descriptions (the self-reflection layer)
// --------------------------------------------------------------------------- //

/// Short, second-person description of each IVT rule, in canonical id order.
pub const RULE_DESCRIPTIONS: [&str; 5] = [
    // 0 target_id
    "your boss will take any upward input as a personal attack and identify you as the critic",
    // 1 need_data
    "you must have solid data or a finished solution before it is acceptable to speak up",
    // 2 no_bypass
    "you must never go over your direct supervisor's head to a higher level",
    // 3 no_embarrass
    "you must never say anything that could embarrass your boss in front of others",
    // 4 career_consq
    "speaking up will damage your promotion, evaluation, or standing",
];

/// Personas keyed by hierarchical level band (low / mid / senior). Index by
/// `min(level, 2)`.
pub const PERSONAS: [&str; 3] = [
    "a junior front-line employee with little positional power",
    "a mid-level employee with some tenure and a direct supervisor above you",
    "a senior employee close to leadership but still reporting upward",
];

/// Persona string for an employee level.
pub fn persona_for_level(level: u8) -> &'static str {
    PERSONAS[(level as usize).min(2)]
}

// --------------------------------------------------------------------------- //
// Prompt construction
// --------------------------------------------------------------------------- //

/// Build the voice-decision prompt for `agent_id` from the world. The five IVT
/// rules are listed with a per-rule salience `ι_i · w_{i,r}` so the LLM weights
/// its self-reflection by the employee's persistent IVT structure.
pub fn build_voice_prompt(world: &SilenceWorld, agent_id: socsim_core::AgentId) -> String {
    let emp = &world.employees[&agent_id];
    let team = &world.teams[emp.team];
    let rho = world.neighbour_silence_ratio(agent_id);
    let sigma = world.issue_salience;

    let context = format_context(emp, team.supervisor_openness, sigma, rho);
    let rules = format_rules(emp);

    format!(
        "You are {persona}.\n\n\
         An ethically or operationally questionable practice has surfaced at work. \
         You must decide whether to SPEAK UP to your supervisor (voice) or REMAIN SILENT.\n\n\
         Your inner state:\n\
         {context}\n\
         Before you decide, reflect on these taken-for-granted rules about speaking up at \
         work. Each is annotated with how strongly it weighs on you (0=not at all, 1=strongly):\n\
         {rules}\n\
         Reply with a SINGLE JSON object on one line:\n\
         {{\"decision\": \"voice\" | \"silence\", \
            \"motive\": \"acquiescent\" | \"defensive\" | \"prosocial\" | null, \
            \"active_rules\": [\"target_id\" | \"need_data\" | \"no_bypass\" | \"no_embarrass\" | \"career_consq\", ...]}}\n\
         Rules: if decision = voice, motive must be null and active_rules should be empty. \
         If decision = silence, motive must be one of the three labels and active_rules must \
         list the rules above that actually drove your silence. Output JSON only.",
        persona = persona_for_level(emp.level),
    )
}

fn format_context(emp: &Employee, supervisor_openness: f64, sigma: f64, rho: f64) -> String {
    format!(
        "  fear of consequences      f = {f:.2}\n\
         \x20 psychological safety      ψ = {psi:.2}\n\
         \x20 implicit-voice strength   ι = {iota:.2}\n\
         \x20 supervisor openness       u = {u:+.2}\n\
         \x20 issue salience            σ = {sigma:.2}\n\
         \x20 perceived peer silence    ρ = {rho:.2}\n\
         \x20 hierarchical level        ℓ = {level}\n",
        f = emp.fear,
        psi = emp.psych_safety,
        iota = emp.ivt_strength,
        u = supervisor_openness,
        sigma = sigma,
        rho = rho,
        level = emp.level,
    )
}

fn format_rules(emp: &Employee) -> String {
    let mut lines = String::new();
    for r in IvtRule::ALL {
        let salience = emp.ivt_strength * emp.ivt_rule_weights[r.id() as usize];
        lines.push_str(&format!(
            "  - [{label}] {desc} (weight {sal:.2})\n",
            label = r.label(),
            desc = RULE_DESCRIPTIONS[r.id() as usize],
            sal = salience,
        ));
    }
    lines
}

// --------------------------------------------------------------------------- //
// Response parsing
// --------------------------------------------------------------------------- //

/// Parsed voice-decision verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceDecisionVerdict {
    pub expression: Expression,
    pub motive: Option<Motive>,
    /// IVT rule ids `0..5` the LLM reported as active.
    pub active_rules: Vec<u8>,
    /// True if parsing failed and we fell back to `Silence + None + []`.
    pub parse_failed: bool,
}

#[derive(Deserialize)]
struct RawDecision {
    decision: Option<String>,
    motive: Option<String>,
    #[serde(default)]
    active_rules: Vec<String>,
}

/// Parse an LLM response into a verdict.
///
/// Lenient: extracts the first balanced `{...}` object, accepts mixed-case
/// labels, and on any failure falls back to `Silence + None + []`.
pub fn parse_voice_decision(text: &str) -> VoiceDecisionVerdict {
    let fallback = VoiceDecisionVerdict {
        expression: Expression::Silence,
        motive: None,
        active_rules: Vec::new(),
        parse_failed: true,
    };

    let json_str = match extract_json_object(text) {
        Some(s) => s,
        None => return fallback,
    };

    if let Ok(raw) = serde_json::from_str::<RawDecision>(&json_str) {
        return finalise_verdict(raw);
    }
    if let Ok(val) = serde_json::from_str::<Value>(&json_str) {
        let active_rules = val
            .get("active_rules")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let raw = RawDecision {
            decision: val
                .get("decision")
                .and_then(|v| v.as_str().map(str::to_string)),
            motive: val.get("motive").and_then(|v| {
                if v.is_null() {
                    None
                } else {
                    v.as_str().map(str::to_string)
                }
            }),
            active_rules,
        };
        return finalise_verdict(raw);
    }
    fallback
}

fn parse_active_rules(labels: &[String]) -> Vec<u8> {
    let mut ids: Vec<u8> = labels
        .iter()
        .filter_map(|s| match s.trim().to_ascii_lowercase().as_str() {
            "target_id" | "target-id" | "targetid" => Some(IvtRule::TargetId.id()),
            "need_data" | "need-data" | "needdata" => Some(IvtRule::NeedData.id()),
            "no_bypass" | "no-bypass" | "nobypass" => Some(IvtRule::NoBypass.id()),
            "no_embarrass" | "no-embarrass" | "noembarrass" => Some(IvtRule::NoEmbarrass.id()),
            "career_consq" | "career-consq" | "career_consequences" => {
                Some(IvtRule::CareerConsq.id())
            }
            _ => None,
        })
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

fn finalise_verdict(raw: RawDecision) -> VoiceDecisionVerdict {
    let decision = raw
        .decision
        .as_deref()
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_default();
    let expression = match decision.as_str() {
        "voice" | "speak" | "speak_up" => Expression::Voice,
        "silence" | "silent" | "withhold" => Expression::Silence,
        _ => {
            return VoiceDecisionVerdict {
                expression: Expression::Silence,
                motive: None,
                active_rules: Vec::new(),
                parse_failed: true,
            };
        }
    };

    if expression == Expression::Voice {
        return VoiceDecisionVerdict {
            expression,
            motive: None,
            active_rules: Vec::new(),
            parse_failed: false,
        };
    }

    // Silence: motive is expected; active_rules are parsed leniently.
    let motive = raw.motive.as_deref().and_then(Motive::parse);
    let active_rules = parse_active_rules(&raw.active_rules);
    let parse_failed = motive.is_none();
    VoiceDecisionVerdict {
        expression,
        motive,
        active_rules,
        parse_failed,
    }
}

/// Extract the first balanced `{...}` substring from `text`.
fn extract_json_object(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{')?;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start..=i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_canonical_voice() {
        let v =
            parse_voice_decision(r#"{"decision": "voice", "motive": null, "active_rules": []}"#);
        assert_eq!(v.expression, Expression::Voice);
        assert_eq!(v.motive, None);
        assert!(v.active_rules.is_empty());
        assert!(!v.parse_failed);
    }

    #[test]
    fn parses_silence_with_rules() {
        let v = parse_voice_decision(
            r#"{"decision":"silence","motive":"defensive","active_rules":["need_data","career_consq"]}"#,
        );
        assert_eq!(v.expression, Expression::Silence);
        assert_eq!(v.motive, Some(Motive::Defensive));
        assert_eq!(v.active_rules, vec![1u8, 4u8]);
        assert!(!v.parse_failed);
    }

    #[test]
    fn tolerates_surrounding_text() {
        let v = parse_voice_decision(
            r#"Here you go: {"decision":"silence","motive":"prosocial","active_rules":["no_embarrass"]}. Done."#,
        );
        assert_eq!(v.expression, Expression::Silence);
        assert_eq!(v.motive, Some(Motive::Prosocial));
        assert_eq!(v.active_rules, vec![3u8]);
    }

    #[test]
    fn dedups_and_sorts_rules() {
        let v = parse_voice_decision(
            r#"{"decision":"silence","motive":"acquiescent","active_rules":["career_consq","need_data","need_data"]}"#,
        );
        assert_eq!(v.active_rules, vec![1u8, 4u8]);
    }

    #[test]
    fn unknown_decision_falls_back() {
        let v = parse_voice_decision(r#"{"decision":"???","motive":"as","active_rules":[]}"#);
        assert!(v.parse_failed);
        assert_eq!(v.expression, Expression::Silence);
    }

    #[test]
    fn silence_without_motive_flags_parse_failed() {
        let v = parse_voice_decision(r#"{"decision":"silence","active_rules":["need_data"]}"#);
        assert!(v.parse_failed);
        assert_eq!(v.active_rules, vec![1u8]);
    }

    #[test]
    fn no_json_falls_back() {
        let v = parse_voice_decision("no json here");
        assert!(v.parse_failed);
        assert_eq!(v.expression, Expression::Silence);
    }

    #[test]
    fn personas_by_level() {
        assert_eq!(persona_for_level(0), PERSONAS[0]);
        assert_eq!(persona_for_level(5), PERSONAS[2]);
    }
}
