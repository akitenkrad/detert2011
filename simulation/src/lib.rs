//! Detert & Edmondson (2011) — Implicit Voice Theories silence simulation.
//!
//! A socsim-based ABM in which employees on a Watts–Strogatz organisational
//! network decide each step whether to VOICE an upward concern or stay SILENT.
//! The paper's **five Implicit Voice Theory (IVT) rules** —
//!
//! 1. presumed target identification (`target_id`)
//! 2. need solid data / solutions (`need_data`)
//! 3. don't bypass the boss upward (`no_bypass`)
//! 4. don't embarrass the boss in public (`no_embarrass`)
//! 5. negative career consequences (`career_consq`)
//!
//! — are embedded as a **self-reflection layer** inside the LLM `voice_decision`
//! prompt (the LLM returns `{decision, motive, active_rules}` as JSON), and as
//! an additive `−β_ι·ι_i` term inside the rule-mode logit.
//!
//! Three **mutually exclusive** decision modes are wired by `config.llm_mode`:
//!
//! - `--llm-mode llm`         — `VoiceDecisionLlm`: LLM decides VOICE/SILENCE,
//!   the silence motive, and which IVT rules fired (`socsim-llm` harness).
//! - `--llm-mode rule`        — `VoiceDecisionRule`: the §4.3 logit formula with
//!   the full `−β_ι·ι_i` IVT term. Zero LLM calls, bit-deterministic.
//! - `--llm-mode rule_no_ivt` — same logit with `β_ι` forced to 0 (the IVT
//!   ablation). Zero LLM calls, bit-deterministic.
//!
//! Eight further (non-decision) mechanisms run unconditionally each step —
//! `IssueSalience`, `RetaliationEvent`, `FearAppraisal`, `SilenceSpiral`,
//! `PrefalseCascade`, `OrgPerformance`, `PsafetyUpdate`, `ClimateSilence` —
//! across socsim's 6-phase fixed loop (see `mechanisms.rs`).
//!
//! See `simulation/src/main.rs` for the `run` / `sweep` / `ablation` /
//! `reproduce` CLI.

pub mod config;
pub mod llm;
pub mod mechanisms;
pub mod metrics;
pub mod prompts;
pub mod simulation;
pub mod world;
