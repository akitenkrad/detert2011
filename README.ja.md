<p align="center"><img src="docs/assets/hero.svg" width="100%"></p>

[English](README.md) | **日本語**

# Detert & Edmondson (2011) — Implicit Voice Theories

**Detert & Edmondson (2011)「Implicit Voice Theories: Taken-for-Granted Rules of Self-Censorship at Work」**（*Academy of Management Journal*, 54(3), 461–488; DOI: 10.5465/AMJ.2011.61967925）の生成的エージェントベース再現実装．

従業員は Watts–Strogatz 組織ネットワーク上で，各ステップに上位への懸念を **VOICE**（発言）するか **SILENCE**（沈黙）を選ぶ．意思決定の前に，論文が同定した **5 つの暗黙の発言理論（IVT）ルール** を内省する：

1. **target_id** — 上司は提案を個人攻撃と受け取り発言者を特定する，
2. **need_data** — 確実なデータ・完成した解決策が無ければ発言してはならない，
3. **no_bypass** — 直属の上司を飛び越えてはならない，
4. **no_embarrass** — 公の場で上司に恥をかかせてはならない，
5. **career_consq** — 発言は昇進・評価に悪影響を及ぼす．

3 つの **排他的な** 意思決定モードを `--llm-mode` で選択する：

- `llm` — LLM（`socsim-llm`，Ollama 第一 → OpenAI フォールバック）が persona と局所文脈から VOICE/SILENCE・沈黙動機・発火したIVTルールを返す．5 ルールはプロンプト内の **内省レイヤ** として埋め込まれる．
- `rule` — §4.3 の VOICE ロジット `σ(β0 + β_ψ ψ + β_u u + β_σ σ − β_f f − β_ι ι − β_C ρ + β_θ·cascade)`．IVT 主効果項 `−β_ι ι` を含む．
- `rule_no_ivt` — 同ロジットで `β_ι = 0`（IVT 項の因果寄与を分離する ablation）．

`rule` / `rule_no_ivt` は **LLM 呼び出しゼロ** で bit 単位に再現する．

## 二層決定論

LLM 出力は socsim の bit 再現性の **外側** にあるため，設計を 2 層に分ける：

- **決定論的 socsim コア** — 従業員初期化，Watts–Strogatz ネットワーク生成，スケジューリング，8 つの非意思決定機構，両 rule ロジット．seed を固定すれば bit 単位で再現する．
- **非決定的 LLM レイヤ** — `voice_decision` 機構のみ．`socsim-llm` の `CachingClient`（`hash(prompt+model)` → 応答キャッシュ）・`temperature=0`・`(agent_id, t)` 由来の固定 seed で擬似決定論化する．モデルではなくキャッシュが再現性の機構であり，warm cache は同一応答を再生する．

各実行は `llm_meta.json` に mode / model / endpoint / temperature / seed / cache-hit 率を記録する．

## インストール & クイックスタート

```bash
# Rust シミュレーションをビルド（socsim と socsim-llm の Ollama+OpenAI バックエンドを取得）
cargo build --release

# === HiCo 50% 較正（rule モード・LLM 不要） ===
cargo run --release -- run \
    --n 200 --n-teams 8 --team-size 25 --n-levels 3 \
    --network-model watts-strogatz --network-k 6 --network-beta 0.1 \
    --ivt-mean 0.55 --ivt-sd 0.20 \
    --beta-psafety 1.2 --beta-fear 1.5 --beta-ivt 2.0 \
    --llm-mode rule --t-max 60 --runs 1 --seed 42

# === LLM モード（Ollama 第一） ===
#   ollama pull llama3.1
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.1
cargo run --release -- run --llm-mode llm \
    --cache-path runs/detert_cache.json \
    --t-max 60 --runs 1 --seed 42

# === 感度スイープ（β_ι × ψ̄ 相図） ===
cargo run --release -- sweep \
    --beta-ivt-min 0.0 --beta-ivt-max 1.6 --beta-ivt-step 0.2 \
    --psafety-mean-values 0.3,0.5,0.7 \
    --runs 30 --seed 42

# === Ablation: IVT 必要性（rule vs rule_no_ivt） ===
cargo run --release -- ablation --modes rule,rule_no_ivt --seed-start 0 --seed-end 30

# === モード別アンカーレポート ===
cargo run --release -- reproduce --llm-mode rule --t-max 60 --runs 30

# Python 可視化・分析ツール（workspace ルート）
uv sync
uv run detert-tools visualize                 # 沈黙率時系列 + IVT ルールヒートマップ + 散布
uv run detert-tools visualize-sweep           # β_ι × ψ̄ 相図
uv run detert-tools show-experiment-settings  # config / sweep_config / llm_meta
uv run detert-tools reproduce                 # Table 4 風レポート + CFA 系適合度指標
```

## リポジトリ構成

```
detert2011/
├── simulation/                       # Rust socsim ABM
│   ├── Cargo.toml                    # socsim-{core,engine,net,llm,results} git 依存
│   ├── src/
│   │   ├── lib.rs / main.rs          # CLI: run / sweep / ablation / reproduce
│   │   ├── config.rs                 # Config / LlmMode / BetaGroup / NetworkKind
│   │   ├── world.rs                  # SilenceWorld + Employee + Team + IvtRule + Motive
│   │   ├── mechanisms.rs             # 9 機構 × 6 フェーズ；rule vs LLM 決定（排他）
│   │   ├── prompts.rs                # IVT 5 ルール内省プロンプト + 決定 JSON パーサ
│   │   ├── llm.rs                    # socsim-llm 共有ハーネス re-export shim
│   │   ├── simulation.rs             # init_world + run_with_client + CSV/JSON ライタ
│   │   └── metrics.rs                # upward_silence / rule_activation / 同時発火 / 相関
│   └── tests/integration_test.rs     # rule bit 決定論 + scripted-LLM スモーク
├── tools/                            # Python detert-tools
│   └── src/detert_tools/{cli,visualize,visualize_sweep,show_experiment_settings,
│                         reproduce_paper}.py
├── docs/                             # bilingual: architecture, cli, usecases, visualization, reproduction
└── results/                          # 実行時生成（gitignore）
    ├── latest -> {YYYYMMDD_HHMMSS}/
    └── {YYYYMMDD_HHMMSS}/
        ├── config.json | sweep_config.json
        ├── metrics.csv               # t, upward_silence_rate, rule_*, max_rule_cooccurrence, …
        ├── agents.csv                # 最終ステップの per-agent 状態 + active_rules
        ├── rule_activation.csv       # ステップ別ルール別発火率
        └── llm_meta.json             # LLM 来歴 + cache-hit + silence_voice_corr
```

## ドキュメント

- [アーキテクチャ](docs/architecture.ja.md) — world state，9 機構 × 6 フェーズ表，二層決定論
- [CLI リファレンス](docs/cli.ja.md) — `run` / `sweep` / `ablation` / `reproduce` フラグ
- [ユースケース](docs/usecases.ja.md) — 較正・ablation・スイープのワークフロー
- [可視化](docs/visualization.ja.md) — Python ツールの出力
- [再現](docs/reproduction.ja.md) — モデルと Detert & Edmondson 2011 の数値の対応

## 参考文献

- Detert, J. R., & Edmondson, A. C. (2011). Implicit Voice Theories: Taken-for-Granted Rules of Self-Censorship at Work. *Academy of Management Journal*, 54(3), 461–488.
- シミュレーションエンジン: [socsim (rs-social-simulation-tools)](https://github.com/akitenkrad/rs-social-simulation-tools).

## ライセンス

MIT — [LICENSE](LICENSE) を参照．

---
*This file was generated by Claude Code.*
