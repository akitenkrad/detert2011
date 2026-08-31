"""detert-tools — unified CLI dispatcher.

    detert-tools visualize                 # silence-rate series + rule heatmap + scatter
    detert-tools visualize-sweep           # β_ι × ψ̄ phase diagram (from the sweep's children)
    detert-tools show-experiment-settings  # print a run's parameters / llm block / run metrics
    detert-tools reproduce                 # Table-4-style report + CFA-style fit indices

Arguments after the subcommand are passed verbatim to that subcommand's argparse.
Add `--help` after a subcommand for its own help.

The dispatcher assembly is delegated to the shared helper
`socsim_tools.cli.build_dispatcher`.
"""

from __future__ import annotations

from socsim_tools.cli import build_dispatcher

main = build_dispatcher(
    prog="detert-tools",
    description="Detert & Edmondson (2011) Implicit Voice Theories — visualization + reproduction",
    subcommands={
        "visualize": (
            "single-run visualization (silence-rate series + IVT rule heatmap + silence-voice scatter)",
            "detert_tools.visualize:main",
        ),
        "visualize-sweep": (
            "sweep visualization (β_ι × ψ̄ phase diagram of upward silence)",
            "detert_tools.visualize_sweep:main",
        ),
        "show-experiment-settings": (
            "print a run's settings (parameters / llm block / run-scope metrics)",
            "detert_tools.show_experiment_settings:main",
        ),
        "reproduce": (
            "Table-4-style report + CFA-style fit indices (RMSEA / CFI) from the ABM rule-firing matrix",
            "detert_tools.reproduce_paper:main",
        ),
    },
)


if __name__ == "__main__":
    main()
