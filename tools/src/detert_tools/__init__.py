"""detert-tools — visualization, sweep analysis, and reproduction utilities for
the Detert & Edmondson (2011) Implicit Voice Theories silence replication.

Modules:
- `visualize`              — silence-rate time series, IVT rule-firing heatmap,
                             silence–voice scatter.
- `visualize_sweep`        — β_ι × ψ̄ phase diagram of upward silence.
- `show_experiment_settings` — pretty-print a results directory's config / meta.
- `reproduce_paper`        — Table-4-style report + CFA-style fit indices
                             (RMSEA / CFI) reproduced from the ABM rule-firing
                             matrix.

All subcommands dispatch through `detert_tools.cli:main` — see `detert-tools --help`.
"""

__version__ = "0.1.0"
