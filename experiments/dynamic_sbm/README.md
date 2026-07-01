# Dynamic SBM Experiments

These scripts are benchmark and exploration drivers, not minimal API examples.
For short examples, see the top-level `examples/` directory.

Build the release Python extension first:

```bash
uv sync
uv run maturin develop --release
```

Then run experiments with `--no-sync` so uv does not replace the release
extension before execution:

```bash
uv run --no-sync python experiments/dynamic_sbm/replay_dynamic_sbm.py --help
uv run --no-sync python experiments/dynamic_sbm/consensus_trial_scaling.py --help
```

`replay_dynamic_sbm.py` is the configurable replay experiment. It can time edge
ingestion, Raven queries, lazy pair scoring, and optionally an igraph Leiden
control.

`consensus_trial_scaling.py` is a hardcoded trial-scaling report generator. It
writes CSV and Plotly artifacts under `target/consensus_trial_scaling/`.
