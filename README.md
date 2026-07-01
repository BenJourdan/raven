# Raven

Raven is a dynamic graph clustering index. The current public API focuses on an
in-memory backend with Python and Rust entry points: update weighted edges,
query a node subset, and optionally combine multiple query trials into lazy
pairwise consensus scores.

## Python Quickstart

Install the local development build:

```bash
uv sync
uv run maturin develop --release
```

Run Python commands with `--no-sync` after `maturin develop` so uv does not
replace the release extension while benchmarking:

```bash
uv run --no-sync python examples/single_trial_ari.py
uv run --no-sync python examples/pair_consensus_scores.py
```

Minimal usage:

```python
import numpy as np
import raven

index = raven.Raven(
    2,
    coreset_size=4,
    sampling_seeds=2,
    num_trials=3,
    rng_seed=42,
)

index.update_edges(
    [
        (1, 2, 1.0),
        (2, 3, 1.0),
        (10, 11, 1.0),
        (11, 12, 1.0),
    ]
)

result = index.query([1, 2, 3, 10, 11, 12])
print(result.labels)

consensus = index.query_consensus([1, 2, 3, 10, 11, 12])
pairs = np.array([[1, 2], [1, 10], [10, 11]], dtype=np.uintp)
print(consensus.score_pairs(pairs))
```

`score_pairs` is fastest with a contiguous `Nx2` NumPy array using
`dtype=np.uintp`. Lists of `(u, v)` pairs also work.

## Rust Quickstart

```rust
use raven::{Raven, RavenConfig, TrialWeighting};

fn main() -> Result<(), raven::RavenError> {
    let mut config = RavenConfig::new(2);
    config.coreset_size = 3;
    config.sampling_seeds = 2;
    config.num_trials = 3;
    config.rng_seed = Some(42);

    let mut index = Raven::new(config)?;
    index.update_edge(1, 2, 1.0)?;
    index.update_edge(2, 3, 1.0)?;

    let result = index.query(&[1, 2, 3])?;
    println!("{:?}", result.labels);

    let consensus = index.query_consensus(
        &[1, 2, 3],
        TrialWeighting::ScoreSoftmax,
        None,
    )?;
    println!("{}", consensus.score_pair(1, 2)?);

    Ok(())
}
```

Runnable Rust examples:

```bash
cargo run -p raven --example single_trial_ari
cargo run -p raven --example pair_consensus_scores
```

## Development Checks

```bash
cargo fmt --all --check
cargo test --workspace
env -u CONDA_PREFIX uv run maturin develop --release
env -u CONDA_PREFIX uv run --no-sync pytest
```

## Experiments

Longer-running dynamic SBM experiments live outside the package in
`experiments/dynamic_sbm`:

```bash
uv run --no-sync python experiments/dynamic_sbm/replay_dynamic_sbm.py --help
uv run --no-sync python experiments/dynamic_sbm/consensus_trial_scaling.py --help
```
