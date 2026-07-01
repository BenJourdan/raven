from __future__ import annotations

from statistics import fmean

import raven

import _helpers as h


# Raven parameters:
CORESET_SIZE = 256
SAMPLING_SEEDS = 64
NUM_TRIALS = 1
RAVEN_RNG_SEED = 42


# Workload parameters:
WORKLOAD_SEED = 42


workload = h.make_workload(seed=WORKLOAD_SEED)

# initialize Raven index:
index = raven.Raven(
    h.NUM_CLUSTERS,
    coreset_size=CORESET_SIZE,
    sampling_seeds=SAMPLING_SEEDS,
    num_trials=NUM_TRIALS,
    rng_seed=RAVEN_RNG_SEED,
    **h.INDEX_KWARGS,
)

scores: list[float] = []

for batch_index, batch in enumerate(workload.batches):

    # edge_ops is a list of (u, v, weight) tuples, where weight=None indicates deletion
    index.update_edges(batch.edge_ops)

    # flush the index to apply all updates before querying
    # (triggers on query anyway)
    index.flush()

    # get info about the current snapshot
    # including which nodes the workload wants to query + their true labels
    snapshot = h.query_snapshot(
        index,
        workload,
        batch_index,
        batch,
        coreset_size=CORESET_SIZE,
        workload_seed=WORKLOAD_SEED,
    )
    if snapshot is None:
        continue

    # query the index
    result = index.query(snapshot.query_nodes)
    score = h.ari(snapshot.true_labels, result.labels)
    scores.append(score)
    print(
        f"batch={snapshot.batch_index:02d} "
        f"time={snapshot.batch_time} "
        f"updates={len(batch.edge_ops)} "
        f"nodes={len(snapshot.query_nodes)} "
        f"ari={score:.3f}"
    )

print(f"mean ARI: {fmean(scores):.3f}")
