from __future__ import annotations

from statistics import fmean

import raven

import _helpers as h


# Raven parameters:
CORESET_SIZE = 256
SAMPLING_SEEDS = 64
NUM_TRIALS = 5
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
roc_aucs: list[float] = []
avg_precisions: list[float] = []

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

    # generate balanced same-cluster/cross-cluster node pairs for this snapshot
    pairs, pair_labels = h.balanced_pair_batch(
        snapshot,
        workload.cluster_labels,
        workload_seed=WORKLOAD_SEED,
    )

    # query the unique nodes in those pairs and score each pair using trial consensus
    pair_scores = index.score_pairs(pairs)
    roc_auc, avg_precision = h.pair_score_metrics(pair_labels, pair_scores)
    roc_aucs.append(roc_auc)
    avg_precisions.append(avg_precision)
    print(
        f"batch={snapshot.batch_index:02d} "
        f"time={snapshot.batch_time} "
        f"updates={len(batch.edge_ops)} "
        f"pairs={h.PAIR_COUNT} "
        f"roc_auc={roc_auc:.3f} "
        f"avg_precision={avg_precision:.3f}"
    )

print(f"mean ROC-AUC: {fmean(roc_aucs):.3f}")
print(f"mean average precision: {fmean(avg_precisions):.3f}")
