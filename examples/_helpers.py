from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score

import raven

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments" / "dynamic_sbm"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from workloads import (
    balanced_query_pairs,
    expected_edges_per_node,
    prepare_diff_workload_sbm,
    query_subset,
)

N_PER_CLUSTER = 128
NUM_CLUSTERS = 8
TOTAL_NODES = N_PER_CLUSTER * NUM_CLUSTERS
P_INTERNAL = 0.30
Q_EXTERNAL = 1.0 / TOTAL_NODES
N_MULTIPLIER = 2
LIFETIME_MULTIPLIER = 1.0
STEP_SIZE = 0.10
SIGMA = 1000.0
QUERY_FRAC = 0.20
PAIR_COUNT = 10_000
DEGREE_REBUILD_THRESHOLD = 4096
EXPECTED_EDGES_PER_NODE = expected_edges_per_node(
    N_PER_CLUSTER,
    NUM_CLUSTERS,
    P_INTERNAL,
    Q_EXTERNAL,
)
INDEX_KWARGS = {
    "sigma": SIGMA,
    "node_capacity": TOTAL_NODES,
    "expected_edges_per_node": EXPECTED_EDGES_PER_NODE,
    "degree_rebuild_threshold": DEGREE_REBUILD_THRESHOLD,
}


@dataclass(frozen=True)
class Snapshot:
    batch_index: int
    batch_time: int
    query_nodes: list[int]
    true_labels: list[int]


def make_workload(*, seed: int):
    return prepare_diff_workload_sbm(
        seed=seed,
        n_per_cluster=N_PER_CLUSTER,
        k_clusters=NUM_CLUSTERS,
        p_internal=P_INTERNAL,
        q_external=Q_EXTERNAL,
        n_multiplier=N_MULTIPLIER,
        lifetime_multiplier=LIFETIME_MULTIPLIER,
        step_size=STEP_SIZE,
    )


def query_snapshot(
    index: raven.Raven,
    workload,
    batch_index: int,
    batch,
    *,
    coreset_size: int,
    workload_seed: int,
) -> Snapshot | None:
    live_nodes = index.live_nodes()
    if len(live_nodes) < coreset_size:
        return None

    query_nodes = query_subset(
        live_nodes,
        frac=QUERY_FRAC,
        min_size=NUM_CLUSTERS,
        seed=workload_seed,
    )
    return Snapshot(
        batch_index=batch_index,
        batch_time=batch.time,
        query_nodes=query_nodes,
        true_labels=[workload.cluster_labels[node] for node in query_nodes],
    )


def ari(true_labels: list[int], predicted_labels: list[int]) -> float:
    return float(adjusted_rand_score(true_labels, predicted_labels))


def balanced_pair_batch(
    snapshot: Snapshot,
    cluster_labels: list[int],
    *,
    workload_seed: int,
) -> tuple[np.ndarray, list[bool]]:
    rng = np.random.default_rng(workload_seed + snapshot.batch_index + 1_000_003)
    pairs, pair_labels = balanced_query_pairs(
        snapshot.query_nodes,
        cluster_labels,
        PAIR_COUNT,
        rng=rng,
    )
    pair_array = np.asarray(pairs, dtype=np.uintp).reshape(-1, 2)
    return pair_array, pair_labels


def pair_score_metrics(pair_labels: list[bool], pair_scores) -> tuple[float, float]:
    return (
        float(roc_auc_score(pair_labels, pair_scores)),
        float(average_precision_score(pair_labels, pair_scores)),
    )
