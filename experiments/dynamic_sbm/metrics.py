from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score

import raven
from workloads import balanced_query_pairs


def winner_summary(
    true_labels: list[int],
    consensus: raven.ConsensusResult,
) -> tuple[float, float, int]:
    """Return `(ARI, score, cluster_count)` for the lowest-score trial."""
    summaries = [
        (
            float(adjusted_rand_score(true_labels, labels)),
            float(score),
            int(num_clusters),
        )
        for labels, score, num_clusters in zip(
            consensus.labels,
            consensus.trial_scores,
            consensus.num_clusters,
            strict=True,
        )
    ]
    return min(summaries, key=lambda item: item[1])


def balanced_pair_array(
    query_nodes: list[int],
    cluster_labels: list[int],
    pair_count: int,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[bool]]:
    pairs, pair_labels = balanced_query_pairs(
        query_nodes,
        cluster_labels,
        pair_count,
        rng=rng,
    )
    pair_array = (
        np.asarray(pairs, dtype=np.uintp).reshape(-1, 2)
        if pairs
        else np.empty((0, 2), dtype=np.uintp)
    )
    return pair_array, pair_labels


def pair_metrics(pair_labels: list[bool], pair_scores) -> tuple[float, float]:
    if len(set(pair_labels)) != 2:
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(pair_labels, pair_scores)),
        float(average_precision_score(pair_labels, pair_scores)),
    )


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")
