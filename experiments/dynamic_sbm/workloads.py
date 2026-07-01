from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UpdateBatch:
    time: int
    edge_ops: list[tuple[int, int, float | None]]


@dataclass(frozen=True)
class SbmWorkload:
    cluster_labels: list[int]
    expected_edges: int
    snapshot_step: int
    batches: list[UpdateBatch]


def prepare_diff_workload_sbm(
    *,
    seed: int,
    n_per_cluster: int,
    k_clusters: int,
    p_internal: float,
    q_external: float,
    n_multiplier: int,
    lifetime_multiplier: float,
    step_size: float,
) -> SbmWorkload:
    validate_sbm_params(
        n_per_cluster,
        k_clusters,
        p_internal,
        q_external,
        n_multiplier,
        lifetime_multiplier,
        step_size,
    )

    total_nodes = n_per_cluster * k_clusters
    expected_internal = (
        k_clusters * n_per_cluster * (n_per_cluster - 1) * 0.5 * p_internal
    )
    expected_external = (
        k_clusters
        * (k_clusters - 1)
        * 0.5
        * n_per_cluster
        * n_per_cluster
        * q_external
    )
    expected_edges = expected_internal + expected_external
    if not math.isfinite(expected_edges) or expected_edges <= 0.0:
        raise ValueError("expected edge count must be positive")

    rng = np.random.default_rng(seed)
    num_updates = math.ceil(n_multiplier * expected_edges)
    lifetime_steps = max(1, math.ceil(lifetime_multiplier * expected_edges))
    internal_prob = expected_internal / expected_edges
    snapshot_step = max(1, int(step_size * max(1, num_updates - 1)))
    next_snapshot = snapshot_step

    edge_weights: dict[tuple[int, int], float] = {}
    expirations: deque[tuple[int, int, int]] = deque()
    batches: list[UpdateBatch] = []
    batch_ops: list[tuple[int, int, float | None]] = []

    for t in range(num_updates):
        while expirations and t - expirations[0][0] >= lifetime_steps:
            _, u, v = expirations.popleft()
            key = ordered_edge(u, v)
            old_weight = edge_weights.get(key)
            if old_weight is None:
                continue

            new_weight = max(0.0, old_weight - 1.0)
            if new_weight == 0.0:
                del edge_weights[key]
                batch_ops.append((u, v, None))
            else:
                edge_weights[key] = new_weight
                batch_ops.append((u, v, new_weight))

        if rng.random() < internal_prob:
            u, v = pick_internal(rng, n_per_cluster, k_clusters)
        else:
            u, v = pick_cross(rng, n_per_cluster, k_clusters)

        if u != v:
            key = ordered_edge(u, v)
            new_weight = edge_weights.get(key, 0.0) + 1.0
            edge_weights[key] = new_weight
            expirations.append((t, u, v))
            batch_ops.append((u, v, new_weight))

        if t >= next_snapshot:
            if batch_ops:
                batches.append(UpdateBatch(time=t, edge_ops=batch_ops))
                batch_ops = []
            while t >= next_snapshot:
                next_snapshot += snapshot_step

    if batch_ops:
        batches.append(UpdateBatch(time=num_updates - 1, edge_ops=batch_ops))

    return SbmWorkload(
        cluster_labels=[node // n_per_cluster for node in range(total_nodes)],
        expected_edges=int(expected_edges),
        snapshot_step=snapshot_step,
        batches=batches,
    )


def pick_internal(
    rng: np.random.Generator,
    n_per_cluster: int,
    k_clusters: int,
) -> tuple[int, int]:
    cluster = int(rng.integers(0, k_clusters))
    start = cluster * n_per_cluster
    if n_per_cluster == 1:
        return start, start

    a = int(rng.integers(0, n_per_cluster))
    b = int(rng.integers(0, n_per_cluster - 1))
    if b >= a:
        b += 1
    return start + a, start + b


def pick_cross(
    rng: np.random.Generator,
    n_per_cluster: int,
    k_clusters: int,
) -> tuple[int, int]:
    if k_clusters < 2:
        return pick_internal(rng, n_per_cluster, k_clusters)

    c1 = int(rng.integers(0, k_clusters))
    c2 = int(rng.integers(0, k_clusters - 1))
    if c2 >= c1:
        c2 += 1

    u = c1 * n_per_cluster + int(rng.integers(0, n_per_cluster))
    v = c2 * n_per_cluster + int(rng.integers(0, n_per_cluster))
    return u, v


def ordered_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def apply_edge_ops_to_weight_map(
    edge_ops: list[tuple[int, int, float | None]],
    edge_weights: dict[tuple[int, int], float],
) -> None:
    for u, v, weight in edge_ops:
        edge = ordered_edge(u, v)
        if weight is None:
            edge_weights.pop(edge, None)
        else:
            edge_weights[edge] = weight


def balanced_query_pairs(
    query_nodes: list[int],
    labels: list[int],
    pair_count: int,
    *,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], list[bool]]:
    by_label: dict[int, list[int]] = {}
    for node in query_nodes:
        by_label.setdefault(labels[node], []).append(node)

    same_labels = [label for label, nodes in by_label.items() if len(nodes) >= 2]
    cross_labels = list(by_label)
    same_target = pair_count // 2
    cross_target = pair_count - same_target

    pairs: list[tuple[int, int]] = []
    pair_labels: list[bool] = []

    for _ in range(same_target):
        if not same_labels:
            break
        label = same_labels[int(rng.integers(0, len(same_labels)))]
        nodes = by_label[label]
        i = int(rng.integers(0, len(nodes)))
        j = int(rng.integers(0, len(nodes) - 1))
        if j >= i:
            j += 1
        pairs.append((nodes[i], nodes[j]))
        pair_labels.append(True)

    for _ in range(cross_target):
        if len(cross_labels) < 2:
            break
        left_label = cross_labels[int(rng.integers(0, len(cross_labels)))]
        right_candidates = [label for label in cross_labels if label != left_label]
        right_label = right_candidates[int(rng.integers(0, len(right_candidates)))]
        left_nodes = by_label[left_label]
        right_nodes = by_label[right_label]
        pairs.append(
            (
                left_nodes[int(rng.integers(0, len(left_nodes)))],
                right_nodes[int(rng.integers(0, len(right_nodes)))],
            )
        )
        pair_labels.append(False)

    return pairs, pair_labels


def query_subset(
    live_nodes: list[int],
    *,
    frac: float,
    min_size: int,
    seed: int,
) -> list[int]:
    query_len = max(min_size, int(len(live_nodes) * frac))
    query_len = min(query_len, len(live_nodes))

    nodes = np.array(live_nodes, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(nodes)
    return sorted(int(node) for node in nodes[:query_len])


def expected_edges_per_node(
    n_per_cluster: int,
    k_clusters: int,
    p_internal: float,
    q_external: float,
) -> int:
    total_nodes = n_per_cluster * k_clusters
    expected_internal = math.ceil((n_per_cluster - 1) * p_internal)
    expected_external = math.ceil((total_nodes - n_per_cluster) * q_external)
    return max(1, expected_internal + expected_external)


def default_sampling_seeds(num_clusters: int, coreset_size: int) -> int:
    return max(2, min(num_clusters * 8, coreset_size - 1))


def parse_pair_sizes(value: str) -> list[int]:
    sizes = []
    for raw_size in value.split(","):
        raw_size = raw_size.strip()
        if not raw_size:
            continue
        size = int(raw_size)
        if size <= 0:
            raise argparse.ArgumentTypeError("pair sizes must be positive")
        sizes.append(size)
    return sizes


def validate_sbm_params(
    n_per_cluster: int,
    k_clusters: int,
    p_internal: float,
    q_external: float,
    n_multiplier: int,
    lifetime_multiplier: float,
    step_size: float,
) -> None:
    if n_per_cluster <= 0:
        raise ValueError("n_per_cluster must be positive")
    if k_clusters <= 0:
        raise ValueError("num_clusters must be positive")
    if n_multiplier <= 0:
        raise ValueError("n_multiplier must be positive")
    if not math.isfinite(p_internal) or p_internal < 0.0:
        raise ValueError("p_internal must be finite and non-negative")
    if not math.isfinite(q_external) or q_external < 0.0:
        raise ValueError("q_external must be finite and non-negative")
    if not math.isfinite(lifetime_multiplier) or lifetime_multiplier <= 0.0:
        raise ValueError("lifetime_multiplier must be positive and finite")
    if not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be positive and finite")
