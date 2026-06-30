from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import dataclass

import igraph as ig
import numpy as np
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score
from tqdm import tqdm

import raven


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


def main() -> None:
    args = parse_args()

    total_nodes = args.n_per_cluster * args.num_clusters
    q_external = args.q_external if args.q_external is not None else 1.0 / total_nodes
    sampling_seeds = args.sampling_seeds or default_sampling_seeds(
        args.num_clusters, args.coreset_size
    )
    node_capacity = args.node_capacity or total_nodes
    expected_degree = args.expected_edges_per_node or expected_edges_per_node(
        args.n_per_cluster,
        args.num_clusters,
        args.p_internal,
        q_external,
    )

    print_config(args, total_nodes, q_external, sampling_seeds, node_capacity, expected_degree)

    print("generating dynamic SBM workload...")
    workload_started = time.perf_counter()
    workload = prepare_diff_workload_sbm(
        seed=args.workload_seed,
        n_per_cluster=args.n_per_cluster,
        k_clusters=args.num_clusters,
        p_internal=args.p_internal,
        q_external=q_external,
        n_multiplier=args.n_multiplier,
        lifetime_multiplier=args.lifetime_multiplier,
        step_size=args.step_size,
    )
    workload_elapsed = time.perf_counter() - workload_started

    index = raven.Raven(
        args.num_clusters,
        sigma=args.sigma,
        coreset_size=args.coreset_size,
        sampling_seeds=sampling_seeds,
        num_trials=args.num_trials,
        rng_seed=args.core_rng_seed,
        node_capacity=node_capacity,
        expected_edges_per_node=expected_degree,
        degree_rebuild_threshold=args.degree_rebuild_threshold,
    )

    total_updates = sum(len(batch.edge_ops) for batch in workload.batches)
    total_started = time.perf_counter()
    ingestion_time = 0.0
    flush_time = 0.0
    query_time = 0.0
    queried_nodes_total = 0
    ari_history: list[tuple[int, float]] = []
    control_edges: dict[tuple[int, int], float] = {}
    control_update_time = 0.0
    control_build_time = 0.0
    control_leiden_time = 0.0
    control_runs = 0
    control_ari_history: list[tuple[int, float]] = []
    consensus_build_time = 0.0
    consensus_pair_time = {size: 0.0 for size in args.consensus_pair_sizes}
    consensus_pair_auc = {size: [] for size in args.consensus_pair_sizes}
    consensus_pair_ap = {size: [] for size in args.consensus_pair_sizes}
    consensus_pair_runs = {size: 0 for size in args.consensus_pair_sizes}

    batches = workload.batches
    if args.max_batches is not None:
        batches = batches[: args.max_batches]

    with tqdm(
        total=sum(len(batch.edge_ops) for batch in batches),
        disable=args.no_progress,
        unit=" updates",
    ) as pbar:
        for batch_index, batch in enumerate(batches):
            started = time.perf_counter()
            index.update_edges(batch.edge_ops)
            ingestion_time += time.perf_counter() - started

            if args.control_leiden:
                started = time.perf_counter()
                apply_edge_ops_to_weight_map(batch.edge_ops, control_edges)
                control_update_time += time.perf_counter() - started

            started = time.perf_counter()
            index.flush()
            flush_time += time.perf_counter() - started

            live_nodes = index.live_nodes()
            if len(live_nodes) < args.coreset_size:
                pbar.update(len(batch.edge_ops))
                pbar.set_postfix_str(
                    f"batch={batch_index} live={len(live_nodes)} skipped"
                )
                continue

            query_nodes = query_subset(
                live_nodes,
                frac=args.query_frac,
                min_size=args.num_clusters,
                seed=args.workload_seed,
            )
            true_labels = [workload.cluster_labels[node] for node in query_nodes]
            queried_nodes_total += len(query_nodes)

            started = time.perf_counter()
            trials = index.query_all_trials(query_nodes)
            query_time += time.perf_counter() - started

            trial_summaries = [
                (
                    float(adjusted_rand_score(true_labels, trial.labels)),
                    sum(trial.scores) if trial.scores is not None else float(trial.trial_index),
                    trial.num_clusters,
                )
                for trial in trials
            ]
            winner_ari, winner_score, winner_k = min(
                trial_summaries, key=lambda item: item[1]
            )
            ari_history.append((batch.time, winner_ari))

            if (len(ari_history) - 1) % args.consensus_every == 0:
                started = time.perf_counter()
                consensus = raven.ConsensusResult.from_trials(
                    query_nodes,
                    trials,
                    trial_weighting=args.consensus_weighting,
                    temperature=args.consensus_temperature,
                )
                consensus_build_time += time.perf_counter() - started

                pair_rng = np.random.default_rng(
                    args.workload_seed + batch_index + 1_000_003
                )
                for pair_count in args.consensus_pair_sizes:
                    pairs, pair_labels = balanced_query_pairs(
                        query_nodes,
                        workload.cluster_labels,
                        pair_count,
                        rng=pair_rng,
                    )
                    if not pairs:
                        continue

                    started = time.perf_counter()
                    pair_scores = consensus.score_pairs(pairs)
                    consensus_pair_time[pair_count] += time.perf_counter() - started
                    consensus_pair_runs[pair_count] += 1

                    if len(set(pair_labels)) == 2:
                        consensus_pair_auc[pair_count].append(
                            float(roc_auc_score(pair_labels, pair_scores))
                        )
                        consensus_pair_ap[pair_count].append(
                            float(average_precision_score(pair_labels, pair_scores))
                        )

            control_ari_text = ""
            if args.control_leiden and (len(ari_history) - 1) % args.control_every == 0:
                labels, build_elapsed, leiden_elapsed = run_igraph_leiden(
                    total_nodes,
                    control_edges,
                    n_iterations=args.control_iterations,
                )
                control_build_time += build_elapsed
                control_leiden_time += leiden_elapsed
                control_runs += 1
                control_ari = float(
                    adjusted_rand_score(true_labels, [labels[node] for node in query_nodes])
                )
                control_ari_history.append((batch.time, control_ari))
                control_ari_text = f" control_ari={control_ari:.3f}"

            pbar.update(len(batch.edge_ops))
            pbar.set_postfix_str(
                f"batch={batch_index} live={len(live_nodes)} "
                f"q={len(query_nodes)} ari={winner_ari:.3f} "
                f"score={winner_score:.3f} k={winner_k}{control_ari_text}"
            )

    total_elapsed = time.perf_counter() - total_started
    queried_batches = len(ari_history)
    avg_query_nodes = queried_nodes_total / queried_batches if queried_batches else 0.0

    print(f"batches: {len(workload.batches)}")
    if args.max_batches is not None:
        print(f"replayed batches: {len(batches)}")
    print(f"nodes: {total_nodes} total")
    print(f"expected edges: {workload.expected_edges}")
    print(f"total updates: {total_updates}")
    print(f"queried batches: {queried_batches}")
    print(f"workload generation: {workload_elapsed:.3f} seconds")
    print(f"total replay elapsed: {total_elapsed:.3f} seconds")
    print("Timing breakdown:")
    print(f"  edge ingestion: {ingestion_time:.3f} seconds")
    print(f"  flush/data structure updates: {flush_time:.3f} seconds")
    print(f"  data structure queries: {query_time:.3f} seconds")
    if queried_batches:
        print(
            "  query avg: "
            f"{(query_time / queried_batches) * 1000.0:.3f} ms/query, "
            f"avg query nodes {avg_query_nodes:.1f}"
        )
    if args.control_leiden:
        print("Control Leiden (igraph full graph):")
        print(f"  control graph updates: {control_update_time:.3f} seconds")
        print(f"  graph build: {control_build_time:.3f} seconds")
        print(f"  Leiden clustering: {control_leiden_time:.3f} seconds")
        print(f"  runs: {control_runs}")
        if control_runs:
            print(
                "  avg full-graph Leiden: "
                f"{(control_leiden_time / control_runs) * 1000.0:.3f} ms/run"
            )
    if queried_batches and args.consensus_pair_sizes:
        print("Consensus benchmark:")
        print(
            "  construction: "
            f"{consensus_build_time:.3f} seconds "
            f"({(consensus_build_time / queried_batches) * 1000.0:.3f} ms/query)"
        )
        for pair_count in args.consensus_pair_sizes:
            runs = consensus_pair_runs[pair_count]
            if runs == 0:
                continue
            elapsed = consensus_pair_time[pair_count]
            auc = mean(consensus_pair_auc[pair_count])
            ap = mean(consensus_pair_ap[pair_count])
            print(
                f"  {pair_count:g} balanced pairs: "
                f"{elapsed:.3f}s total, "
                f"{(elapsed / runs) * 1000.0:.3f} ms/run, "
                f"{(elapsed / (runs * pair_count)) * 1_000_000.0:.3f} us/pair, "
                f"roc_auc={auc:.3f}, avg_precision={ap:.3f}"
            )

    print("ARI history (batch time, winner ARI):")
    print([(str(t), f"{ari:.3f}") for t, ari in ari_history])
    if args.control_leiden:
        print("Control ARI history (batch time, igraph Leiden ARI):")
        print([(str(t), f"{ari:.3f}") for t, ari in control_ari_history])


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


def run_igraph_leiden(
    total_nodes: int,
    edge_weights: dict[tuple[int, int], float],
    *,
    n_iterations: int,
) -> tuple[list[int], float, float]:
    started = time.perf_counter()
    edges = list(edge_weights)
    weights = list(edge_weights.values())
    graph = ig.Graph(n=total_nodes, edges=edges, directed=False)
    build_elapsed = time.perf_counter() - started

    if not edges:
        return list(range(total_nodes)), build_elapsed, 0.0

    started = time.perf_counter()
    partition = graph.community_leiden(
        objective_function="modularity",
        weights=weights,
        n_iterations=n_iterations,
    )
    leiden_elapsed = time.perf_counter() - started
    return partition.membership, build_elapsed, leiden_elapsed


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


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


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


def print_config(
    args: argparse.Namespace,
    total_nodes: int,
    q_external: float,
    sampling_seeds: int,
    node_capacity: int,
    expected_degree: int,
) -> None:
    rows = {
        "nodes": total_nodes,
        "clusters": args.num_clusters,
        "n_per_cluster": args.n_per_cluster,
        "p_internal": args.p_internal,
        "q_external": q_external,
        "n_multiplier": args.n_multiplier,
        "lifetime_multiplier": args.lifetime_multiplier,
        "step_size": args.step_size,
        "sigma": args.sigma,
        "coreset_size": args.coreset_size,
        "sampling_seeds": sampling_seeds,
        "num_trials": args.num_trials,
        "query_frac": args.query_frac,
        "node_capacity": node_capacity,
        "expected_edges_per_node": expected_degree,
        "degree_rebuild_threshold": args.degree_rebuild_threshold,
        "control_leiden": args.control_leiden,
        "control_every": args.control_every,
        "control_iterations": args.control_iterations,
        "consensus_every": args.consensus_every,
        "consensus_pair_sizes": args.consensus_pair_sizes,
        "consensus_weighting": args.consensus_weighting,
        "consensus_temperature": args.consensus_temperature,
        "workload_seed": args.workload_seed,
        "core_rng_seed": args.core_rng_seed,
    }
    print("Config:")
    for key, value in rows.items():
        print(f"  {key}: {value}")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a dynamic SBM edge-update workload through the Python Raven API."
        )
    )
    parser.add_argument("--workload-seed", type=int, default=42)
    parser.add_argument("--core-rng-seed", type=int, default=42)
    parser.add_argument("--n-per-cluster", type=int, default=1024)
    parser.add_argument("--num-clusters", type=int, default=32)
    parser.add_argument("--p-internal", type=float, default=0.33)
    parser.add_argument(
        "--q-external",
        type=float,
        default=None,
        help="Defaults to 1 / total_nodes, matching the Rust playground.",
    )
    parser.add_argument("--n-multiplier", type=int, default=2)
    parser.add_argument("--lifetime-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.05,
        help="Snapshot batch size as a fraction of generated update time.",
    )
    parser.add_argument("--sigma", type=float, default=1000.0)
    parser.add_argument("--coreset-size", type=int, default=2048)
    parser.add_argument("--sampling-seeds", type=int, default=None)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--query-frac", type=float, default=0.1)
    parser.add_argument("--node-capacity", type=int, default=None)
    parser.add_argument("--expected-edges-per-node", type=int, default=None)
    parser.add_argument("--degree-rebuild-threshold", type=int, default=4096)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--control-leiden",
        action="store_true",
        help="Also run igraph full-graph Leiden at query snapshots.",
    )
    parser.add_argument(
        "--control-every",
        type=int,
        default=1,
        help="Run the control every N Raven query snapshots.",
    )
    parser.add_argument(
        "--control-iterations",
        type=int,
        default=2,
        help="igraph Leiden n_iterations value.",
    )
    parser.add_argument(
        "--consensus-every",
        type=int,
        default=1,
        help="Run the consensus pair benchmark every N Raven query snapshots.",
    )
    parser.add_argument(
        "--consensus-pair-sizes",
        type=parse_pair_sizes,
        default=parse_pair_sizes("1000,10000,100000"),
        help="Comma-separated balanced pair counts for lazy consensus scoring.",
    )
    parser.add_argument(
        "--consensus-weighting",
        choices=("uniform", "inverse_score", "score_softmax"),
        default="score_softmax",
    )
    parser.add_argument(
        "--consensus-temperature",
        type=float_or_auto,
        default="auto",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.control_every <= 0:
        parser.error("--control-every must be positive")
    if args.control_iterations <= 0:
        parser.error("--control-iterations must be positive")
    if args.consensus_every <= 0:
        parser.error("--consensus-every must be positive")
    return args


def float_or_auto(value: str) -> float | str:
    if value == "auto":
        return value
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("temperature must be positive and finite")
    return parsed


if __name__ == "__main__":
    main()
