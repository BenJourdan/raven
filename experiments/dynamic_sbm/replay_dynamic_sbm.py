from __future__ import annotations

import argparse
import math
import time

import igraph as ig
import numpy as np
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

import raven
from metrics import balanced_pair_array, pair_metrics, winner_summary
from workloads import (
    apply_edge_ops_to_weight_map,
    default_sampling_seeds,
    expected_edges_per_node,
    parse_pair_sizes,
    prepare_diff_workload_sbm,
    query_subset,
)


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
            consensus = index.query_consensus(
                query_nodes,
                trial_weighting=args.consensus_weighting,
                temperature=args.consensus_temperature,
            )
            query_time += time.perf_counter() - started

            winner_ari, winner_score, winner_k = winner_summary(true_labels, consensus)
            ari_history.append((batch.time, winner_ari))

            if (len(ari_history) - 1) % args.consensus_every == 0:
                pair_rng = np.random.default_rng(
                    args.workload_seed + batch_index + 1_000_003
                )
                for pair_count in args.consensus_pair_sizes:
                    pair_array, pair_labels = balanced_pair_array(
                        query_nodes,
                        workload.cluster_labels,
                        pair_count,
                        rng=pair_rng,
                    )
                    if len(pair_array) == 0:
                        continue

                    started = time.perf_counter()
                    pair_scores = consensus.score_pairs(pair_array)
                    consensus_pair_time[pair_count] += time.perf_counter() - started
                    consensus_pair_runs[pair_count] += 1

                    pair_auc, pair_ap = pair_metrics(pair_labels, pair_scores)
                    if math.isfinite(pair_auc):
                        consensus_pair_auc[pair_count].append(pair_auc)
                    if math.isfinite(pair_ap):
                        consensus_pair_ap[pair_count].append(pair_ap)

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


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


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
