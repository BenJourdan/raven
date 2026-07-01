from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

import raven
from metrics import balanced_pair_array, pair_metrics, winner_summary
from reporting import summarize, write_csv, write_plot_files, write_report
from workloads import (
    expected_edges_per_node,
    prepare_diff_workload_sbm,
    query_subset,
)

WORKLOAD_SEED = 42
CORE_RNG_SEED = 42
N_PER_CLUSTER = 1024
NUM_CLUSTERS = 64
TOTAL_NODES = N_PER_CLUSTER * NUM_CLUSTERS
P_INTERNAL = 0.25
Q_EXTERNAL = 1.0 / TOTAL_NODES
N_MULTIPLIER = 2
LIFETIME_MULTIPLIER = 1.0
STEP_SIZE = 0.05
SIGMA = 1000.0
CORESET_SIZE = 2048
SAMPLING_SEEDS = 256
NUM_TRIALS = range(1, 20)
QUERY_FRAC = 0.1
DEGREE_REBUILD_THRESHOLD = 4096
PAIR_COUNT = 50_000
OUTPUT_DIR = Path("target/consensus_trial_scaling")


@dataclass(frozen=True)
class PairBatch:
    pairs: np.ndarray
    labels: list[bool]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_config(output_dir)
    print("generating dynamic SBM workload...")
    started = time.perf_counter()
    workload = prepare_diff_workload_sbm(
        seed=WORKLOAD_SEED,
        n_per_cluster=N_PER_CLUSTER,
        k_clusters=NUM_CLUSTERS,
        p_internal=P_INTERNAL,
        q_external=Q_EXTERNAL,
        n_multiplier=N_MULTIPLIER,
        lifetime_multiplier=LIFETIME_MULTIPLIER,
        step_size=STEP_SIZE,
    )
    print(f"workload generation: {time.perf_counter() - started:.3f}s")

    batches = workload.batches[: args.max_batches] if args.max_batches else workload.batches
    trial_counts = list(NUM_TRIALS)
    if args.max_trials:
        trial_counts = trial_counts[: args.max_trials]

    pair_cache: dict[int, PairBatch] = {}
    per_batch_rows: list[dict[str, float | int]] = []

    for num_trials in trial_counts:
        print(f"replaying workload with num_trials={num_trials}...")
        per_batch_rows.extend(
            replay_trial_count(
                num_trials,
                batches,
                workload.cluster_labels,
                pair_cache,
                no_progress=args.no_progress,
            )
        )

    summary_rows = summarize(per_batch_rows)
    write_csv(output_dir / "per_batch.csv", per_batch_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_report(
        output_dir / "report.html",
        per_batch_rows,
        summary_rows,
    )
    write_plot_files(
        output_dir / "plots",
        per_batch_rows,
        summary_rows,
    )

    print(f"wrote CSV: {output_dir / 'per_batch.csv'}")
    print(f"wrote CSV: {output_dir / 'summary.csv'}")
    print(f"wrote report: {output_dir / 'report.html'}")
    print(f"wrote plot files: {output_dir / 'plots'}")


def replay_trial_count(
    num_trials: int,
    batches,
    cluster_labels: list[int],
    pair_cache: dict[int, PairBatch],
    *,
    no_progress: bool,
) -> list[dict[str, float | int]]:
    index = raven.Raven(
        NUM_CLUSTERS,
        sigma=SIGMA,
        coreset_size=CORESET_SIZE,
        sampling_seeds=SAMPLING_SEEDS,
        num_trials=num_trials,
        rng_seed=CORE_RNG_SEED,
        node_capacity=TOTAL_NODES,
        expected_edges_per_node=expected_edges_per_node(
            N_PER_CLUSTER, NUM_CLUSTERS, P_INTERNAL, Q_EXTERNAL
        ),
        degree_rebuild_threshold=DEGREE_REBUILD_THRESHOLD,
    )

    rows: list[dict[str, float | int]] = []
    with tqdm(
        total=sum(len(batch.edge_ops) for batch in batches),
        disable=no_progress,
        unit=" updates",
    ) as pbar:
        for batch_index, batch in enumerate(batches):
            started = time.perf_counter()
            index.update_edges(batch.edge_ops)
            ingestion_s = time.perf_counter() - started

            started = time.perf_counter()
            index.flush()
            flush_s = time.perf_counter() - started

            live_nodes = index.live_nodes()
            if len(live_nodes) < CORESET_SIZE:
                pbar.update(len(batch.edge_ops))
                pbar.set_postfix_str(
                    f"trials={num_trials} batch={batch_index} live={len(live_nodes)} skipped"
                )
                continue

            query_nodes = query_subset(
                live_nodes,
                frac=QUERY_FRAC,
                min_size=NUM_CLUSTERS,
                seed=WORKLOAD_SEED,
            )
            true_labels = [cluster_labels[node] for node in query_nodes]

            started = time.perf_counter()
            consensus = index.query_consensus(query_nodes)
            query_s = time.perf_counter() - started

            pair_batch = pair_cache.get(batch_index)
            if pair_batch is None:
                pair_batch = build_pair_batch(
                    batch_index,
                    query_nodes,
                    cluster_labels,
                )
                pair_cache[batch_index] = pair_batch

            started = time.perf_counter()
            pair_scores = consensus.score_pairs(pair_batch.pairs)
            pair_score_s = time.perf_counter() - started

            winner_ari, winner_score, winner_k = winner_summary(true_labels, consensus)
            pair_auc, pair_ap = pair_metrics(pair_batch.labels, pair_scores)

            rows.append(
                {
                    "num_trials": num_trials,
                    "batch_index": batch_index,
                    "batch_time": batch.time,
                    "live_nodes": len(live_nodes),
                    "query_nodes": len(query_nodes),
                    "edge_ops": len(batch.edge_ops),
                    "ingestion_s": ingestion_s,
                    "flush_s": flush_s,
                    "query_s": query_s,
                    "pair_score_s": pair_score_s,
                    "pair_count": len(pair_batch.pairs),
                    "winner_ari": winner_ari,
                    "winner_score": winner_score,
                    "winner_num_clusters": winner_k,
                    "pair_roc_auc": pair_auc,
                    "pair_average_precision": pair_ap,
                }
            )

            pbar.update(len(batch.edge_ops))
            pbar.set_postfix_str(
                f"trials={num_trials} batch={batch_index} "
                f"q={len(query_nodes)} query={query_s * 1000:.1f}ms "
                f"pairs={pair_score_s * 1000:.1f}ms auc={pair_auc:.3f}"
            )

    return rows


def build_pair_batch(
    batch_index: int,
    query_nodes: list[int],
    cluster_labels: list[int],
) -> PairBatch:
    rng = np.random.default_rng(WORKLOAD_SEED + batch_index + 1_000_003)
    pairs, labels = balanced_pair_array(
        query_nodes,
        cluster_labels,
        PAIR_COUNT,
        rng=rng,
    )
    return PairBatch(pairs=pairs, labels=labels)


def print_config(output_dir: Path) -> None:
    rows = {
        "nodes": TOTAL_NODES,
        "clusters": NUM_CLUSTERS,
        "n_per_cluster": N_PER_CLUSTER,
        "p_internal": P_INTERNAL,
        "q_external": Q_EXTERNAL,
        "n_multiplier": N_MULTIPLIER,
        "lifetime_multiplier": LIFETIME_MULTIPLIER,
        "step_size": STEP_SIZE,
        "sigma": SIGMA,
        "coreset_size": CORESET_SIZE,
        "sampling_seeds": SAMPLING_SEEDS,
        "num_trials": f"{min(NUM_TRIALS)}..{max(NUM_TRIALS)}",
        "query_frac": QUERY_FRAC,
        "expected_edges_per_node": expected_edges_per_node(
            N_PER_CLUSTER, NUM_CLUSTERS, P_INTERNAL, Q_EXTERNAL
        ),
        "degree_rebuild_threshold": DEGREE_REBUILD_THRESHOLD,
        "pair_count": PAIR_COUNT,
        "output_dir": output_dir,
        "workload_seed": WORKLOAD_SEED,
        "core_rng_seed": CORE_RNG_SEED,
    }
    print("Config:")
    for key, value in rows.items():
        print(f"  {key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Raven query and consensus scaling for "
            f"num_trials={min(NUM_TRIALS)}..{max(NUM_TRIALS)}."
        )
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.max_trials is not None and not 1 <= args.max_trials <= len(NUM_TRIALS):
        parser.error(f"--max-trials must be between 1 and {len(NUM_TRIALS)}")
    if args.max_batches is not None and args.max_batches <= 0:
        parser.error("--max-batches must be positive")
    return args


if __name__ == "__main__":
    main()
