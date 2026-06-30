from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score
from tqdm import tqdm

import raven
from in_memory_playground import (
    balanced_query_pairs,
    expected_edges_per_node,
    prepare_diff_workload_sbm,
    query_subset,
)

WORKLOAD_SEED = 42
CORE_RNG_SEED = 42
N_PER_CLUSTER = 1024
NUM_CLUSTERS = 64
TOTAL_NODES = N_PER_CLUSTER * NUM_CLUSTERS
P_INTERNAL = 0.33
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
PAIR_COUNT = 100_000
OUTPUT_DIR = Path("target/consensus_trials_playground")


@dataclass(frozen=True)
class PairBatch:
    pairs: list[tuple[int, int]]
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
            trials = index.query_all_trials(query_nodes)
            query_s = time.perf_counter() - started

            started = time.perf_counter()
            consensus = raven.ConsensusResult.from_trials(query_nodes, trials)
            consensus_build_s = time.perf_counter() - started

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

            winner_ari, winner_score, winner_k = winner_summary(true_labels, trials)
            pair_auc = float(roc_auc_score(pair_batch.labels, pair_scores))
            pair_ap = float(average_precision_score(pair_batch.labels, pair_scores))

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
                    "consensus_build_s": consensus_build_s,
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
    pairs, labels = balanced_query_pairs(
        query_nodes,
        cluster_labels,
        PAIR_COUNT,
        rng=rng,
    )
    return PairBatch(pairs=pairs, labels=labels)


def winner_summary(true_labels: list[int], trials) -> tuple[float, float, int]:
    summaries = [
        (
            float(adjusted_rand_score(true_labels, trial.labels)),
            float(np.sum(trial.scores)) if trial.scores is not None else float(trial.trial_index),
            int(trial.num_clusters),
        )
        for trial in trials
    ]
    return min(summaries, key=lambda item: item[1])


def summarize(rows: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    summary_rows = []
    for num_trials in sorted({int(row["num_trials"]) for row in rows}):
        trial_rows = [row for row in rows if int(row["num_trials"]) == num_trials]
        if not trial_rows:
            continue
        summary_rows.append(
            {
                "num_trials": num_trials,
                "queried_batches": len(trial_rows),
                "total_ingestion_s": total(trial_rows, "ingestion_s"),
                "total_flush_s": total(trial_rows, "flush_s"),
                "total_query_s": total(trial_rows, "query_s"),
                "total_consensus_build_s": total(trial_rows, "consensus_build_s"),
                "total_pair_score_s": total(trial_rows, "pair_score_s"),
                "mean_query_ms": mean_ms(trial_rows, "query_s"),
                "mean_consensus_build_ms": mean_ms(trial_rows, "consensus_build_s"),
                "mean_pair_score_ms": mean_ms(trial_rows, "pair_score_s"),
                "mean_pair_us": mean_pair_us(trial_rows),
                "mean_winner_ari": mean(trial_rows, "winner_ari"),
                "mean_pair_roc_auc": mean(trial_rows, "pair_roc_auc"),
                "mean_pair_average_precision": mean(
                    trial_rows, "pair_average_precision"
                ),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows to write for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    per_batch_rows: list[dict[str, float | int]],
    summary_rows: list[dict[str, float | int]],
) -> None:
    early_batches = early_batch_indices(per_batch_rows)
    early_label = early_batch_label(early_batches, per_batch_rows)

    fig = make_subplots(
        rows=6,
        cols=1,
        specs=[
            [{"type": "surface"}],
            [{"type": "surface"}],
            [{"type": "surface"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
        ],
        row_heights=[0.19, 0.19, 0.19, 0.14, 0.14, 0.15],
        vertical_spacing=0.08,
        subplot_titles=[
            "Pair ROC-AUC Surface",
            "Query Time Surface",
            "Total Decision Time Surface",
            "Mean Time vs Trials",
            "Pair Metrics and Winner ARI vs Trials",
            f"Early Pair Metrics vs Trials ({early_label})",
        ],
    )

    batch_times, trial_counts, auc_z = surface_grid(
        per_batch_rows,
        lambda row: float(row["pair_roc_auc"]),
    )
    _, _, query_z = surface_grid(
        per_batch_rows,
        lambda row: float(row["query_s"]) * 1000.0,
    )
    _, _, total_decision_z = surface_grid(
        per_batch_rows,
        lambda row: (
            float(row["query_s"])
            + float(row["consensus_build_s"])
            + float(row["pair_score_s"])
        )
        * 1000.0,
    )

    add_surface(
        fig,
        batch_times,
        trial_counts,
        auc_z,
        "ROC-AUC",
        colorbar_x=1.02,
        colorbar_y=0.92,
        row=1,
        col=1,
    )
    add_surface(
        fig,
        batch_times,
        trial_counts,
        query_z,
        "query ms",
        colorbar_x=1.02,
        colorbar_y=0.72,
        row=2,
        col=1,
    )
    add_surface(
        fig,
        batch_times,
        trial_counts,
        total_decision_z,
        "total ms",
        colorbar_x=1.02,
        colorbar_y=0.52,
        row=3,
        col=1,
    )

    x_summary = [int(row["num_trials"]) for row in summary_rows]
    fig.add_trace(
        go.Scatter(
            x=x_summary,
            y=[float(row["mean_query_ms"]) for row in summary_rows],
            mode="lines+markers",
            name="query",
            hovertemplate="trials=%{x}<br>ms=%{y:.2f}<extra>%{fullData.name}</extra>",
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_summary,
            y=[
                float(row["mean_query_ms"])
                + float(row["mean_consensus_build_ms"])
                + float(row["mean_pair_score_ms"])
                for row in summary_rows
            ],
            mode="lines+markers",
            name="total decision",
            hovertemplate="trials=%{x}<br>ms=%{y:.2f}<extra>%{fullData.name}</extra>",
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_summary,
            y=[float(row["mean_consensus_build_ms"]) for row in summary_rows],
            mode="lines+markers",
            name="consensus build",
            hovertemplate="trials=%{x}<br>ms=%{y:.2f}<extra>%{fullData.name}</extra>",
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_summary,
            y=[float(row["mean_pair_score_ms"]) for row in summary_rows],
            mode="lines+markers",
            name="100k pair scoring",
            hovertemplate="trials=%{x}<br>ms=%{y:.2f}<extra>%{fullData.name}</extra>",
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_summary,
            y=[float(row["mean_pair_roc_auc"]) for row in summary_rows],
            mode="lines+markers",
            name="ROC-AUC",
            hovertemplate="trials=%{x}<br>score=%{y:.3f}<extra>%{fullData.name}</extra>",
        ),
        row=5,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_summary,
            y=[float(row["mean_pair_average_precision"]) for row in summary_rows],
            mode="lines+markers",
            name="average precision",
            hovertemplate="trials=%{x}<br>score=%{y:.3f}<extra>%{fullData.name}</extra>",
        ),
        row=5,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_summary,
            y=[float(row["mean_winner_ari"]) for row in summary_rows],
            mode="lines+markers",
            name="winner ARI",
            hovertemplate="trials=%{x}<br>score=%{y:.3f}<extra>%{fullData.name}</extra>",
        ),
        row=5,
        col=1,
    )

    early_rows = [
        row for row in per_batch_rows if int(row["batch_index"]) in early_batches
    ]
    early_summary = summarize(early_rows)
    fig.add_trace(
        go.Scatter(
            x=[int(row["num_trials"]) for row in early_summary],
            y=[float(row["mean_pair_roc_auc"]) for row in early_summary],
            mode="lines+markers",
            name="early ROC-AUC",
            hovertemplate="trials=%{x}<br>score=%{y:.3f}<extra>%{fullData.name}</extra>",
        ),
        row=6,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[int(row["num_trials"]) for row in early_summary],
            y=[float(row["mean_pair_average_precision"]) for row in early_summary],
            mode="lines+markers",
            name="early average precision",
            hovertemplate="trials=%{x}<br>score=%{y:.3f}<extra>%{fullData.name}</extra>",
        ),
        row=6,
        col=1,
    )

    fig.update_xaxes(title_text="num_trials", row=4, col=1)
    fig.update_xaxes(title_text="num_trials", row=5, col=1)
    fig.update_xaxes(title_text="num_trials", row=6, col=1)
    fig.update_yaxes(title_text="ms", tickformat=".2f", row=4, col=1)
    fig.update_yaxes(title_text="score", tickformat=".3f", row=5, col=1)
    fig.update_yaxes(title_text="score", tickformat=".3f", row=6, col=1)
    fig.update_layout(
        title=(
            "Raven Consensus Trial Scaling"
            f"<br><sup>Early = {early_label}</sup>"
        ),
        height=4200,
        legend_title="series",
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.025,
            "yanchor": "top",
        },
        margin={"r": 130, "b": 130, "t": 110},
        scene=surface_scene("batch time", "num_trials", "ROC-AUC"),
        scene2=surface_scene("batch time", "num_trials", "query ms"),
        scene3=surface_scene("batch time", "num_trials", "total ms"),
    )
    fig.write_html(path)


def write_plot_files(
    output_dir: Path,
    per_batch_rows: list[dict[str, float | int]],
    summary_rows: list[dict[str, float | int]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    early_batches = early_batch_indices(per_batch_rows)
    early_label = early_batch_label(early_batches, per_batch_rows)
    early_rows = [
        row for row in per_batch_rows if int(row["batch_index"]) in early_batches
    ]
    early_summary = summarize(early_rows)

    batch_times, trial_counts, auc_z = surface_grid(
        per_batch_rows,
        lambda row: float(row["pair_roc_auc"]),
    )
    _, _, query_z = surface_grid(
        per_batch_rows,
        lambda row: float(row["query_s"]) * 1000.0,
    )
    _, _, total_decision_z = surface_grid(
        per_batch_rows,
        lambda row: (
            float(row["query_s"])
            + float(row["consensus_build_s"])
            + float(row["pair_score_s"])
        )
        * 1000.0,
    )

    plots = [
        (
            "pair_roc_auc_surface.html",
            "Pair ROC-AUC Surface",
            standalone_surface_figure(
                "Pair ROC-AUC Surface",
                batch_times,
                trial_counts,
                auc_z,
                "ROC-AUC",
            ),
        ),
        (
            "query_time_surface.html",
            "Query Time Surface",
            standalone_surface_figure(
                "Query Time Surface",
                batch_times,
                trial_counts,
                query_z,
                "query ms",
            ),
        ),
        (
            "total_decision_time_surface.html",
            "Total Decision Time Surface",
            standalone_surface_figure(
                "Total Decision Time Surface",
                batch_times,
                trial_counts,
                total_decision_z,
                "total ms",
            ),
        ),
        (
            "mean_time_vs_trials.html",
            "Mean Time vs Trials",
            mean_time_figure(summary_rows),
        ),
        (
            "pair_metrics_winner_ari.html",
            "Pair Metrics and Winner ARI vs Trials",
            pair_metrics_figure(summary_rows),
        ),
        (
            "early_pair_metrics.html",
            f"Early Pair Metrics vs Trials ({early_label})",
            early_pair_metrics_figure(early_summary, early_label),
        ),
    ]

    for filename, _, fig in plots:
        fig.write_html(output_dir / filename)
    write_plot_index(output_dir, [(filename, title) for filename, title, _ in plots])


def standalone_surface_figure(
    title: str,
    batch_times: list[float],
    trial_counts: list[int],
    z: list[list[float]],
    colorbar_title: str,
) -> go.Figure:
    fig = go.Figure(
        data=[
            surface_trace(
                batch_times,
                trial_counts,
                z,
                colorbar_title,
                colorbar_len=0.65,
            )
        ]
    )
    fig.update_layout(
        title=title,
        height=950,
        margin={"r": 100, "b": 70, "t": 80},
        scene=surface_scene("batch time", "num_trials", colorbar_title),
    )
    return fig


def mean_time_figure(summary_rows: list[dict[str, float | int]]) -> go.Figure:
    x_summary = [int(row["num_trials"]) for row in summary_rows]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_summary,
            y=[float(row["mean_query_ms"]) for row in summary_rows],
            mode="lines+markers",
            name="query",
            hovertemplate="trials=%{x}<br>ms=%{y:.2f}<extra>%{fullData.name}</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_summary,
            y=[
                float(row["mean_query_ms"])
                + float(row["mean_consensus_build_ms"])
                + float(row["mean_pair_score_ms"])
                for row in summary_rows
            ],
            mode="lines+markers",
            name="total decision",
            hovertemplate="trials=%{x}<br>ms=%{y:.2f}<extra>%{fullData.name}</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_summary,
            y=[float(row["mean_consensus_build_ms"]) for row in summary_rows],
            mode="lines+markers",
            name="consensus build",
            hovertemplate="trials=%{x}<br>ms=%{y:.2f}<extra>%{fullData.name}</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_summary,
            y=[float(row["mean_pair_score_ms"]) for row in summary_rows],
            mode="lines+markers",
            name="100k pair scoring",
            hovertemplate="trials=%{x}<br>ms=%{y:.2f}<extra>%{fullData.name}</extra>",
        )
    )
    style_line_figure(fig, "Mean Time vs Trials", "ms")
    return fig


def pair_metrics_figure(summary_rows: list[dict[str, float | int]]) -> go.Figure:
    x_summary = [int(row["num_trials"]) for row in summary_rows]
    fig = go.Figure()
    fig.add_trace(
        score_trace(
            x_summary,
            [float(row["mean_pair_roc_auc"]) for row in summary_rows],
            "ROC-AUC",
        )
    )
    fig.add_trace(
        score_trace(
            x_summary,
            [float(row["mean_pair_average_precision"]) for row in summary_rows],
            "average precision",
        )
    )
    fig.add_trace(
        score_trace(
            x_summary,
            [float(row["mean_winner_ari"]) for row in summary_rows],
            "winner ARI",
        )
    )
    style_line_figure(fig, "Pair Metrics and Winner ARI vs Trials", "score")
    return fig


def early_pair_metrics_figure(
    early_summary: list[dict[str, float | int]],
    early_label: str,
) -> go.Figure:
    x_summary = [int(row["num_trials"]) for row in early_summary]
    fig = go.Figure()
    fig.add_trace(
        score_trace(
            x_summary,
            [float(row["mean_pair_roc_auc"]) for row in early_summary],
            "early ROC-AUC",
        )
    )
    fig.add_trace(
        score_trace(
            x_summary,
            [float(row["mean_pair_average_precision"]) for row in early_summary],
            "early average precision",
        )
    )
    style_line_figure(fig, f"Early Pair Metrics vs Trials ({early_label})", "score")
    return fig


def score_trace(x: list[int], y: list[float], name: str) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        mode="lines+markers",
        name=name,
        hovertemplate="trials=%{x}<br>score=%{y:.3f}<extra>%{fullData.name}</extra>",
    )


def style_line_figure(fig: go.Figure, title: str, y_title: str) -> None:
    tickformat = ".2f" if y_title == "ms" else ".3f"
    fig.update_xaxes(title_text="num_trials")
    fig.update_yaxes(title_text=y_title, tickformat=tickformat)
    fig.update_layout(
        title=title,
        height=700,
        legend_title="series",
        margin={"r": 40, "b": 70, "t": 80},
    )


def write_plot_index(output_dir: Path, plots: list[tuple[str, str]]) -> None:
    links = "\n".join(
        f'<li><a href="{filename}">{title}</a></li>' for filename, title in plots
    )
    output_dir.joinpath("index.html").write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head><meta charset=\"utf-8\"><title>Raven Consensus Plots</title></head>",
                "<body>",
                "<h1>Raven Consensus Trial Scaling</h1>",
                "<ul>",
                links,
                "</ul>",
                "</body>",
                "</html>",
            ]
        )
    )


def surface_grid(
    rows: list[dict[str, float | int]],
    value_fn,
) -> tuple[list[float], list[int], list[list[float]]]:
    batches = sorted({int(row["batch_index"]) for row in rows})
    trials = sorted({int(row["num_trials"]) for row in rows})
    row_by_key = {
        (int(row["num_trials"]), int(row["batch_index"])): row for row in rows
    }
    batch_times = [
        float(next(row["batch_time"] for row in rows if int(row["batch_index"]) == batch))
        for batch in batches
    ]
    z = []
    for num_trials in trials:
        z.append(
            [
                value_fn(row_by_key[(num_trials, batch)])
                if (num_trials, batch) in row_by_key
                else float("nan")
                for batch in batches
            ]
        )
    return batch_times, trials, z


def add_surface(
    fig,
    batch_times: list[float],
    trial_counts: list[int],
    z: list[list[float]],
    colorbar_title: str,
    *,
    colorbar_x: float,
    colorbar_y: float,
    row: int,
    col: int,
) -> None:
    fig.add_trace(
        surface_trace(
            batch_times,
            trial_counts,
            z,
            colorbar_title,
            colorbar_len=0.13,
            colorbar_x=colorbar_x,
            colorbar_y=colorbar_y,
        ),
        row=row,
        col=col,
    )


def surface_trace(
    batch_times: list[float],
    trial_counts: list[int],
    z: list[list[float]],
    colorbar_title: str,
    *,
    colorbar_len: float,
    colorbar_x: float | None = None,
    colorbar_y: float | None = None,
) -> go.Surface:
    colorbar = {
        "title": colorbar_title,
        "len": colorbar_len,
        "thickness": 12,
    }
    if colorbar_x is not None:
        colorbar["x"] = colorbar_x
    if colorbar_y is not None:
        colorbar["y"] = colorbar_y
    return go.Surface(
        x=batch_times,
        y=trial_counts,
        z=z,
        hovertemplate=(
            "batch time=%{x:.4g}<br>"
            "trials=%{y}<br>"
            f"{colorbar_title}=%{{z:.4g}}"
            "<extra></extra>"
        ),
        colorbar=colorbar,
    )


def surface_scene(x_title: str, y_title: str, z_title: str) -> dict:
    return {
        "xaxis": {"title": x_title},
        "yaxis": {"title": y_title},
        "zaxis": {"title": z_title},
    }


def early_batch_indices(rows: list[dict[str, float | int]]) -> list[int]:
    one_trial_rows = sorted(
        (row for row in rows if int(row["num_trials"]) == 1),
        key=lambda row: int(row["batch_index"]),
    )
    early = [
        int(row["batch_index"])
        for row in one_trial_rows
        if float(row["pair_roc_auc"]) < 0.98
    ]
    if early:
        return early
    return [int(row["batch_index"]) for row in one_trial_rows[:5]]


def early_batch_label(
    early_batches: list[int],
    rows: list[dict[str, float | int]],
) -> str:
    all_batches = sorted({int(row["batch_index"]) for row in rows})
    pct = len(early_batches) / len(all_batches) * 100.0 if all_batches else 0.0
    if not early_batches:
        return "no early batches"
    if early_batches == list(range(min(early_batches), max(early_batches) + 1)):
        batch_text = f"batches {min(early_batches)}-{max(early_batches)}"
    else:
        batch_text = "batches " + ",".join(str(batch) for batch in early_batches)
    return f"{batch_text}; first {pct:.0f}% where 1-trial AUC < 0.98"


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
        "num_trials": "1..10",
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
        description="Measure Raven query and consensus scaling for num_trials=1..10."
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.max_trials is not None and not 1 <= args.max_trials <= 10:
        parser.error("--max-trials must be between 1 and 10")
    if args.max_batches is not None and args.max_batches <= 0:
        parser.error("--max-batches must be positive")
    return args


def total(rows: list[dict[str, float | int]], field: str) -> float:
    return float(sum(float(row[field]) for row in rows))


def mean(rows: list[dict[str, float | int]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def mean_ms(rows: list[dict[str, float | int]], field: str) -> float:
    return mean(rows, field) * 1000.0


def mean_pair_us(rows: list[dict[str, float | int]]) -> float:
    return float(
        np.mean(
            [
                float(row["pair_score_s"]) / float(row["pair_count"]) * 1_000_000.0
                for row in rows
            ]
        )
    )


def ms(rows: list[dict[str, float | int]], field: str) -> list[float]:
    return [float(row[field]) * 1000.0 for row in rows]


if __name__ == "__main__":
    main()
