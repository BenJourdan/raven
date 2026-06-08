use std::{hint::black_box, sync::Arc, time::Duration};

use criterion::{Criterion, criterion_group};
use raven_adapters::in_memory::{
    InMemoryUndirectedGraph,
    workloads::{SbmDiffWorkload, prepare_diff_workload_sbm},
};
use raven_core::{
    DynamicClusteringAlg, GraphOracle,
    alg::DynamicClustering,
    types::{AlgType, PartitionOutput, PartitionType, Strict, TrialOutputMode},
};

#[cfg(feature = "bench-clustering")]
use raven_core::clustering::{LeidenConfig, leiden_community_detection_alg};

const WORKLOAD_SEED: u64 = 42;
const CORE_RNG_SEED: u64 = 42;

const N_PER_CLUSTER: usize = 500;
const NUM_CLUSTERS: usize = 50;
const TOTAL_NODES: usize = N_PER_CLUSTER * NUM_CLUSTERS;
const P_INTERNAL: f64 = 0.33;
const Q_EXTERNAL: f64 = 1.0 / TOTAL_NODES as f64;
const N_MULTIPLIER: usize = 1;
const LIFETIME_MULTIPLIER: f64 = 1.0;
const STEP_SIZE: f64 = 0.01;

const NUM_TRIALS: usize = 1;
const CORESET_SIZE: usize = NUM_CLUSTERS * 10;
const SAMPLING_SEEDS: usize = NUM_CLUSTERS * 2;

type BenchClustering = DynamicClustering<64, usize, f64>;

fn criterion_config() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5))
}

fn strict(value: f64) -> Strict<f64> {
    Strict::<f64>::new(value).expect("benchmark constants should be positive and finite")
}

fn workload() -> SbmDiffWorkload<f64> {
    prepare_diff_workload_sbm::<f64>(
        WORKLOAD_SEED,
        N_PER_CLUSTER,
        NUM_CLUSTERS,
        P_INTERNAL,
        Q_EXTERNAL,
        N_MULTIPLIER,
        LIFETIME_MULTIPLIER,
        STEP_SIZE,
    )
    .expect("deterministic SBM workload should build")
}

fn replay_graph(workload: &SbmDiffWorkload<f64>) -> InMemoryUndirectedGraph<usize, f64> {
    let mut graph = InMemoryUndirectedGraph::new();
    for batch in &workload.batches {
        batch
            .apply_to_graph(&mut graph)
            .expect("generated workload should replay into graph");
        let _ = graph.flush_node_ops();
    }
    graph
}

fn live_nodes(
    workload: &SbmDiffWorkload<f64>,
    graph: &InMemoryUndirectedGraph<usize, f64>,
) -> Vec<usize> {
    workload
        .nodes
        .iter()
        .copied()
        .filter(|node| graph.contains_node(*node))
        .collect()
}

fn query_subset(live_nodes: &[usize]) -> Vec<usize> {
    let requested = (live_nodes.len() / 10).max(CORESET_SIZE.min(live_nodes.len()));
    live_nodes
        .iter()
        .copied()
        .take(requested.min(live_nodes.len()))
        .collect()
}

fn dummy_cluster_alg() -> AlgType<f64> {
    Arc::new(|graph, _| {
        let n = graph.symbolic().nrows();
        (vec![0; n], usize::from(n > 0))
    })
}

fn build_query_fixture(
    workload: &SbmDiffWorkload<f64>,
    cluster_alg: AlgType<f64>,
) -> (
    BenchClustering,
    InMemoryUndirectedGraph<usize, f64>,
    Vec<usize>,
) {
    let mut graph = InMemoryUndirectedGraph::new();
    let mut clustering = DynamicClustering::new(cluster_alg)
        .with_sigma(strict(1000.0))
        .with_num_trials(NUM_TRIALS)
        .with_coreset_size(CORESET_SIZE)
        .with_sampling_seeds(SAMPLING_SEEDS)
        .with_rng_seed(CORE_RNG_SEED)
        .with_num_clusters(NUM_CLUSTERS)
        .with_prop_name("w");

    for batch in &workload.batches {
        batch
            .apply_to_graph(&mut graph)
            .expect("generated workload should replay into graph");
        let node_ops = graph.flush_node_ops();
        debug_assert_eq!(node_ops, batch.node_ops);
        clustering
            .apply_node_ops(&node_ops)
            .expect("generated node ops should replay into core");
    }

    let live_nodes = live_nodes(workload, &graph);
    assert!(
        live_nodes.len() >= CORESET_SIZE,
        "benchmark workload produced {} live nodes, but coreset_size is {}",
        live_nodes.len(),
        CORESET_SIZE
    );
    let subset = query_subset(&live_nodes);
    assert!(
        subset.len() >= CORESET_SIZE,
        "benchmark query subset produced {} nodes, but coreset_size is {}",
        subset.len(),
        CORESET_SIZE
    );

    (clustering, graph, subset)
}

fn run_subset_query(
    clustering: &mut BenchClustering,
    graph: &InMemoryUndirectedGraph<usize, f64>,
    subset: &[usize],
) {
    let mut oracle_handles = graph.oracles(NUM_TRIALS);
    let mut oracle_refs = oracle_handles.iter_mut().collect::<Vec<_>>();
    let output = clustering
        .query(
            PartitionType::Subset(std::hint::black_box(subset)),
            TrialOutputMode::AllTrials,
            &mut oracle_refs,
        )
        .expect("benchmark query should succeed");

    match &output {
        PartitionOutput::Subset(trials) => {
            assert_eq!(trials.len(), NUM_TRIALS);
            assert!(
                trials
                    .iter()
                    .all(|trial| trial.labels.len() == subset.len())
            );
        }
        PartitionOutput::All(_, _) => panic!("subset query should return subset output"),
    }
    black_box(output);
}

fn bench_oracle_batch_lookup(c: &mut Criterion) {
    let workload = workload();
    let graph = replay_graph(&workload);
    let live_nodes = live_nodes(&workload, &graph);
    assert!(
        live_nodes.len() >= CORESET_SIZE,
        "benchmark workload produced {} live nodes, but coreset_size is {}",
        live_nodes.len(),
        CORESET_SIZE
    );

    let graph_batch = query_subset(&live_nodes);
    let coreset_batch = live_nodes
        .iter()
        .copied()
        .take(CORESET_SIZE)
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("in_memory/oracle");
    group.bench_function("graph_neighbourhoods", |b| {
        let mut oracle = graph.oracle();
        b.iter(|| {
            let neighbourhoods = oracle
                .graph_neighbourhoods(black_box(graph_batch.as_slice()))
                .expect("graph neighbourhood batch should succeed");
            black_box(neighbourhoods.data().len());
            black_box(neighbourhoods.offsets().len());
        });
    });
    group.bench_function("coreset_neighbourhoods", |b| {
        let mut oracle = graph.oracle();
        b.iter(|| {
            let neighbourhoods = oracle
                .coreset_neighbourhoods(black_box(coreset_batch.as_slice()))
                .expect("coreset neighbourhood batch should succeed");
            black_box(neighbourhoods.data().len());
            black_box(neighbourhoods.offsets().len());
        });
    });
    group.finish();
}

fn bench_query_dummy_subset(c: &mut Criterion) {
    let workload = workload();
    let (mut clustering, graph, subset) = build_query_fixture(&workload, dummy_cluster_alg());

    c.bench_function("in_memory/query_dummy_subset", |b| {
        b.iter(|| run_subset_query(&mut clustering, &graph, &subset));
    });
}

#[cfg(feature = "bench-clustering")]
fn bench_query_leiden_subset(c: &mut Criterion) {
    let workload = workload();
    let cluster_alg = leiden_community_detection_alg::<f64>(LeidenConfig {
        seed: Some(42),
        ..LeidenConfig::default()
    });
    let (mut clustering, graph, subset) = build_query_fixture(&workload, cluster_alg);

    c.bench_function("in_memory/query_leiden_subset", |b| {
        b.iter(|| run_subset_query(&mut clustering, &graph, &subset));
    });
}

fn profile_once_oracle_graph(repeat: usize) {
    let workload = workload();
    let graph = replay_graph(&workload);
    let live_nodes = live_nodes(&workload, &graph);
    let graph_batch = query_subset(&live_nodes);

    let mut oracle = graph.oracle();
    for _ in 0..repeat {
        let neighbourhoods = oracle
            .graph_neighbourhoods(black_box(graph_batch.as_slice()))
            .expect("graph neighbourhood batch should succeed");
        black_box(neighbourhoods.data().len());
        black_box(neighbourhoods.offsets().len());
    }
}

fn profile_once_oracle_coreset(repeat: usize) {
    let workload = workload();
    let graph = replay_graph(&workload);
    let live_nodes = live_nodes(&workload, &graph);
    assert!(
        live_nodes.len() >= CORESET_SIZE,
        "benchmark workload produced {} live nodes, but coreset_size is {}",
        live_nodes.len(),
        CORESET_SIZE
    );
    let coreset_batch = live_nodes
        .iter()
        .copied()
        .take(CORESET_SIZE)
        .collect::<Vec<_>>();

    let mut oracle = graph.oracle();
    for _ in 0..repeat {
        let neighbourhoods = oracle
            .coreset_neighbourhoods(black_box(coreset_batch.as_slice()))
            .expect("coreset neighbourhood batch should succeed");
        black_box(neighbourhoods.data().len());
        black_box(neighbourhoods.offsets().len());
    }
}

fn profile_once_dummy_query(repeat: usize) {
    let workload = workload();
    let (mut clustering, graph, subset) = build_query_fixture(&workload, dummy_cluster_alg());
    for _ in 0..repeat {
        run_subset_query(&mut clustering, &graph, &subset);
    }
}

#[cfg(feature = "bench-clustering")]
fn profile_once_leiden_query(repeat: usize) {
    let workload = workload();
    let cluster_alg = leiden_community_detection_alg::<f64>(LeidenConfig {
        seed: Some(42),
        ..LeidenConfig::default()
    });
    let (mut clustering, graph, subset) = build_query_fixture(&workload, cluster_alg);
    for _ in 0..repeat {
        run_subset_query(&mut clustering, &graph, &subset);
    }
}

#[cfg(not(feature = "bench-clustering"))]
fn profile_once_leiden_query(_repeat: usize) {
    panic!("Leiden profiling requires --features bench-clustering");
}

fn run_profile_once_target(target: &str, repeat: usize) {
    match target {
        "dummy" | "query-dummy" | "query_dummy_subset" => profile_once_dummy_query(repeat),
        "leiden" | "query-leiden" | "query_leiden_subset" => profile_once_leiden_query(repeat),
        "oracle-graph" | "graph" | "graph_neighbourhoods" => profile_once_oracle_graph(repeat),
        "oracle-coreset" | "coreset" | "coreset_neighbourhoods" => {
            profile_once_oracle_coreset(repeat)
        }
        other => panic!("unknown profile-once target: {other}"),
    }
}

fn maybe_run_profile_once() -> bool {
    let mut args = std::env::args().skip(1);
    let Some(first) = args.next() else {
        return false;
    };
    if first != "--profile-once" {
        return false;
    }

    let target = args.next().unwrap_or_else(|| "dummy".to_string());
    let repeat = args
        .next()
        .and_then(|value| value.parse::<usize>().ok())
        .or_else(|| {
            std::env::var("PROFILE_REPEAT")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
        })
        .unwrap_or(1);

    assert!(repeat > 0, "profile repeat must be non-zero");

    run_profile_once_target(&target, repeat);
    true
}

#[cfg(feature = "bench-clustering")]
criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_oracle_batch_lookup, bench_query_dummy_subset, bench_query_leiden_subset
}

#[cfg(not(feature = "bench-clustering"))]
criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_oracle_batch_lookup, bench_query_dummy_subset
}

fn main() {
    if maybe_run_profile_once() {
        return;
    }

    benches();
    Criterion::default().configure_from_args().final_summary();
}
