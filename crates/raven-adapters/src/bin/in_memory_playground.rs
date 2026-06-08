#[cfg(feature = "bench-clustering")]
fn main() -> playground::Result<()> {
    playground::run()
}

#[cfg(not(feature = "bench-clustering"))]
fn main() {
    eprintln!(
        "in_memory_playground runs the Leiden clustering path; use:\n\
         cargo run -p raven-adapters --features bench-clustering --bin in_memory_playground"
    );
}

#[cfg(feature = "bench-clustering")]
mod playground {
    use std::{error::Error, num::NonZeroUsize, time::Instant};

    use rand::{RngExt, SeedableRng};
    use raven_adapters::in_memory::{
        InMemoryUndirectedGraph,
        workloads::{SbmDiffWorkload, prepare_diff_workload_sbm},
    };
    use raven_core::{
        DynamicClusteringAlg,
        alg::{DynamicClustering, ResizeQueryInfo},
        clustering::{LeidenConfig, leiden_community_detection_alg},
        metrics::adjusted_rand_index,
        types::{PartitionOutput, PartitionType, Strict, TrialOutputMode},
    };

    use indicatif::{ProgressBar, ProgressStyle};

    pub type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync + 'static>>;

    const WORKLOAD_SEED: u64 = 42;
    const CORE_RNG_SEED: u64 = 42;
    const N_PER_CLUSTER: usize = 1024;
    const NUM_CLUSTERS: usize = 256;
    const TOTAL_NODES: usize = N_PER_CLUSTER * NUM_CLUSTERS;
    const P_INTERNAL: f64 = 0.5;
    const Q_EXTERNAL: f64 = 1.0 / TOTAL_NODES as f64;
    const N_MULTIPLIER: usize = 1;
    const LIFETIME_MULTIPLIER: f64 = 1.0;
    const STEP_SIZE: f64 = 0.01;
    const SIGMA: f64 = 1000.0;
    const DEGREE_CACHE_THRESHOLD: usize = 4096;

    const NUM_TRIALS: usize = 3;
    const CORESET_SIZE: usize = 4096*2;
    const SAMPLING_SEEDS: usize = NUM_CLUSTERS * 8;
    const QUERY_FRAC: f64 = 0.01;

    #[derive(Debug)]
    struct TrialSummary {
        trial_index: usize,
        ari: f64,
        score: f64,
        num_clusters: usize,
    }

    pub fn run() -> Result<()> {
        let workload = workload()?;
        let mut graph = InMemoryUndirectedGraph::with_capacity(
            NonZeroUsize::new(TOTAL_NODES).expect("total node count is non-zero"),
            expected_edges_per_node(),
            NonZeroUsize::new(DEGREE_CACHE_THRESHOLD)
            .expect("degree rebuild threshold is non-zero"),
        );

        let alg = leiden_community_detection_alg(LeidenConfig {
            seed: Some(42),
            ..LeidenConfig::default()
        });
        let cluster_alg = alg;
        let mut clustering = DynamicClustering::<128, usize, f64>::new(cluster_alg)
            .with_sigma(strict(SIGMA)?)
            .with_num_trials(NUM_TRIALS)
            .with_coreset_size(CORESET_SIZE)
            .with_sampling_seeds(SAMPLING_SEEDS)
            .with_rng_seed(CORE_RNG_SEED)
            .with_num_clusters(NUM_CLUSTERS)
            .with_resize_query_info(ResizeQueryInfo::Updates)
            .with_prop_name("w");

        let total_updates = workload
            .batches
            .iter()
            .map(|batch| batch.edge_ops.len())
            .sum::<usize>();

        let pbar_style = ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} updates ({eta_precise} remaining) - batch {batch_idx}",
        )?;
        let pbar = ProgressBar::new(total_updates as u64).with_style(pbar_style);

        let total_started = Instant::now();
        let mut ari_history = Vec::new();

        let mut graph_update_time = 0;
        let mut node_ops_time = 0;
        let mut data_structure_update_time = 0;
        let mut data_structure_query_time = 0;

        for (batch_idx, batch) in workload.batches.iter().enumerate() {
            // time graph updates:
            let graph_update_started = Instant::now();
            batch.apply_to_graph(&mut graph)?;
            graph_update_time += graph_update_started.elapsed().as_micros();

            // time node ops flush:
            let node_ops_started = Instant::now();
            let node_ops = graph.flush_node_ops();
            node_ops_time += node_ops_started.elapsed().as_micros();

            // time clustering updates:
            let data_structure_update_started = Instant::now();
            clustering.apply_node_ops(&node_ops)?;
            data_structure_update_time += data_structure_update_started.elapsed().as_micros();

            let live_nodes = live_nodes(&workload, &graph);
            if live_nodes.len() < CORESET_SIZE {
                println!(
                    "batch {batch_idx}: time={}, live={}, skipped until live >= coreset_size",
                    batch.time,
                    live_nodes.len()
                );
                continue;
            }

            let query_nodes = query_subset(&live_nodes, QUERY_FRAC)?;
            let true_labels = true_labels(&workload, &query_nodes)?;
            validate_query_label_mapping(&workload, &query_nodes, &true_labels)?;

            // time clustering queries:
            let query_started = Instant::now();

            let mut oracles = graph.oracles(NUM_TRIALS);
            let mut oracle_refs = oracles.iter_mut().collect::<Vec<_>>();

            let output = clustering.query(
                PartitionType::Subset(&query_nodes),
                TrialOutputMode::AllTrials,
                &mut oracle_refs,
            )?;
            data_structure_query_time += query_started.elapsed().as_micros();
            let summaries = trial_summaries_for_output(&true_labels, &output);
            let aris_scores = summaries
                .iter()
                .map(|summary| (summary.ari, summary.score))
                .collect::<Vec<_>>();

            let _summary_text = summaries
                .iter()
                .map(format_trial_summary)
                .collect::<Vec<_>>()
                .join(", ");

            let _best_ari = aris_scores
                .iter()
                .map(|(ari, _)| *ari)
                .fold(f64::NEG_INFINITY, f64::max);
            let _worst_ari = aris_scores
                .iter()
                .map(|(ari, _)| *ari)
                .fold(f64::INFINITY, f64::min);

            let winner_ari = aris_scores.iter().fold((0.0, f64::INFINITY), |best, nxt| {
                let (cur_ari, cur_score) = best;
                let (ari, score) = *nxt;
                if score < cur_score {
                    (ari, score)
                } else {
                    (cur_ari, cur_score)
                }
            });

            // println!(
            //     "batch {batch_idx}: time={}, live={}, query={}, query_elapsed={:?}, best_ari={:.6}, worst_ari={:.6}, winner_ari={:.6}, trial_summaries=[{}]",
            //     batch.time,
            //     live_nodes.len(),
            //     query_nodes.len(),
            //     query_elapsed,
            //     best_ari,
            //     worst_ari,
            //     winner_ari.0,
            //     summary_text
            // );
            pbar.inc(batch.edge_ops.len() as u64);

            ari_history.push((batch.time, winner_ari));
        }

        pbar.finish_with_message("done");

        println!("batches: {}", workload.batches.len());
        println!("nodes: {TOTAL_NODES} total");
        println!("queried batches: {}", ari_history.len());
        println!("total elapsed: {:?}", total_started.elapsed());

        println!("Timing breakdown:");
        println!(
            "  graph updates: {:.3} seconds",
            graph_update_time as f64 / 1_000_000.0
        );
        println!(
            "  node ops flush: {:.3} seconds",
            node_ops_time as f64 / 1_000_000.0
        );
        println!(
            "  data structure updates: {:.3} seconds",
            data_structure_update_time as f64 / 1_000_000.0
        );
        println!(
            "  data structure queries: {:.3} seconds",
            data_structure_query_time as f64 / 1_000_000.0
        );

        println!("ARI history (batch time, winner ARI):");
        println!("{:?}", ari_history.iter().map(|(time, ari)| (format!("{time}"), format!("{:.3}", ari.0))).collect::<Vec<_>>());

        Ok(())
    }

    fn workload() -> Result<SbmDiffWorkload<f64>> {
        let workload = prepare_diff_workload_sbm::<f64>(
            WORKLOAD_SEED,
            N_PER_CLUSTER,
            NUM_CLUSTERS,
            P_INTERNAL,
            Q_EXTERNAL,
            N_MULTIPLIER,
            LIFETIME_MULTIPLIER,
            STEP_SIZE,
        )?;
        validate_workload_label_mapping(&workload)?;
        Ok(workload)
    }

    fn expected_edges_per_node() -> NonZeroUsize {
        let expected_internal = ((N_PER_CLUSTER - 1) as f64 * P_INTERNAL).ceil() as usize;
        let expected_external = ((TOTAL_NODES - N_PER_CLUSTER) as f64 * Q_EXTERNAL).ceil() as usize;
        NonZeroUsize::new((expected_internal + expected_external).max(1))
            .expect("expected degree hint is non-zero")
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

    fn query_subset(live_nodes: &[usize], frac: f64) -> Result<Vec<usize>> {
        let query_len = (live_nodes.len() as f64 * frac) as usize;
        let query_len = query_len.max(NUM_CLUSTERS).min(live_nodes.len());
        let mut nodes = live_nodes.to_vec();
        let mut rng = rand::rngs::StdRng::seed_from_u64(WORKLOAD_SEED);
        for i in (1..nodes.len()).rev() {
            let j = rng.random_range(0..=i);
            nodes.swap(i, j);
        }
        nodes.truncate(query_len);
        nodes.sort_unstable();
        Ok(nodes)
    }

    fn true_labels(workload: &SbmDiffWorkload<f64>, nodes: &[usize]) -> Result<Vec<usize>> {
        nodes
            .iter()
            .map(|&node| {
                workload
                    .cluster_labels
                    .get(node)
                    .copied()
                    .ok_or_else(|| format!("node {node} had no planted cluster label").into())
            })
            .collect()
    }

    fn validate_workload_label_mapping(workload: &SbmDiffWorkload<f64>) -> Result<()> {
        for &node in &workload.nodes {
            let actual = workload
                .cluster_labels
                .get(node)
                .copied()
                .ok_or_else(|| format!("node {node} had no planted cluster label"))?;
            let expected = node / N_PER_CLUSTER;
            if expected >= NUM_CLUSTERS || actual != expected {
                return Err(format!(
                    "workload label mapping mismatch for node {node}: expected {expected}, got {actual}"
                )
                .into());
            }
        }
        Ok(())
    }

    fn validate_query_label_mapping(
        workload: &SbmDiffWorkload<f64>,
        nodes: &[usize],
        labels: &[usize],
    ) -> Result<()> {
        if nodes.len() != labels.len() {
            return Err(format!(
                "query mapping mismatch: {} nodes but {} labels",
                nodes.len(),
                labels.len()
            )
            .into());
        }

        for (&node, &label) in nodes.iter().zip(labels) {
            let workload_label = workload
                .cluster_labels
                .get(node)
                .copied()
                .ok_or_else(|| format!("node {node} had no planted cluster label"))?;
            let expected = node / N_PER_CLUSTER;
            if label != workload_label || label != expected {
                return Err(format!(
                    "query label mapping mismatch for node {node}: true_labels gave {label}, workload has {workload_label}, block convention gives {expected}"
                )
                .into());
            }
        }

        Ok(())
    }

    fn strict(value: f64) -> Result<Strict<f64>> {
        Strict::<f64>::new(value)
            .map_err(|err| format!("expected a positive finite scalar, got {value}: {err}").into())
    }

    fn trial_summaries_for_output(
        labels_true: &[usize],
        output: &PartitionOutput<usize, f64>,
    ) -> Vec<TrialSummary> {
        let PartitionOutput::Subset(trials) = output else {
            panic!("playground uses subset queries");
        };

        trials
            .iter()
            .map(|trial| TrialSummary {
                trial_index: trial.trial_index,
                ari: adjusted_rand_index(labels_true, &trial.labels),
                score: trial.scores.as_ref().unwrap().iter().sum::<f64>(),
                num_clusters: trial.num_clusters,
            })
            .collect()
    }

    fn format_trial_summary(summary: &TrialSummary) -> String {
        format!(
            "#{}  ari={:.6} score={:.6}, k={}",
            summary.trial_index, summary.ari, summary.score, summary.num_clusters
        )
    }
}
