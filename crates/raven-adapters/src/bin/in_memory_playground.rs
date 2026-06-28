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
    use std::{
        error::Error,
        num::NonZeroUsize,
        time::{Duration, Instant},
    };

    use rand::{RngExt, SeedableRng};
    use raven_adapters::in_memory::{
        InMemoryUndirectedGraph,
        workloads::{SbmDiffWorkload, prepare_diff_workload_sbm},
    };
    use raven_core::{
        DynamicClusteringAlg, GraphOracle,
        alg::{DynamicClustering, QueryTiming, ResizeQueryInfo},
        clustering::{LeidenConfig, leiden_community_detection_alg},
        error::OracleError,
        metrics::adjusted_rand_index,
        types::{Neighbourhoods, PartitionOutput, PartitionType, Strict, TrialOutputMode},
    };

    use indicatif::{ProgressBar, ProgressStyle};

    pub type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync + 'static>>;

    const WORKLOAD_SEED: u64 = 42;
    const CORE_RNG_SEED: u64 = 42;
    const N_PER_CLUSTER: usize = 1024;
    const NUM_CLUSTERS: usize = 256;
    const TOTAL_NODES: usize = N_PER_CLUSTER * NUM_CLUSTERS;
    const P_INTERNAL: f64 = 0.66;
    const Q_EXTERNAL: f64 = 1.0 / TOTAL_NODES as f64;
    const N_MULTIPLIER: usize = 1;
    const LIFETIME_MULTIPLIER: f64 = 1.0;
    const STEP_SIZE: f64 = 0.01;
    const SIGMA: f64 = 1000.0;
    const DEGREE_CACHE_THRESHOLD: usize = 4096;
    const ARITY: usize = 8;

    const NUM_TRIALS: usize = 1;
    const CORESET_SIZE: usize = 8192;
    const SAMPLING_SEEDS: usize = NUM_CLUSTERS * 4;
    const QUERY_FRAC: f64 = 0.1;

    #[derive(Debug)]
    struct TrialSummary {
        trial_index: usize,
        ari: f64,
        score: f64,
        num_clusters: usize,
    }

    #[derive(Debug, Default)]
    struct QueryTimingTotals {
        queries: usize,
        wall: Duration,
        setup: Duration,
        output: Duration,
        trial_total: Duration,
        trial_critical_path: Duration,
        extract_coreset: Duration,
        build_coreset_graph: Duration,
        cluster_coreset: Duration,
        label_partition: Duration,
    }

    #[derive(Debug, Default, Clone, Copy)]
    struct OracleTiming {
        graph_calls: usize,
        graph_sources: usize,
        graph_edges: usize,
        graph_time: Duration,
        intersecting_calls: usize,
        intersecting_sources: usize,
        intersecting_targets: usize,
        intersecting_edges: usize,
        intersecting_time: Duration,
        coreset_calls: usize,
        coreset_sources: usize,
        coreset_edges: usize,
        coreset_time: Duration,
    }

    #[derive(Debug)]
    struct TimedOracle<O> {
        inner: O,
        timing: OracleTiming,
    }

    impl<O> TimedOracle<O> {
        fn new(inner: O) -> Self {
            Self {
                inner,
                timing: OracleTiming::default(),
            }
        }
    }

    impl QueryTimingTotals {
        fn add(&mut self, timing: &QueryTiming) {
            self.queries += 1;
            self.wall += timing.total;
            self.setup += timing.setup;
            self.output += timing.output;

            let mut critical_path = Duration::ZERO;
            for trial in &timing.trials {
                self.trial_total += trial.total;
                self.extract_coreset += trial.extract_coreset;
                self.build_coreset_graph += trial.build_coreset_graph;
                self.cluster_coreset += trial.cluster_coreset;
                self.label_partition += trial.label_partition;
                critical_path = critical_path.max(trial.total);
            }
            self.trial_critical_path += critical_path;
        }
    }

    impl OracleTiming {
        fn add(&mut self, other: Self) {
            self.graph_calls += other.graph_calls;
            self.graph_sources += other.graph_sources;
            self.graph_edges += other.graph_edges;
            self.graph_time += other.graph_time;

            self.intersecting_calls += other.intersecting_calls;
            self.intersecting_sources += other.intersecting_sources;
            self.intersecting_targets += other.intersecting_targets;
            self.intersecting_edges += other.intersecting_edges;
            self.intersecting_time += other.intersecting_time;

            self.coreset_calls += other.coreset_calls;
            self.coreset_sources += other.coreset_sources;
            self.coreset_edges += other.coreset_edges;
            self.coreset_time += other.coreset_time;
        }

        fn total_time(self) -> Duration {
            self.graph_time + self.intersecting_time + self.coreset_time
        }
    }

    impl<V, T, E, O> GraphOracle<V, T, E> for TimedOracle<O>
    where
        O: GraphOracle<V, T, E>,
    {
        fn graph_neighbourhoods<'a>(
            &'a mut self,
            nodes: &[V],
        ) -> std::result::Result<Neighbourhoods<'a, V, T>, OracleError<E>> {
            let started = Instant::now();
            let result = self.inner.graph_neighbourhoods(nodes);
            let elapsed = started.elapsed();
            let edges = result
                .as_ref()
                .map(Neighbourhoods::data)
                .map_or(0, |data| data.len());

            self.timing.graph_calls += 1;
            self.timing.graph_sources += nodes.len();
            self.timing.graph_edges += edges;
            self.timing.graph_time += elapsed;

            result
        }

        fn graph_neighbourhoods_intersecting<'a>(
            &'a mut self,
            sources: &[V],
            targets: &[V],
        ) -> std::result::Result<Neighbourhoods<'a, V, T>, OracleError<E>> {
            let started = Instant::now();
            let result = self
                .inner
                .graph_neighbourhoods_intersecting(sources, targets);
            let elapsed = started.elapsed();
            let edges = result
                .as_ref()
                .map(Neighbourhoods::data)
                .map_or(0, |data| data.len());

            self.timing.intersecting_calls += 1;
            self.timing.intersecting_sources += sources.len();
            self.timing.intersecting_targets += targets.len();
            self.timing.intersecting_edges += edges;
            self.timing.intersecting_time += elapsed;

            result
        }

        fn coreset_neighbourhoods<'a>(
            &'a mut self,
            nodes: &[V],
        ) -> std::result::Result<Neighbourhoods<'a, V, T>, OracleError<E>> {
            let started = Instant::now();
            let result = self.inner.coreset_neighbourhoods(nodes);
            let elapsed = started.elapsed();
            let edges = result
                .as_ref()
                .map(Neighbourhoods::data)
                .map_or(0, |data| data.len());

            self.timing.coreset_calls += 1;
            self.timing.coreset_sources += nodes.len();
            self.timing.coreset_edges += edges;
            self.timing.coreset_time += elapsed;

            result
        }
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
        let mut clustering = DynamicClustering::<ARITY, usize, f64>::new(cluster_alg)
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
        let mut query_timing_totals = QueryTimingTotals::default();
        let mut oracle_timing_totals = OracleTiming::default();
        let mut queried_nodes_total = 0usize;

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
            queried_nodes_total += query_nodes.len();
            let true_labels = true_labels(&workload, &query_nodes)?;

            // time clustering queries:
            let query_started = Instant::now();

            let oracles = graph.oracles(NUM_TRIALS);
            let mut oracles = oracles
                .into_iter()
                .map(TimedOracle::new)
                .collect::<Vec<_>>();
            let mut oracle_refs = oracles.iter_mut().collect::<Vec<_>>();

            let output = clustering.query(
                PartitionType::Subset(&query_nodes),
                TrialOutputMode::AllTrials,
                &mut oracle_refs,
            )?;
            data_structure_query_time += query_started.elapsed().as_micros();
            if let Some(timing) = clustering.last_query_timing() {
                query_timing_totals.add(timing);
            }
            for oracle in &oracles {
                oracle_timing_totals.add(oracle.timing);
            }
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
        print_query_timing_breakdown(
            &query_timing_totals,
            &oracle_timing_totals,
            queried_nodes_total,
        );

        println!("ARI history (batch time, winner ARI):");
        println!(
            "{:?}",
            ari_history
                .iter()
                .map(|(time, ari)| (format!("{time}"), format!("{:.3}", ari.0)))
                .collect::<Vec<_>>()
        );

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

    fn print_query_timing_breakdown(
        query_timing: &QueryTimingTotals,
        oracle_timing: &OracleTiming,
        queried_nodes_total: usize,
    ) {
        if query_timing.queries == 0 {
            println!("Query timing breakdown: no queries ran");
            return;
        }

        let queries = query_timing.queries;
        let avg_query_nodes = queried_nodes_total as f64 / queries as f64;

        println!("Query timing breakdown:");
        println!(
            "  profiled query wall: {:.3} seconds ({:.3} ms/query, avg query nodes {:.1})",
            seconds(query_timing.wall),
            millis_per_query(query_timing.wall, queries),
            avg_query_nodes
        );
        println!(
            "  setup/validation: {:.3} seconds ({:.1}% of wall)",
            seconds(query_timing.setup),
            percent(query_timing.setup, query_timing.wall)
        );
        println!(
            "  trial critical path: {:.3} seconds ({:.1}% of wall)",
            seconds(query_timing.trial_critical_path),
            percent(query_timing.trial_critical_path, query_timing.wall)
        );
        println!(
            "  output shaping: {:.3} seconds ({:.1}% of wall)",
            seconds(query_timing.output),
            percent(query_timing.output, query_timing.wall)
        );
        println!(
            "  summed trial work: {:.3} seconds ({:.3} ms/query)",
            seconds(query_timing.trial_total),
            millis_per_query(query_timing.trial_total, queries)
        );
        println!(
            "    coreset extraction: {:.3} seconds ({:.1}% of trial work)",
            seconds(query_timing.extract_coreset),
            percent(query_timing.extract_coreset, query_timing.trial_total)
        );
        println!(
            "    coreset graph build: {:.3} seconds ({:.1}% of trial work)",
            seconds(query_timing.build_coreset_graph),
            percent(query_timing.build_coreset_graph, query_timing.trial_total)
        );
        println!(
            "    coreset clustering: {:.3} seconds ({:.1}% of trial work)",
            seconds(query_timing.cluster_coreset),
            percent(query_timing.cluster_coreset, query_timing.trial_total)
        );
        println!(
            "    partition labelling: {:.3} seconds ({:.1}% of trial work)",
            seconds(query_timing.label_partition),
            percent(query_timing.label_partition, query_timing.trial_total)
        );

        let oracle_total = oracle_timing.total_time();
        println!(
            "  oracle calls: {:.3} seconds ({:.1}% of trial work)",
            seconds(oracle_total),
            percent(oracle_total, query_timing.trial_total)
        );
        print_oracle_row(
            "graph_neighbourhoods",
            oracle_timing.graph_calls,
            oracle_timing.graph_sources,
            None,
            oracle_timing.graph_edges,
            oracle_timing.graph_time,
        );
        print_oracle_row(
            "graph_neighbourhoods_intersecting",
            oracle_timing.intersecting_calls,
            oracle_timing.intersecting_sources,
            Some(oracle_timing.intersecting_targets),
            oracle_timing.intersecting_edges,
            oracle_timing.intersecting_time,
        );
        print_oracle_row(
            "coreset_neighbourhoods",
            oracle_timing.coreset_calls,
            oracle_timing.coreset_sources,
            None,
            oracle_timing.coreset_edges,
            oracle_timing.coreset_time,
        );
    }

    fn print_oracle_row(
        name: &str,
        calls: usize,
        sources: usize,
        targets: Option<usize>,
        edges: usize,
        duration: Duration,
    ) {
        match targets {
            Some(targets) => println!(
                "    {name}: {:.3}s, calls={}, sources={}, targets={}, returned_edges={}",
                seconds(duration),
                calls,
                sources,
                targets,
                edges
            ),
            None => println!(
                "    {name}: {:.3}s, calls={}, sources={}, returned_edges={}",
                seconds(duration),
                calls,
                sources,
                edges
            ),
        }
    }

    fn seconds(duration: Duration) -> f64 {
        duration.as_secs_f64()
    }

    fn millis_per_query(duration: Duration, queries: usize) -> f64 {
        duration.as_secs_f64() * 1_000.0 / queries.max(1) as f64
    }

    fn percent(part: Duration, total: Duration) -> f64 {
        let total = total.as_secs_f64();
        if total == 0.0 {
            0.0
        } else {
            part.as_secs_f64() * 100.0 / total
        }
    }
}
