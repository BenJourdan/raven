#[cfg(feature = "remote-playground")]
fn main() -> playground::Result<()> {
    playground::run()
}

#[cfg(not(feature = "remote-playground"))]
fn main() {
    eprintln!(
        "native_memgraph_playground runs Docker/Memgraph and Raven's Leiden clustering path; use:\n\
         cargo run -p raven-adapters --features remote-playground --bin native_memgraph_playground"
    );
}

#[cfg(feature = "remote-playground")]
mod playground {
    use std::{
        collections::{HashMap, HashSet},
        error::Error,
        fs::{File, create_dir_all},
        io::Write,
        num::NonZeroUsize,
        path::Path,
        thread,
        time::{Duration, Instant},
    };

    use indicatif::{ProgressBar, ProgressStyle};
    use neo4rs::{BoltMap, BoltString, BoltType, Graph, Row, query};
    use rand::{RngExt, SeedableRng};
    use raven_adapters::in_memory::{
        InMemoryUndirectedGraph,
        workloads::{SbmDiffWorkload, SbmEdgeOp, SbmUpdateBatch, prepare_diff_workload_sbm},
    };
    use raven_core::{
        DynamicClusteringAlg,
        alg::{DynamicClustering, ResizeQueryInfo},
        clustering::{LeidenConfig, leiden_community_detection_alg},
        metrics::adjusted_rand_index,
        types::{PartitionOutput, PartitionType, Strict, StrictCarrierOps, TrialOutputMode},
    };
    use rustc_hash::FxHashSet;
    use testcontainers::{
        ContainerAsync, GenericImage, ImageExt,
        core::{IntoContainerPort, WaitFor},
        runners::AsyncRunner,
    };

    pub type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync + 'static>>;

    const WORKLOAD_SEED: u64 = 42;
    const CORE_RNG_SEED: u64 = 42;
    const N_PER_CLUSTER: usize = 1024;
    const NUM_CLUSTERS: usize = 16;
    const TOTAL_NODES: usize = N_PER_CLUSTER * NUM_CLUSTERS;
    const P_INTERNAL: f64 = 0.5;
    const Q_EXTERNAL: f64 = 1.0 / TOTAL_NODES as f64;
    const N_MULTIPLIER: usize = 10;
    const LIFETIME_MULTIPLIER: f64 = 3.0;
    const STEP_SIZE: f64 = 0.1;
    const SIGMA: f64 = 1000.0;
    const ARITY: usize = 8;

    const NUM_TRIALS: usize = 1;
    const CORESET_SIZE: usize = 2048;
    const SAMPLING_SEEDS: usize = NUM_CLUSTERS * 8;
    const QUERY_FRAC: f64 = 0.01;
    const DEGREE_CACHE_THRESHOLD: usize = 4096;
    const MEMGRAPH_WRITE_CHUNK_SIZE: usize = 10_000;

    // Memgraph `community_detection.get` parameters:
    // https://memgraph.com/docs/advanced-algorithms/available-algorithms/community_detection
    const MEMGRAPH_COMMUNITY_DETECTION: MemgraphCommunityDetectionConfig =
        MemgraphCommunityDetectionConfig {
            weight_property: "weight",
            coloring: false,
            min_graph_shrink: 100_000,
            community_alg_threshold: 0.000001,
            coloring_alg_threshold: 0.01,
            num_threads: None,
        };

    const CSV_PATH: &str = "target/native_memgraph_playground/results.csv";

    #[derive(Clone, Copy, Debug)]
    struct MemgraphCommunityDetectionConfig {
        weight_property: &'static str,
        coloring: bool,
        min_graph_shrink: i64,
        community_alg_threshold: f64,
        coloring_alg_threshold: f64,
        num_threads: Option<i64>,
    }

    #[derive(Debug)]
    struct BatchResult {
        batch_idx: usize,
        time: i64,
        live_nodes: usize,
        query_nodes: usize,
        edge_ops: usize,
        memgraph_set_ops: usize,
        memgraph_delete_ops: usize,
        memgraph_dead_node_ops: usize,
        in_memory_graph_update_ms: f64,
        node_ops_flush_ms: f64,
        memgraph_update_ms: f64,
        memgraph_node_merge_ms: f64,
        memgraph_set_ms: f64,
        memgraph_delete_ms: f64,
        memgraph_dead_node_ms: f64,
        raven_core_update_ms: f64,
        raven_query_ms: f64,
        memgraph_community_detection_ms: f64,
        raven_winner_ari: f64,
        memgraph_ari: f64,
        raven_winner_score: f64,
        raven_winner_clusters: usize,
        memgraph_clusters: usize,
    }

    #[derive(Debug)]
    struct RavenWinner {
        ari: f64,
        score: f64,
        num_clusters: usize,
    }

    struct MemgraphFixture {
        graph: Graph,
        container: ContainerAsync<GenericImage>,
    }

    #[derive(Debug, Default)]
    struct MemgraphUpdateTiming {
        node_merge: Duration,
        set_edges: Duration,
        delete_edges: Duration,
        delete_dead_nodes: Duration,
        set_ops: usize,
        delete_ops: usize,
        dead_node_ops: usize,
    }

    pub fn run() -> Result<()> {
        let setup = ProgressBar::new_spinner()
            .with_style(ProgressStyle::with_template("{spinner:.cyan} {msg}")?);
        setup.enable_steady_tick(Duration::from_millis(100));

        setup.set_message("starting memgraph/memgraph-mage container...");
        let runtime = runtime();
        let MemgraphFixture { graph, container } = runtime.block_on(start_memgraph())?;

        setup.set_message("generating deterministic SBM workload...");
        let workload = workload()?;
        setup.finish_with_message(format!(
            "generated {} batches over {} nodes",
            workload.batches.len(),
            workload.nodes.len()
        ));

        let mut in_memory_graph = InMemoryUndirectedGraph::with_capacity(
            NonZeroUsize::new(TOTAL_NODES).expect("total node count is non-zero"),
            expected_edges_per_node(),
            NonZeroUsize::new(DEGREE_CACHE_THRESHOLD)
                .expect("degree rebuild threshold is non-zero"),
        );

        let alg = leiden_community_detection_alg(LeidenConfig {
            seed: Some(42),
            ..LeidenConfig::default()
        });
        let mut clustering = DynamicClustering::<ARITY, usize, f64>::new(alg)
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
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} updates ({eta_precise} remaining) - {msg}",
        )?;
        let pbar = ProgressBar::new(total_updates as u64).with_style(pbar_style);

        let total_started = Instant::now();
        let mut results = Vec::new();

        let mut total_in_memory_graph_update_us = 0u128;
        let mut total_node_ops_flush_us = 0u128;
        let mut total_memgraph_update_us = 0u128;
        let mut total_memgraph_node_merge_us = 0u128;
        let mut total_memgraph_set_us = 0u128;
        let mut total_memgraph_delete_us = 0u128;
        let mut total_memgraph_dead_node_us = 0u128;
        let mut total_raven_core_update_us = 0u128;
        let mut total_raven_query_us = 0u128;
        let mut total_memgraph_detection_us = 0u128;

        for (batch_idx, batch) in workload.batches.iter().enumerate() {
            pbar.set_message(format!(
                "batch {batch_idx}: replaying {} edge ops",
                batch.edge_ops.len()
            ));

            let in_memory_graph_update_started = Instant::now();
            batch.apply_to_graph(&mut in_memory_graph)?;
            let in_memory_graph_update_elapsed = in_memory_graph_update_started.elapsed();
            total_in_memory_graph_update_us += in_memory_graph_update_elapsed.as_micros();

            let node_ops_started = Instant::now();
            let node_ops = in_memory_graph.flush_node_ops();
            let node_ops_elapsed = node_ops_started.elapsed();
            total_node_ops_flush_us += node_ops_elapsed.as_micros();

            let memgraph_update_started = Instant::now();
            let memgraph_update =
                runtime.block_on(apply_batch_to_memgraph(&graph, batch, &pbar, batch_idx))?;
            let memgraph_update_elapsed = memgraph_update_started.elapsed();
            total_memgraph_update_us += memgraph_update_elapsed.as_micros();
            total_memgraph_node_merge_us += memgraph_update.node_merge.as_micros();
            total_memgraph_set_us += memgraph_update.set_edges.as_micros();
            total_memgraph_delete_us += memgraph_update.delete_edges.as_micros();
            total_memgraph_dead_node_us += memgraph_update.delete_dead_nodes.as_micros();

            let raven_core_update_started = Instant::now();
            clustering.apply_node_ops(&node_ops)?;
            let raven_core_update_elapsed = raven_core_update_started.elapsed();
            total_raven_core_update_us += raven_core_update_elapsed.as_micros();

            let live_nodes = live_nodes(&workload, &in_memory_graph);
            if live_nodes.len() < CORESET_SIZE {
                pbar.set_message(format!(
                    "batch {batch_idx}: time={}, live={}, skipped until live >= coreset_size",
                    batch.time,
                    live_nodes.len()
                ));
                continue;
            }

            let query_nodes = query_subset(&live_nodes, QUERY_FRAC)?;
            let true_labels = true_labels(&workload, &query_nodes)?;

            pbar.set_message(format!(
                "batch {batch_idx}: running in-memory Raven over {} nodes",
                query_nodes.len()
            ));
            let raven_query_started = Instant::now();
            let mut oracles = in_memory_graph.oracles(NUM_TRIALS);
            let mut oracle_refs = oracles.iter_mut().collect::<Vec<_>>();
            let raven_output = clustering.query(
                PartitionType::Subset(&query_nodes),
                TrialOutputMode::AllTrials,
                &mut oracle_refs,
            )?;
            let raven_query_elapsed = raven_query_started.elapsed();
            total_raven_query_us += raven_query_elapsed.as_micros();
            let raven_winner = raven_winner(&true_labels, &raven_output)?;

            pbar.set_message(format!(
                "batch {batch_idx}: running Memgraph community_detection.get()"
            ));
            let memgraph_detection_started = Instant::now();
            let memgraph_labels = runtime.block_on(memgraph_community_detection(
                &graph,
                MEMGRAPH_COMMUNITY_DETECTION,
            ))?;
            let memgraph_detection_elapsed = memgraph_detection_started.elapsed();
            total_memgraph_detection_us += memgraph_detection_elapsed.as_micros();

            let memgraph_subset_labels = labels_from_memgraph_map(&memgraph_labels, &query_nodes)?;
            let memgraph_clusters = memgraph_labels
                .values()
                .copied()
                .collect::<HashSet<_>>()
                .len();
            let memgraph_ari = adjusted_rand_index(&true_labels, &memgraph_subset_labels);

            let result = BatchResult {
                batch_idx,
                time: batch.time,
                live_nodes: live_nodes.len(),
                query_nodes: query_nodes.len(),
                edge_ops: batch.edge_ops.len(),
                memgraph_set_ops: memgraph_update.set_ops,
                memgraph_delete_ops: memgraph_update.delete_ops,
                memgraph_dead_node_ops: memgraph_update.dead_node_ops,
                in_memory_graph_update_ms: millis(in_memory_graph_update_elapsed),
                node_ops_flush_ms: millis(node_ops_elapsed),
                memgraph_update_ms: millis(memgraph_update_elapsed),
                memgraph_node_merge_ms: millis(memgraph_update.node_merge),
                memgraph_set_ms: millis(memgraph_update.set_edges),
                memgraph_delete_ms: millis(memgraph_update.delete_edges),
                memgraph_dead_node_ms: millis(memgraph_update.delete_dead_nodes),
                raven_core_update_ms: millis(raven_core_update_elapsed),
                raven_query_ms: millis(raven_query_elapsed),
                memgraph_community_detection_ms: millis(memgraph_detection_elapsed),
                raven_winner_ari: raven_winner.ari,
                memgraph_ari,
                raven_winner_score: raven_winner.score,
                raven_winner_clusters: raven_winner.num_clusters,
                memgraph_clusters,
            };

            pbar.println(format!(
                "batch {batch_idx}: time={}, live={}, query={}, raven={:.3}ms ari={:.4}, memgraph={:.3}ms ari={:.4}, memgraph_update={:.3}ms (set={:.3}, del={:.3})",
                result.time,
                result.live_nodes,
                result.query_nodes,
                result.raven_query_ms,
                result.raven_winner_ari,
                result.memgraph_community_detection_ms,
                result.memgraph_ari,
                result.memgraph_update_ms,
                result.memgraph_set_ms,
                result.memgraph_delete_ms
            ));
            results.push(result);
        }

        pbar.finish_with_message("done");
        write_csv(CSV_PATH, &results)?;

        println!("batches: {}", workload.batches.len());
        println!("nodes: {TOTAL_NODES} total");
        println!("queried batches: {}", results.len());
        println!("total elapsed: {:?}", total_started.elapsed());
        println!("Timing breakdown:");
        println!(
            "  in-memory graph updates: {:.3} seconds",
            total_in_memory_graph_update_us as f64 / 1_000_000.0
        );
        println!(
            "  node ops flush: {:.3} seconds",
            total_node_ops_flush_us as f64 / 1_000_000.0
        );
        println!(
            "  Memgraph updates: {:.3} seconds",
            total_memgraph_update_us as f64 / 1_000_000.0
        );
        println!(
            "    node merges: {:.3} seconds",
            total_memgraph_node_merge_us as f64 / 1_000_000.0
        );
        println!(
            "    edge upserts: {:.3} seconds",
            total_memgraph_set_us as f64 / 1_000_000.0
        );
        println!(
            "    edge deletes: {:.3} seconds",
            total_memgraph_delete_us as f64 / 1_000_000.0
        );
        println!(
            "    dead-node deletes: {:.3} seconds",
            total_memgraph_dead_node_us as f64 / 1_000_000.0
        );
        println!(
            "  Raven core updates: {:.3} seconds",
            total_raven_core_update_us as f64 / 1_000_000.0
        );
        println!(
            "  Raven in-memory queries: {:.3} seconds",
            total_raven_query_us as f64 / 1_000_000.0
        );
        println!(
            "  Memgraph community_detection: {:.3} seconds",
            total_memgraph_detection_us as f64 / 1_000_000.0
        );
        println!("wrote CSV: {CSV_PATH}");

        drop(graph);
        runtime.block_on(container.rm())?;

        Ok(())
    }

    async fn start_memgraph() -> Result<MemgraphFixture> {
        let container = GenericImage::new("memgraph/memgraph-mage", "latest")
            .with_exposed_port(7687.tcp())
            .with_wait_for(WaitFor::seconds(2))
            .with_cmd(["--storage-properties-on-edges=true"])
            .start()
            .await?;
        let port = container.get_host_port_ipv4(7687.tcp()).await?;
        let graph = Graph::new(format!("bolt://127.0.0.1:{port}"), "", "")?;

        wait_for_bolt(&graph).await?;
        create_memgraph_schema(&graph).await?;
        Ok(MemgraphFixture { graph, container })
    }

    async fn wait_for_bolt(graph: &Graph) -> Result<()> {
        let mut last_error = None;

        for _ in 0..30 {
            match graph.run(query("RETURN 1")).await {
                Ok(()) => return Ok(()),
                Err(err) => {
                    last_error = Some(err);
                    thread::sleep(Duration::from_millis(250));
                }
            }
        }

        Err(format!("Memgraph Bolt endpoint did not become ready: {last_error:?}").into())
    }

    async fn create_memgraph_schema(graph: &Graph) -> Result<()> {
        graph.run(query("CREATE INDEX ON :RavenNode(id)")).await?;
        graph
            .run(query("CREATE EDGE INDEX ON :RAVEN_EDGE(key)"))
            .await?;
        Ok(())
    }

    async fn apply_batch_to_memgraph(
        graph: &Graph,
        batch: &SbmUpdateBatch<f64>,
        pbar: &ProgressBar,
        batch_idx: usize,
    ) -> Result<MemgraphUpdateTiming> {
        let mut nodes_to_merge = FxHashSet::default();
        let mut sets = Vec::new();
        let mut deletes = Vec::new();

        for edge_op in &batch.edge_ops {
            match *edge_op {
                SbmEdgeOp::Set { u, v, weight } => {
                    let (u, v) = ordered_edge(u, v);
                    nodes_to_merge.insert(u);
                    nodes_to_merge.insert(v);
                    sets.push(edge_row(u, v, Some(weight.into_scalar()))?);
                }
                SbmEdgeOp::Delete { u, v } => {
                    let (u, v) = ordered_edge(u, v);
                    deletes.push(edge_row(u, v, None)?);
                }
            }
        }

        let nodes = nodes_to_merge.into_iter().map(node_id).collect::<Vec<_>>();
        let node_merge_started = Instant::now();
        run_chunked_memgraph_write(
            graph,
            pbar,
            batch_idx,
            "merging nodes",
            "nodes",
            nodes,
            false,
            r#"
            UNWIND $nodes AS id
            MERGE (:RavenNode {id: id})
            "#,
        )
        .await?;
        let node_merge = node_merge_started.elapsed();

        let set_ops = sets.len();
        let set_started = Instant::now();
        run_chunked_memgraph_write(
            graph,
            pbar,
            batch_idx,
            "setting edges",
            "sets",
            sets,
            true,
            r#"
            UNWIND $sets AS row
            MATCH (u:RavenNode {id: row.u})
            MATCH (v:RavenNode {id: row.v})
            MERGE (u)-[edge:RAVEN_EDGE {key: row.key}]->(v)
            SET edge.weight = row.weight
            "#,
        )
        .await?;
        let set_edges = set_started.elapsed();

        let delete_ops = deletes.len();
        let delete_started = Instant::now();
        run_chunked_memgraph_write(
            graph,
            pbar,
            batch_idx,
            "deleting edges",
            "deletes",
            deletes,
            true,
            r#"
            UNWIND $deletes AS row
            MATCH ()-[edge:RAVEN_EDGE {key: row.key}]->()
            DELETE edge
            "#,
        )
        .await?;
        let delete_edges = delete_started.elapsed();

        let dead_nodes = batch
            .node_ops
            .iter()
            .filter_map(|&(node, degree)| degree.is_none().then(|| node_id(node)))
            .collect::<Vec<_>>();
        let dead_node_ops = dead_nodes.len();
        let dead_node_started = Instant::now();
        run_chunked_memgraph_write(
            graph,
            pbar,
            batch_idx,
            "deleting inactive nodes",
            "nodes",
            dead_nodes,
            false,
            r#"
            UNWIND $nodes AS id
            MATCH (node:RavenNode {id: id})
            DELETE node
            "#,
        )
        .await?;
        let delete_dead_nodes = dead_node_started.elapsed();

        Ok(MemgraphUpdateTiming {
            node_merge,
            set_edges,
            delete_edges,
            delete_dead_nodes,
            set_ops,
            delete_ops,
            dead_node_ops,
        })
    }

    async fn run_chunked_memgraph_write(
        graph: &Graph,
        pbar: &ProgressBar,
        batch_idx: usize,
        phase: &str,
        param_name: &'static str,
        rows: Vec<BoltType>,
        count_as_edge_ops: bool,
        cypher: &'static str,
    ) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }

        let total_chunks = rows.len().div_ceil(MEMGRAPH_WRITE_CHUNK_SIZE);
        for (chunk_idx, chunk) in rows.chunks(MEMGRAPH_WRITE_CHUNK_SIZE).enumerate() {
            pbar.set_message(format!(
                "batch {batch_idx}: {phase} chunk {}/{} ({} rows)",
                chunk_idx + 1,
                total_chunks,
                chunk.len()
            ));
            graph
                .run(query(cypher).param(param_name, BoltType::from(chunk.to_vec())))
                .await?;
            if count_as_edge_ops {
                pbar.inc(chunk.len() as u64);
            }
        }

        Ok(())
    }

    async fn memgraph_community_detection(
        graph: &Graph,
        config: MemgraphCommunityDetectionConfig,
    ) -> Result<HashMap<usize, usize>> {
        let mut query_builder = if let Some(num_threads) = config.num_threads {
            query(
                r#"
                CALL community_detection.get(
                    $weight,
                    $coloring,
                    $min_graph_shrink,
                    $community_alg_threshold,
                    $coloring_alg_threshold,
                    $num_threads
                )
                YIELD node, community_id
                RETURN node.id AS id, community_id AS label
                "#,
            )
            .param("num_threads", BoltType::from(num_threads))
        } else {
            query(
                r#"
                CALL community_detection.get(
                    $weight,
                    $coloring,
                    $min_graph_shrink,
                    $community_alg_threshold,
                    $coloring_alg_threshold
                )
                YIELD node, community_id
                RETURN node.id AS id, community_id AS label
                "#,
            )
        };
        query_builder = query_builder
            .param("weight", BoltType::from(config.weight_property.to_string()))
            .param("coloring", BoltType::from(config.coloring))
            .param("min_graph_shrink", BoltType::from(config.min_graph_shrink))
            .param(
                "community_alg_threshold",
                BoltType::from(config.community_alg_threshold),
            )
            .param(
                "coloring_alg_threshold",
                BoltType::from(config.coloring_alg_threshold),
            );

        let mut stream = graph.execute_read(query_builder).await?;
        let mut labels = HashMap::new();

        while let Some(row) = stream.next().await? {
            let id = decode_usize_column(&row, "id")?;
            let label = decode_usize_column(&row, "label")?;
            labels.insert(id, label);
        }

        Ok(labels)
    }

    fn workload() -> Result<SbmDiffWorkload<f64>> {
        Ok(prepare_diff_workload_sbm::<f64>(
            WORKLOAD_SEED,
            N_PER_CLUSTER,
            NUM_CLUSTERS,
            P_INTERNAL,
            Q_EXTERNAL,
            N_MULTIPLIER,
            LIFETIME_MULTIPLIER,
            STEP_SIZE,
        )?)
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

    fn raven_winner(
        labels_true: &[usize],
        output: &PartitionOutput<usize, f64>,
    ) -> Result<RavenWinner> {
        let PartitionOutput::Subset(trials) = output else {
            return Err("playground uses subset queries".into());
        };

        trials
            .iter()
            .map(|trial| {
                let score = trial
                    .scores
                    .as_ref()
                    .ok_or("expected trial scores for AllTrials output")?
                    .iter()
                    .copied()
                    .sum::<f64>();
                Ok(RavenWinner {
                    ari: adjusted_rand_index(labels_true, &trial.labels),
                    score,
                    num_clusters: trial.num_clusters,
                })
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .min_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| "Raven returned no trials".into())
    }

    fn labels_from_memgraph_map(
        labels: &HashMap<usize, usize>,
        query_nodes: &[usize],
    ) -> Result<Vec<usize>> {
        query_nodes
            .iter()
            .map(|node| {
                labels.get(node).copied().ok_or_else(|| {
                    format!("Memgraph returned no community label for node {node}").into()
                })
            })
            .collect()
    }

    fn write_csv(path: &str, results: &[BatchResult]) -> Result<()> {
        let path = Path::new(path);
        if let Some(parent) = path.parent() {
            create_dir_all(parent)?;
        }
        let mut file = File::create(path)?;
        writeln!(
            file,
            "batch_idx,time,live_nodes,query_nodes,edge_ops,memgraph_set_ops,memgraph_delete_ops,memgraph_dead_node_ops,in_memory_graph_update_ms,node_ops_flush_ms,memgraph_update_ms,memgraph_node_merge_ms,memgraph_set_ms,memgraph_delete_ms,memgraph_dead_node_ms,raven_core_update_ms,raven_query_ms,memgraph_community_detection_ms,raven_winner_ari,memgraph_ari,raven_winner_score,raven_winner_clusters,memgraph_clusters"
        )?;

        for result in results {
            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.6},{:.6},{:.6},{},{}",
                result.batch_idx,
                result.time,
                result.live_nodes,
                result.query_nodes,
                result.edge_ops,
                result.memgraph_set_ops,
                result.memgraph_delete_ops,
                result.memgraph_dead_node_ops,
                result.in_memory_graph_update_ms,
                result.node_ops_flush_ms,
                result.memgraph_update_ms,
                result.memgraph_node_merge_ms,
                result.memgraph_set_ms,
                result.memgraph_delete_ms,
                result.memgraph_dead_node_ms,
                result.raven_core_update_ms,
                result.raven_query_ms,
                result.memgraph_community_detection_ms,
                result.raven_winner_ari,
                result.memgraph_ari,
                result.raven_winner_score,
                result.raven_winner_clusters,
                result.memgraph_clusters
            )?;
        }
        Ok(())
    }

    fn edge_row(u: usize, v: usize, weight: Option<f64>) -> Result<BoltType> {
        let mut value = HashMap::new();
        value.insert(BoltString::from("u"), BoltType::from(u as i64));
        value.insert(BoltString::from("v"), BoltType::from(v as i64));
        value.insert(BoltString::from("key"), BoltType::from(edge_key(u, v)?));
        if let Some(weight) = weight {
            value.insert(BoltString::from("weight"), BoltType::from(weight));
        }
        Ok(BoltType::Map(BoltMap { value }))
    }

    fn node_id(node: usize) -> BoltType {
        BoltType::from(node as i64)
    }

    fn edge_key(u: usize, v: usize) -> Result<i64> {
        let u = i64::try_from(u)?;
        let v = i64::try_from(v)?;
        let total_nodes = i64::try_from(TOTAL_NODES)?;
        u.checked_mul(total_nodes)
            .and_then(|base| base.checked_add(v))
            .ok_or_else(|| "edge key overflowed i64".into())
    }

    fn decode_usize_column(row: &Row, column: &'static str) -> Result<usize> {
        let value = row
            .get::<BoltType>(column)
            .map_err(|err| format!("failed to decode Memgraph column `{column}`: {err}"))?;
        let BoltType::Integer(value) = value else {
            return Err(format!("Memgraph column `{column}` was not an integer").into());
        };
        usize::try_from(value.value)
            .map_err(|err| format!("Memgraph column `{column}` did not fit usize: {err}").into())
    }

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .enable_time()
            .build()
            .unwrap()
    }

    fn strict(value: f64) -> Result<Strict<f64>> {
        Strict::<f64>::new(value)
            .map_err(|err| format!("expected a positive finite scalar, got {value}: {err}").into())
    }

    fn ordered_edge(u: usize, v: usize) -> (usize, usize) {
        if u <= v { (u, v) } else { (v, u) }
    }

    fn millis(duration: Duration) -> f64 {
        duration.as_secs_f64() * 1000.0
    }
}
