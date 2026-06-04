use std::{num::NonZeroUsize, sync::Arc};

use raven_core::{
    DynamicClusteringAlg, GraphOracle,
    alg::DynamicClustering,
    types::{AlgType, PartitionOutput, PartitionType, Strict, TrialOutputMode},
};

use super::*;

type TestClustering = DynamicClustering<2, usize, f64>;

fn strict(value: f64) -> Strict<f64> {
    Strict::<f64>::new(value).unwrap()
}

fn test_clustering() -> TestClustering {
    let cluster_alg: AlgType<f64> = Arc::new(|graph, _| {
        let n = graph.symbolic().nrows();
        (vec![0; n], 1)
    });

    DynamicClustering::new(cluster_alg)
        .with_sigma(strict(1.0))
        .with_num_trials(1)
        .with_coreset_size(3)
        .with_sampling_seeds(2)
        .with_num_clusters(1)
        .with_prop_name("w")
}

#[test]
fn graph_oracle_returns_full_adjacency_rows() {
    let mut graph = InMemoryUndirectedGraph::<usize, f64>::new();
    graph.update_edge(1, 2, Some(strict(1.5))).unwrap();
    graph.update_edge(1, 3, Some(strict(2.5))).unwrap();

    let mut oracle = graph.oracle();
    let rows = oracle.graph_neighbourhoods(&[1, 2]).unwrap();

    assert_eq!(rows.len(), 2);
    assert!(rows[0].contains(&(2, strict(1.5))));
    assert!(rows[0].contains(&(3, strict(2.5))));
    assert_eq!(rows[1], &[(1, strict(1.5))]);
}

#[test]
fn graph_lends_independent_oracle_handles() {
    let mut graph = InMemoryUndirectedGraph::<usize, f64>::new();
    graph.update_edge(1, 2, Some(strict(1.5))).unwrap();
    graph.update_edge(1, 3, Some(strict(2.5))).unwrap();

    let mut oracles = graph.oracles(2);
    let (left, right) = oracles.split_at_mut(1);

    let left_rows = left[0].graph_neighbourhoods(&[1]).unwrap();
    let right_rows = right[0].graph_neighbourhoods(&[2]).unwrap();

    assert!(left_rows[0].contains(&(2, strict(1.5))));
    assert!(left_rows[0].contains(&(3, strict(2.5))));
    assert_eq!(right_rows[0], &[(1, strict(1.5))]);
}

#[test]
fn reversed_edge_updates_the_same_undirected_relationship() {
    let mut graph = InMemoryUndirectedGraph::<usize, f64>::new();
    graph.update_edge(1, 2, Some(strict(1.5))).unwrap();
    graph.update_edge(2, 1, Some(strict(2.5))).unwrap();

    assert_eq!(graph.degree(1), Some(strict(2.5)));
    assert_eq!(graph.degree(2), Some(strict(2.5)));

    let mut oracle = graph.oracle();
    let rows = oracle.graph_neighbourhoods(&[1, 2]).unwrap();
    assert_eq!(rows[0], &[(2, strict(2.5))]);
    assert_eq!(rows[1], &[(1, strict(2.5))]);
}

#[test]
fn coreset_oracle_filters_to_input_batch() {
    let mut graph = InMemoryUndirectedGraph::<usize, f64>::new();
    graph.update_edge(1, 2, Some(strict(1.0))).unwrap();
    graph.update_edge(1, 3, Some(strict(2.0))).unwrap();
    graph.update_edge(2, 3, Some(strict(3.0))).unwrap();

    let mut oracle = graph.oracle();
    let rows = oracle.coreset_neighbourhoods(&[1, 3]).unwrap();

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0], &[(3, strict(2.0))]);
    assert_eq!(rows[1], &[(1, strict(2.0))]);
}

#[test]
fn flush_node_ops_reports_updated_and_deleted_nodes() {
    let mut graph = InMemoryUndirectedGraph::<usize, f64>::new();
    graph.update_edge(1, 2, Some(strict(4.0))).unwrap();

    assert_eq!(graph.degree(1), Some(strict(4.0)));

    let initial_ops = graph.flush_node_ops();
    assert_eq!(
        initial_ops,
        vec![(1, Some(strict(4.0))), (2, Some(strict(4.0)))]
    );
    assert!(graph.node_ops_buffer_is_empty());

    graph.update_edge(1, 2, None).unwrap();

    assert_eq!(graph.flush_node_ops(), vec![(1, None), (2, None)]);
}

#[test]
fn node_ops_buffer_tracks_unique_touched_nodes_until_flush() {
    let mut graph = InMemoryUndirectedGraph::<usize, f64>::with_node_ops_capacity(
        NonZeroUsize::new(3).unwrap(),
    );

    graph.update_edge(1, 2, Some(strict(1.0))).unwrap();
    assert_eq!(graph.node_ops_buffer_len(), 2);
    assert!(!graph.node_ops_buffer_is_full());

    graph.update_edge(2, 3, Some(strict(1.0))).unwrap();
    assert_eq!(graph.node_ops_buffer_len(), 3);
    assert!(graph.node_ops_buffer_is_full());

    let ops = graph.flush_node_ops();
    assert_eq!(ops.len(), 3);
    assert!(graph.node_ops_buffer_is_empty());
}

#[test]
fn graph_updates_flush_into_core_queries() {
    let mut graph = InMemoryUndirectedGraph::<usize, f64>::new();
    graph.update_edge(1, 2, Some(strict(1.0))).unwrap();
    graph.update_edge(2, 3, Some(strict(1.0))).unwrap();
    graph.update_edge(3, 4, Some(strict(1.0))).unwrap();
    graph.update_edge(4, 5, Some(strict(1.0))).unwrap();
    graph.update_edge(5, 6, Some(strict(1.0))).unwrap();

    let mut clustering = test_clustering();
    let initial_ops = graph.flush_node_ops();
    clustering.apply_node_ops(&initial_ops).unwrap();

    let output = {
        let mut oracle = graph.oracle();
        let mut oracles = [&mut oracle];
        clustering
            .query(PartitionType::All, TrialOutputMode::AllTrials, &mut oracles)
            .unwrap()
    };
    match output {
        PartitionOutput::All(nodes, trial_parts) => {
            assert_eq!(trial_parts.len(), 1);
            let trial = &trial_parts[0];
            assert_eq!(trial.num_clusters, 1);
            assert_eq!(nodes.len(), 6);
            assert_eq!(trial.labels.len(), nodes.len());
            assert!(nodes.contains(&1));
            assert!(nodes.contains(&6));
        }
        PartitionOutput::Subset(_) => panic!("expected all-node query output"),
    }

    graph.update_edge(5, 6, None).unwrap();
    graph.update_edge(4, 7, Some(strict(2.0))).unwrap();

    let update_ops = graph.flush_node_ops();
    clustering.apply_node_ops(&update_ops).unwrap();

    let output = {
        let mut oracle = graph.oracle();
        let mut oracles = [&mut oracle];
        clustering
            .query(PartitionType::All, TrialOutputMode::AllTrials, &mut oracles)
            .unwrap()
    };
    match output {
        PartitionOutput::All(nodes, trial_parts) => {
            assert_eq!(trial_parts.len(), 1);
            let trial = &trial_parts[0];
            assert_eq!(trial.num_clusters, 1);
            assert_eq!(nodes.len(), 6);
            assert_eq!(trial.labels.len(), nodes.len());
            assert!(nodes.contains(&7));
            assert!(!nodes.contains(&6));
        }
        PartitionOutput::Subset(_) => panic!("expected all-node query output"),
    }
}

#[test]
fn graph_lends_oracles_to_parallel_core_trials() {
    let mut graph = InMemoryUndirectedGraph::<usize, f64>::new();
    graph.update_edge(1, 2, Some(strict(1.0))).unwrap();
    graph.update_edge(2, 3, Some(strict(1.0))).unwrap();
    graph.update_edge(3, 4, Some(strict(1.0))).unwrap();
    graph.update_edge(4, 5, Some(strict(1.0))).unwrap();
    graph.update_edge(5, 6, Some(strict(1.0))).unwrap();

    let mut clustering = test_clustering().with_num_trials(2);
    clustering.apply_node_ops(&graph.flush_node_ops()).unwrap();

    let mut oracle_handles = graph.oracles(2);
    let mut oracle_refs = oracle_handles.iter_mut().collect::<Vec<_>>();
    let output = clustering
        .query(
            PartitionType::All,
            TrialOutputMode::AllTrials,
            &mut oracle_refs,
        )
        .unwrap();

    match output {
        PartitionOutput::All(nodes, trial_parts) => {
            assert_eq!(nodes.len(), 6);
            assert_eq!(trial_parts.len(), 2);
            assert!(trial_parts.iter().all(|trial| trial.labels.len() == 6));
        }
        PartitionOutput::Subset(_) => panic!("expected all-node query output"),
    }
}
