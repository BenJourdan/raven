use std::{num::NonZeroUsize, sync::Arc};

use raven_core::{
    DynamicClusteringAlg, GraphOracle,
    alg::DynamicClustering,
    types::{
        AlgType, Neighbourhoods, NodeIdentity, PartitionOutput, PartitionType, Strict,
        TrialOutputMode,
    },
};

use super::*;
use super::{index::NodeInterner, oracle::DenseMarker};
use crate::in_memory::workloads::{generate_sbm_commands, prepare_diff_workload_sbm};

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

fn test_identity_clustering() -> DynamicClustering<2, NodeIdentity, f64> {
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
    assert!(rows.row(0).unwrap().contains(&(2, strict(1.5))));
    assert!(rows.row(0).unwrap().contains(&(3, strict(2.5))));
    assert_eq!(rows.row(1).unwrap(), &[(1, strict(1.5))]);
}

#[test]
fn graph_oracle_returns_intersecting_adjacency_rows() {
    let mut graph = InMemoryUndirectedGraph::<usize, f64>::new();
    graph.update_edge(1, 2, Some(strict(1.0))).unwrap();
    graph.update_edge(1, 3, Some(strict(2.0))).unwrap();
    graph.update_edge(1, 4, Some(strict(3.0))).unwrap();

    let mut oracle = graph.oracle();
    let rows = oracle
        .graph_neighbourhoods_intersecting(&[1, 2], &[3, 4, 99])
        .unwrap();

    assert_eq!(rows.len(), 2);
    assert_eq!(rows.row(0).unwrap().len(), 2);
    assert!(rows.row(0).unwrap().contains(&(3, strict(2.0))));
    assert!(rows.row(0).unwrap().contains(&(4, strict(3.0))));
    assert!(rows.row(1).unwrap().is_empty());
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

    assert!(left_rows.row(0).unwrap().contains(&(2, strict(1.5))));
    assert!(left_rows.row(0).unwrap().contains(&(3, strict(2.5))));
    assert_eq!(right_rows.row(0).unwrap(), &[(1, strict(1.5))]);
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
    assert_eq!(rows.row(0).unwrap(), &[(2, strict(2.5))]);
    assert_eq!(rows.row(1).unwrap(), &[(1, strict(2.5))]);
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
    assert_eq!(rows.row(0).unwrap(), &[(3, strict(2.0))]);
    assert_eq!(rows.row(1).unwrap(), &[(1, strict(2.0))]);
}

#[test]
fn dense_oracle_matches_generic_node_identity_oracle() {
    let mut graph = InMemoryUndirectedGraph::<NodeIdentity, f64>::new();
    graph
        .update_edge(NodeIdentity(0), NodeIdentity(2), Some(strict(1.0)))
        .unwrap();
    graph
        .update_edge(NodeIdentity(0), NodeIdentity(100), Some(strict(2.0)))
        .unwrap();
    graph
        .update_edge(NodeIdentity(2), NodeIdentity(100), Some(strict(3.0)))
        .unwrap();

    let sources = [NodeIdentity(0), NodeIdentity(2)];
    let targets = [NodeIdentity(100)];
    let coreset = [NodeIdentity(0), NodeIdentity(100)];

    let generic_intersecting = {
        let mut oracle = graph.oracle();
        let rows = oracle
            .graph_neighbourhoods_intersecting(&sources, &targets)
            .unwrap();
        sorted_rows(rows)
    };
    let dense_intersecting = {
        let mut oracle = graph.dense_oracle();
        let rows = oracle
            .graph_neighbourhoods_intersecting(&sources, &targets)
            .unwrap();
        sorted_rows(rows)
    };
    assert_eq!(dense_intersecting, generic_intersecting);

    let generic_visited = {
        let mut oracle = graph.oracle();
        let mut rows = vec![Vec::new(); sources.len()];
        let visited_edges = oracle
            .visit_graph_neighbourhoods_intersecting(&sources, &targets, |row, node, weight| {
                rows[row].push((node, weight));
            })
            .unwrap();
        assert_eq!(
            visited_edges,
            generic_intersecting.iter().map(Vec::len).sum()
        );
        sort_row_vecs(rows)
    };
    assert_eq!(generic_visited, generic_intersecting);

    let dense_visited = {
        let mut oracle = graph.dense_oracle();
        let mut rows = vec![Vec::new(); sources.len()];
        let visited_edges = oracle
            .visit_graph_neighbourhoods_intersecting(&sources, &targets, |row, node, weight| {
                rows[row].push((node, weight));
            })
            .unwrap();
        assert_eq!(
            visited_edges,
            generic_intersecting.iter().map(Vec::len).sum()
        );
        sort_row_vecs(rows)
    };
    assert_eq!(dense_visited, generic_intersecting);

    let ordinal_targets = [NodeIdentity(100), NodeIdentity(0)];
    let expected_ordinal_rows = vec![vec![
        (1usize, NodeIdentity(0), strict(1.0)),
        (0usize, NodeIdentity(100), strict(3.0)),
    ]];

    let generic_ordinal_visited = {
        let mut oracle = graph.oracle();
        let mut rows = vec![Vec::new(); 1];
        let visited_edges = oracle
            .visit_graph_neighbourhoods_intersecting_with_target_indices(
                &[NodeIdentity(2)],
                &ordinal_targets,
                |row, target_idx, node, weight| {
                    rows[row].push((target_idx, node, weight));
                },
            )
            .unwrap();
        assert_eq!(visited_edges, 2);
        sort_ordinal_row_vecs(rows)
    };
    assert_eq!(generic_ordinal_visited, expected_ordinal_rows);

    let dense_ordinal_visited = {
        let mut oracle = graph.dense_oracle();
        let mut rows = vec![Vec::new(); 1];
        let visited_edges = oracle
            .visit_graph_neighbourhoods_intersecting_with_target_indices(
                &[NodeIdentity(2)],
                &ordinal_targets,
                |row, target_idx, node, weight| {
                    rows[row].push((target_idx, node, weight));
                },
            )
            .unwrap();
        assert_eq!(visited_edges, 2);
        sort_ordinal_row_vecs(rows)
    };
    assert_eq!(dense_ordinal_visited, expected_ordinal_rows);

    let generic_coreset = {
        let mut oracle = graph.oracle();
        let rows = oracle.coreset_neighbourhoods(&coreset).unwrap();
        sorted_rows(rows)
    };
    let dense_coreset = {
        let mut oracle = graph.dense_oracle();
        let rows = oracle.coreset_neighbourhoods(&coreset).unwrap();
        sorted_rows(rows)
    };
    assert_eq!(dense_coreset, generic_coreset);

    let expected_coreset_ordinals = vec![
        vec![(1usize, NodeIdentity(100), strict(2.0))],
        vec![(0usize, NodeIdentity(0), strict(2.0))],
    ];
    let generic_coreset_ordinals = {
        let mut oracle = graph.oracle();
        let mut rows = vec![Vec::new(); coreset.len()];
        let visited_edges = oracle
            .visit_coreset_neighbourhoods_with_target_indices(
                &coreset,
                |row, target_idx, node, weight| {
                    rows[row].push((target_idx, node, weight));
                },
            )
            .unwrap();
        assert_eq!(visited_edges, 2);
        sort_ordinal_row_vecs(rows)
    };
    assert_eq!(generic_coreset_ordinals, expected_coreset_ordinals);

    let dense_coreset_ordinals = {
        let mut oracle = graph.dense_oracle();
        let mut rows = vec![Vec::new(); coreset.len()];
        let visited_edges = oracle
            .visit_coreset_neighbourhoods_with_target_indices(
                &coreset,
                |row, target_idx, node, weight| {
                    rows[row].push((target_idx, node, weight));
                },
            )
            .unwrap();
        assert_eq!(visited_edges, 2);
        sort_ordinal_row_vecs(rows)
    };
    assert_eq!(dense_coreset_ordinals, expected_coreset_ordinals);

    let dense_empty = {
        let mut oracle = graph.dense_oracle();
        let rows = oracle
            .graph_neighbourhoods_intersecting(&sources, &[])
            .unwrap();
        sorted_rows(rows)
    };
    assert_eq!(dense_empty, vec![Vec::new(), Vec::new()]);
}

#[test]
fn dense_marker_epoch_overflow_clears_old_marks() {
    let mut marker = DenseMarker::default();
    marker.mark_all(&[NodeIdentity(2)]);
    assert!(marker.contains(NodeIdentity(2)));

    marker.set_epoch(u32::MAX);
    marker.mark_all(&[NodeIdentity(1)]);

    assert_eq!(marker.epoch(), 1);
    assert!(marker.contains(NodeIdentity(1)));
    assert!(!marker.contains(NodeIdentity(2)));
}

fn sorted_rows(
    rows: Neighbourhoods<'_, NodeIdentity, f64>,
) -> Vec<Vec<(NodeIdentity, Strict<f64>)>> {
    rows.iter()
        .map(|row| {
            let mut row = row.to_vec();
            row.sort_unstable_by_key(|(node, _)| *node);
            row
        })
        .collect()
}

fn sort_row_vecs(
    mut rows: Vec<Vec<(NodeIdentity, Strict<f64>)>>,
) -> Vec<Vec<(NodeIdentity, Strict<f64>)>> {
    for row in rows.iter_mut() {
        row.sort_unstable_by_key(|(node, _)| *node);
    }
    rows
}

fn sort_ordinal_row_vecs(
    mut rows: Vec<Vec<(usize, NodeIdentity, Strict<f64>)>>,
) -> Vec<Vec<(usize, NodeIdentity, Strict<f64>)>> {
    for row in rows.iter_mut() {
        row.sort_unstable_by_key(|(_, node, _)| *node);
    }
    rows
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
fn node_interner_reuses_released_slots_for_clone_only_ids() {
    let mut interner = NodeInterner::new();

    let alpha = interner.intern("alpha".to_string());
    let beta = interner.intern("beta".to_string());

    assert_eq!(interner.get(&"alpha".to_string()), Some(alpha));
    assert_eq!(interner.external(beta), Some(&"beta".to_string()));

    assert_eq!(interner.release(alpha), Some("alpha".to_string()));
    assert_eq!(interner.get(&"alpha".to_string()), None);

    let gamma = interner.intern("gamma".to_string());
    assert_eq!(gamma, alpha);
    assert_eq!(interner.external(gamma), Some(&"gamma".to_string()));
    assert_eq!(interner.get(&"beta".to_string()), Some(beta));
}

#[test]
fn in_memory_index_interns_strings_queries_and_reuses_deleted_slots() {
    let mut index = InMemoryIndex::<2, String, f64>::new(test_identity_clustering());

    for (u, v) in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f")] {
        index
            .update_edge(u.to_string(), v.to_string(), Some(strict(1.0)))
            .unwrap();
    }
    index.apply_pending_node_ops().unwrap();
    assert_eq!(index.live_node_count(), 6);

    let subset = vec!["f".to_string(), "a".to_string(), "c".to_string()];
    let output = index
        .query(PartitionType::Subset(&subset), TrialOutputMode::AllTrials)
        .unwrap();
    match output {
        PartitionOutput::Subset(trials) => {
            assert_eq!(trials.len(), 1);
            assert_eq!(trials[0].labels.len(), subset.len());
        }
        PartitionOutput::All(_, _) => panic!("expected subset output"),
    }

    let output = index
        .query(PartitionType::All, TrialOutputMode::AllTrials)
        .unwrap();
    match output {
        PartitionOutput::All(nodes, trials) => {
            assert_eq!(nodes.len(), 6);
            assert_eq!(trials.len(), 1);
            assert!(nodes.contains(&"a".to_string()));
            assert!(nodes.contains(&"f".to_string()));
        }
        PartitionOutput::Subset(_) => panic!("expected all-node output"),
    }

    let f_id = index.internal_id_for_test(&"f".to_string()).unwrap();
    index
        .update_edge("e".to_string(), "f".to_string(), None)
        .unwrap();
    assert_eq!(index.internal_id_for_test(&"f".to_string()), Some(f_id));
    assert!(!index.contains_node(&"f".to_string()));

    index.apply_pending_node_ops().unwrap();
    assert_eq!(index.internal_id_for_test(&"f".to_string()), None);
    assert_eq!(index.live_node_count(), 5);

    index
        .update_edge("g".to_string(), "h".to_string(), Some(strict(1.0)))
        .unwrap();
    let g_id = index.internal_id_for_test(&"g".to_string()).unwrap();
    assert_eq!(g_id, f_id);
}

#[test]
fn in_memory_index_delete_of_unknown_edge_does_not_intern_nodes() {
    let mut index = InMemoryIndex::<2, String, f64>::new(test_identity_clustering());

    let err = index
        .update_edge("missing-a".to_string(), "missing-b".to_string(), None)
        .unwrap_err();

    assert!(matches!(
        err,
        InMemoryIndexError::Graph(InMemoryGraphError::MissingEdge)
    ));
    assert_eq!(index.internal_id_for_test(&"missing-a".to_string()), None);
    assert_eq!(index.internal_id_for_test(&"missing-b".to_string()), None);
    assert_eq!(index.live_node_count(), 0);
}

#[test]
fn node_ops_buffer_tracks_unique_touched_nodes_until_flush() {
    let mut graph = InMemoryUndirectedGraph::<usize, f64>::with_capacity(
        NonZeroUsize::new(3).unwrap(),
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(2).unwrap(),
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
fn cached_degrees_track_replacements_and_deletions() {
    let mut graph = InMemoryUndirectedGraph::<usize, f64>::with_capacity(
        NonZeroUsize::new(4).unwrap(),
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(2).unwrap(),
    );

    graph.update_edge(1, 2, Some(strict(1.0))).unwrap();
    graph.update_edge(1, 2, Some(strict(2.5))).unwrap();
    graph.update_edge(1, 3, Some(strict(1.5))).unwrap();

    assert_eq!(graph.degree(1), Some(strict(4.0)));
    assert_eq!(graph.degree(2), Some(strict(2.5)));
    assert_eq!(graph.degree(3), Some(strict(1.5)));

    assert_eq!(
        graph.flush_node_ops(),
        vec![
            (1, Some(strict(4.0))),
            (2, Some(strict(2.5))),
            (3, Some(strict(1.5)))
        ]
    );

    graph.update_edge(1, 2, None).unwrap();
    assert_eq!(
        graph.flush_node_ops(),
        vec![(1, Some(strict(1.5))), (2, None)]
    );
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

#[test]
fn sbm_command_generation_is_deterministic() {
    let first = generate_sbm_commands(42, 8, 3, 0.5, 0.02, 1, 1.0).unwrap();
    let second = generate_sbm_commands(42, 8, 3, 0.5, 0.02, 1, 1.0).unwrap();

    assert_eq!(first, second);
    assert_eq!(first.nodes.len(), 24);
    assert_eq!(first.cluster_labels.len(), 24);
    assert_eq!(first.cluster_labels[0], 0);
    assert_eq!(first.cluster_labels[8], 1);
    assert_eq!(first.cluster_labels[16], 2);
    assert!(!first.operations.is_empty());
}

#[test]
fn prepared_sbm_workload_replays_into_in_memory_graph() {
    let workload = prepare_diff_workload_sbm::<f64>(42, 8, 3, 0.5, 0.02, 1, 1.0, 0.25).unwrap();

    assert_eq!(workload.nodes.len(), 24);
    assert_eq!(workload.cluster_labels.len(), 24);
    assert!(!workload.batches.is_empty());

    let mut graph = InMemoryUndirectedGraph::<usize, f64>::new();
    for batch in &workload.batches {
        let node_ops = batch.apply_to_graph_and_flush_node_ops(&mut graph).unwrap();
        assert_eq!(node_ops, batch.node_ops);
    }

    assert!(workload.nodes.iter().any(|node| graph.contains_node(*node)));
}
