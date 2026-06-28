use std::sync::Arc;

use rand::RngExt;

use super::common::*;
use crate::{
    DynamicClusteringAlg, GraphOracle,
    alg::{QueryTime, ResizeQueryInfo, RngMode, TrialWorkspace, coreset_impls::Coreset},
    error::{DynamicCoresetError, OracleError},
    types::{Neighbourhoods, PartitionOutput, PartitionType, TrialObjective, TrialOutputMode},
};

struct IntersectOnlyEmptyOracle {
    offsets: Vec<usize>,
}

impl IntersectOnlyEmptyOracle {
    fn new() -> Self {
        Self {
            offsets: Vec::new(),
        }
    }

    fn empty_rows<'a>(
        &'a mut self,
        sources: &[usize],
    ) -> Result<Neighbourhoods<'a, usize, f64>, OracleError<String>> {
        self.offsets.clear();
        self.offsets.resize(sources.len() + 1, 0);
        Ok(Neighbourhoods::new(&[], &self.offsets))
    }
}

impl GraphOracle<usize, f64, String> for IntersectOnlyEmptyOracle {
    fn graph_neighbourhoods<'a>(
        &'a mut self,
        _nodes: &[usize],
    ) -> Result<Neighbourhoods<'a, usize, f64>, OracleError<String>> {
        Err(OracleError::GraphError(
            "plain graph_neighbourhoods should not be called".to_string(),
        ))
    }

    fn graph_neighbourhoods_intersecting<'a>(
        &'a mut self,
        sources: &[usize],
        _targets: &[usize],
    ) -> Result<Neighbourhoods<'a, usize, f64>, OracleError<String>> {
        self.empty_rows(sources)
    }

    fn coreset_neighbourhoods<'a>(
        &'a mut self,
        nodes: &[usize],
    ) -> Result<Neighbourhoods<'a, usize, f64>, OracleError<String>> {
        self.empty_rows(nodes)
    }
}

#[test]
fn query_empty_tree_returns_no_data_error() {
    let mut clustering = test_clustering();
    let mut oracle = EmptyOracle::new();

    let err = <TestClustering as DynamicClusteringAlg<usize, f64>>::query::<_, String>(
        &mut clustering,
        PartitionType::All,
        TrialOutputMode::AllTrials,
        &mut [&mut oracle],
    )
    .unwrap_err();

    assert!(matches!(
        err.downcast_ref::<DynamicCoresetError>(),
        Some(DynamicCoresetError::NoData)
    ));
}

#[test]
fn query_rejects_zero_trials() {
    let mut clustering = query_ready_clustering(ResizeQueryInfo::Updates, 0);
    let mut oracles: [&mut EmptyOracle; 0] = [];

    let err = <TestClustering as DynamicClusteringAlg<usize, f64>>::query::<_, String>(
        &mut clustering,
        PartitionType::All,
        TrialOutputMode::AllTrials,
        &mut oracles,
    )
    .unwrap_err();

    assert!(err.to_string().contains("num_trials must be non-zero"));
}

#[test]
fn query_rejects_oracle_count_mismatch() {
    let mut clustering = query_ready_clustering(ResizeQueryInfo::Updates, 2);
    let mut oracle = EmptyOracle::new();

    let err = <TestClustering as DynamicClusteringAlg<usize, f64>>::query::<_, String>(
        &mut clustering,
        PartitionType::All,
        TrialOutputMode::AllTrials,
        &mut [&mut oracle],
    )
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("expected 2 oracles for 2 trials, but got 1")
    );
}

#[test]
fn query_rejects_stale_update_resize_scratch_lengths() {
    let mut clustering = query_ready_clustering(ResizeQueryInfo::Updates, 1);
    clustering.tree_data.query_time[0].h_s.pop();
    let mut oracle = EmptyOracle::new();

    let err = <TestClustering as DynamicClusteringAlg<usize, f64>>::query::<_, String>(
        &mut clustering,
        PartitionType::All,
        TrialOutputMode::AllTrials,
        &mut [&mut oracle],
    )
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("query time arrays are not the right length")
    );
}

#[test]
fn seeded_rng_is_stable_and_trial_specific() {
    let mut first_trial_a = RngMode::Seeded(42).rng_for_trial(0);
    let mut first_trial_b = RngMode::Seeded(42).rng_for_trial(0);
    let mut second_trial = RngMode::Seeded(42).rng_for_trial(1);

    let seq_a = (0..16)
        .map(|_| first_trial_a.random::<u64>())
        .collect::<Vec<_>>();
    let seq_b = (0..16)
        .map(|_| first_trial_b.random::<u64>())
        .collect::<Vec<_>>();
    let seq_c = (0..16)
        .map(|_| second_trial.random::<u64>())
        .collect::<Vec<_>>();

    assert_eq!(seq_a, seq_b);
    assert_ne!(seq_a, seq_c);
}

#[test]
fn query_rejects_wrong_cluster_label_count() {
    let mut clustering = query_ready_clustering(ResizeQueryInfo::Updates, 1);
    clustering.cluster_alg = Arc::new(|_, _| (Vec::new(), 1));
    let mut oracle = EmptyOracle::new();

    let err = <TestClustering as DynamicClusteringAlg<usize, f64>>::query::<_, String>(
        &mut clustering,
        PartitionType::All,
        TrialOutputMode::AllTrials,
        &mut [&mut oracle],
    )
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("cluster algorithm returned 0 labels")
    );
}

#[test]
fn full_graph_labelling_does_not_invent_projection_self_loops() {
    let mut clustering = test_clustering().with_num_clusters(2);
    <TestClustering as DynamicClusteringAlg<usize, f64>>::apply_node_ops(
        &mut clustering,
        &[(1, Some(strict(10.0))), (2, Some(strict(1.0)))],
    )
    .unwrap();

    let mut query_time = QueryTime::default();
    let node_one_idx = *clustering.node_to_tree_map.get(&1).unwrap();
    let node_two_idx = *clustering.node_to_tree_map.get(&2).unwrap();
    let workspace = TrialWorkspace::<2, _, _> {
        timestamp: clustering.timestamp,
        persistent: &clustering.tree_data.persistent,
        query_time: &mut query_time,
        tree_to_node_map: &clustering.tree_to_node_map,
        node_to_tree_map: &clustering.node_to_tree_map,
    };
    let coreset = Coreset {
        nodes: vec![1, 2],
        node_indices: vec![node_one_idx, node_two_idx],
        weights: vec![strict(1.0), strict(1.0)],
        coreset_labels: Some(vec![0, 1]),
    };
    let mut oracle = EmptyOracle::new();

    let (_nodes, labels, _scores) = workspace
        .rust_label_full_graph(&coreset, 2, &mut oracle, &[2], strict(1.0))
        .unwrap();

    assert_eq!(labels, vec![0]);
}

#[test]
fn full_graph_labelling_uses_intersecting_oracle_lookup() {
    let mut clustering = test_clustering().with_num_clusters(2);
    <TestClustering as DynamicClusteringAlg<usize, f64>>::apply_node_ops(
        &mut clustering,
        &[(1, Some(strict(10.0))), (2, Some(strict(1.0)))],
    )
    .unwrap();

    let mut query_time = QueryTime::default();
    let node_one_idx = *clustering.node_to_tree_map.get(&1).unwrap();
    let node_two_idx = *clustering.node_to_tree_map.get(&2).unwrap();
    let workspace = TrialWorkspace::<2, _, _> {
        timestamp: clustering.timestamp,
        persistent: &clustering.tree_data.persistent,
        query_time: &mut query_time,
        tree_to_node_map: &clustering.tree_to_node_map,
        node_to_tree_map: &clustering.node_to_tree_map,
    };
    let coreset = Coreset {
        nodes: vec![1, 2],
        node_indices: vec![node_one_idx, node_two_idx],
        weights: vec![strict(1.0), strict(1.0)],
        coreset_labels: Some(vec![0, 1]),
    };
    let mut oracle = IntersectOnlyEmptyOracle::new();

    workspace
        .rust_label_full_graph(&coreset, 2, &mut oracle, &[2], strict(1.0))
        .unwrap();
}

#[test]
fn query_runs_after_mixed_node_updates() {
    let mut clustering = test_clustering();
    clustering.coreset_size = 3;
    clustering.sampling_seeds = 2;
    clustering.cluster_alg = Arc::new(|graph, _| {
        let n = graph.symbolic().nrows();
        (vec![0; n], 1)
    });
    let mut oracle = EmptyOracle::new();

    <TestClustering as DynamicClusteringAlg<usize, f64>>::apply_node_ops(
        &mut clustering,
        &[
            (1, Some(strict(1.0))),
            (2, Some(strict(2.0))),
            (3, Some(strict(3.0))),
            (4, Some(strict(4.0))),
            (5, Some(strict(5.0))),
            (6, Some(strict(6.0))),
        ],
    )
    .unwrap();

    <TestClustering as DynamicClusteringAlg<usize, f64>>::apply_node_ops(
        &mut clustering,
        &[(2, None), (3, Some(strict(30.0))), (7, Some(strict(7.0)))],
    )
    .unwrap();

    let output = <TestClustering as DynamicClusteringAlg<usize, f64>>::query::<_, String>(
        &mut clustering,
        PartitionType::All,
        TrialOutputMode::AllTrials,
        &mut [&mut oracle],
    )
    .unwrap();

    match output {
        PartitionOutput::All(nodes, trial_parts) => {
            assert_eq!(trial_parts.len(), 1);
            let trial = &trial_parts[0];
            assert_eq!(trial.num_clusters, 1);
            assert_eq!(nodes.len(), clustering.num_leaves());
            assert_eq!(trial.labels.len(), nodes.len());
            assert!(trial.labels.iter().all(|label| *label == 0));
            assert!(!nodes.contains(&2));
            assert!(nodes.contains(&7));
        }
        PartitionOutput::Subset(_) => panic!("expected all-node partition output"),
    }

    assert_tree_consistent(&clustering);
}

#[test]
fn query_succeeds_with_update_and_query_time_resize_modes() {
    for resize_query_info in [ResizeQueryInfo::Updates, ResizeQueryInfo::Query] {
        let mut clustering = test_clustering()
            .with_resize_query_info(resize_query_info)
            .with_num_trials(2)
            .with_coreset_size(3)
            .with_sampling_seeds(2);
        use_zero_label_cluster_alg(&mut clustering);
        apply_six_node_fixture(&mut clustering);

        let tree_len = clustering.tree_data.persistent.size.len();
        for query_time in &clustering.tree_data.query_time {
            match resize_query_info {
                ResizeQueryInfo::Updates => {
                    assert_eq!(query_time.timestamp.len(), tree_len);
                    assert_eq!(query_time.f_delta.len(), tree_len);
                    assert_eq!(query_time.h_b.len(), tree_len);
                    assert_eq!(query_time.h_s.len(), tree_len);
                }
                ResizeQueryInfo::Query => {
                    assert!(query_time.timestamp.is_empty());
                    assert!(query_time.f_delta.is_empty());
                    assert!(query_time.h_b.is_empty());
                    assert!(query_time.h_s.is_empty());
                }
            }
        }

        let mut oracle_a = EmptyOracle::new();
        let mut oracle_b = EmptyOracle::new();
        let output = <TestClustering as DynamicClusteringAlg<usize, f64>>::query::<_, String>(
            &mut clustering,
            PartitionType::All,
            TrialOutputMode::AllTrials,
            &mut [&mut oracle_a, &mut oracle_b],
        )
        .unwrap();

        match output {
            PartitionOutput::All(nodes, trial_parts) => {
                assert_eq!(nodes.len(), clustering.num_leaves());
                assert_eq!(trial_parts.len(), 2);
            }
            PartitionOutput::Subset(_) => panic!("expected all-node partition output"),
        }

        for query_time in &clustering.tree_data.query_time {
            assert_eq!(query_time.timestamp.len(), tree_len);
            assert_eq!(query_time.f_delta.len(), tree_len);
            assert_eq!(query_time.h_b.len(), tree_len);
            assert_eq!(query_time.h_s.len(), tree_len);
        }
    }
}

#[test]
fn query_all_trials_returns_one_partition_per_trial() {
    let mut clustering = test_clustering()
        .with_num_trials(2)
        .with_coreset_size(3)
        .with_sampling_seeds(2);
    clustering.cluster_alg = Arc::new(|graph, _| {
        let n = graph.symbolic().nrows();
        (vec![0; n], 1)
    });

    apply_six_node_fixture(&mut clustering);

    let mut oracle_a = EmptyOracle::new();
    let mut oracle_b = EmptyOracle::new();
    let output = <TestClustering as DynamicClusteringAlg<usize, f64>>::query::<_, String>(
        &mut clustering,
        PartitionType::All,
        TrialOutputMode::AllTrials,
        &mut [&mut oracle_a, &mut oracle_b],
    )
    .unwrap();

    match output {
        PartitionOutput::All(nodes, trial_parts) => {
            assert_eq!(nodes.len(), clustering.num_leaves());
            assert_eq!(trial_parts.len(), 2);
            for (expected_idx, trial) in trial_parts.iter().enumerate() {
                assert_eq!(trial.trial_index, expected_idx);
                assert_eq!(trial.num_clusters, 1);
                assert_eq!(trial.labels.len(), nodes.len());
                assert!(
                    trial
                        .scores
                        .as_ref()
                        .is_some_and(|scores| scores.len() == nodes.len())
                );
            }
        }
        PartitionOutput::Subset(_) => panic!("expected all-node partition output"),
    }
}

#[test]
fn query_subset_all_trials_returns_one_partition_per_trial() {
    let mut clustering = query_ready_clustering(ResizeQueryInfo::Updates, 2);
    let subset = [6, 1, 4];
    let mut oracle_a = EmptyOracle::new();
    let mut oracle_b = EmptyOracle::new();

    let output = <TestClustering as DynamicClusteringAlg<usize, f64>>::query::<_, String>(
        &mut clustering,
        PartitionType::Subset(&subset),
        TrialOutputMode::AllTrials,
        &mut [&mut oracle_a, &mut oracle_b],
    )
    .unwrap();

    match output {
        PartitionOutput::Subset(trial_parts) => {
            assert_eq!(trial_parts.len(), 2);
            for (expected_idx, trial) in trial_parts.iter().enumerate() {
                assert_eq!(trial.trial_index, expected_idx);
                assert_eq!(trial.num_clusters, 1);
                assert_eq!(trial.labels.len(), subset.len());
                assert!(
                    trial
                        .scores
                        .as_ref()
                        .is_some_and(|scores| scores.len() == subset.len())
                );
            }
        }
        PartitionOutput::All(_, _) => panic!("expected subset partition output"),
    }
}

#[test]
fn query_winner_returns_single_partition_with_trial_index() {
    let mut clustering = test_clustering()
        .with_num_trials(2)
        .with_coreset_size(3)
        .with_sampling_seeds(2);
    clustering.cluster_alg = Arc::new(|graph, _| {
        let n = graph.symbolic().nrows();
        (vec![0; n], 1)
    });

    apply_six_node_fixture(&mut clustering);

    let mut oracle_a = EmptyOracle::new();
    let mut oracle_b = EmptyOracle::new();
    let output = <TestClustering as DynamicClusteringAlg<usize, f64>>::query::<_, String>(
        &mut clustering,
        PartitionType::All,
        TrialOutputMode::Winner(TrialObjective::KernelDistance),
        &mut [&mut oracle_a, &mut oracle_b],
    )
    .unwrap();

    match output {
        PartitionOutput::All(nodes, trial_parts) => {
            assert_eq!(nodes.len(), clustering.num_leaves());
            assert_eq!(trial_parts.len(), 1);
            let trial = &trial_parts[0];
            assert!(trial.trial_index < 2);
            assert_eq!(trial.num_clusters, 1);
            assert_eq!(trial.labels.len(), nodes.len());
            assert!(trial.scores.is_none());
        }
        PartitionOutput::Subset(_) => panic!("expected all-node partition output"),
    }
}

#[test]
fn query_subset_winner_returns_single_partition_with_trial_index() {
    let mut clustering = query_ready_clustering(ResizeQueryInfo::Updates, 2);
    let subset = [6, 1, 4];
    let mut oracle_a = EmptyOracle::new();
    let mut oracle_b = EmptyOracle::new();

    let output = <TestClustering as DynamicClusteringAlg<usize, f64>>::query::<_, String>(
        &mut clustering,
        PartitionType::Subset(&subset),
        TrialOutputMode::Winner(TrialObjective::KernelDistance),
        &mut [&mut oracle_a, &mut oracle_b],
    )
    .unwrap();

    match output {
        PartitionOutput::Subset(trial_parts) => {
            assert_eq!(trial_parts.len(), 1);
            let trial = &trial_parts[0];
            assert!(trial.trial_index < 2);
            assert_eq!(trial.num_clusters, 1);
            assert_eq!(trial.labels.len(), subset.len());
            assert!(trial.scores.is_none());
        }
        PartitionOutput::All(_, _) => panic!("expected subset partition output"),
    }
}
