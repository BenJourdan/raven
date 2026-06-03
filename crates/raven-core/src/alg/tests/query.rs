use std::sync::Arc;

use super::common::*;
use crate::{
    DynamicClusteringAlg,
    alg::ResizeQueryInfo,
    error::DynamicCoresetError,
    types::{PartitionOutput, PartitionType, TrialObjective, TrialOutputMode},
};

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
