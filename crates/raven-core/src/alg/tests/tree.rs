use std::num::NonZeroUsize;

use rustc_hash::FxHashSet;

use super::common::*;
use crate::{
    alg::DynamicClustering,
    types::{NodeDegree, TreeIndex},
    DynamicClusteringAlg,
};

#[test]
fn layout_helpers_match_d_ary_leaf_math() {
    type D4 = DynamicClustering<4, usize, f64>;

    // I(n) = ceil((n - 1) / (d - 1)) for n > 1.
    assert_eq!(D4::internal_count_for_leaves(0), 0);
    assert_eq!(D4::internal_count_for_leaves(1), 0);
    assert_eq!(D4::internal_count_for_leaves(2), 1);
    assert_eq!(D4::internal_count_for_leaves(4), 1);
    assert_eq!(D4::internal_count_for_leaves(5), 2);
    assert_eq!(D4::internal_count_for_leaves(7), 2);
    assert_eq!(D4::internal_count_for_leaves(8), 3);

    assert_eq!(D4::total_count_for_leaves(0), 0);
    assert_eq!(D4::total_count_for_leaves(1), 1);
    assert_eq!(D4::total_count_for_leaves(5), 7);
    assert_eq!(D4::leaf_start_for_leaves(5), 2);
    assert_eq!(D4::leaf_range_for_leaves(5), 2..7);
}

#[test]
fn binary_layout_degenerates_to_standard_heap_shape() {
    type D2 = DynamicClustering<2, usize, f64>;

    for leaves in 1..12 {
        assert_eq!(D2::internal_count_for_leaves(leaves), leaves - 1);
        assert_eq!(D2::total_count_for_leaves(leaves), 2 * leaves - 1);
        assert_eq!(D2::leaf_start_for_leaves(leaves), leaves - 1);
    }
}

#[test]
fn insert_big_height_jump_clears_pending_touched_indices() {
    let mut clustering = test_clustering();
    let mut touched = FxHashSet::default();

    clustering
        .insert_fresh_nodes(&[(1, strict(1.0))], &mut touched)
        .unwrap();
    touched.insert(TreeIndex(0));

    clustering
        .insert_fresh_nodes(&[(2, strict(2.0))], &mut touched)
        .unwrap();

    assert!(touched.is_empty());
    assert_eq!(
        clustering.tree_data.persistent.volume[TreeIndex(0)],
        volume(3.0)
    );
    assert_eq!(
        clustering.tree_data.persistent.size[TreeIndex(0)],
        NonZeroUsize::new(2).unwrap()
    );
    assert_eq!(clustering.node_to_tree_map.get(&1), Some(&TreeIndex(1)));
    assert_eq!(clustering.node_to_tree_map.get(&2), Some(&TreeIndex(2)));
    assert_eq!(
        clustering.degrees.get_priority(&1),
        Some(&NodeDegree::new(strict(1.0)))
    );
    assert_eq!(
        clustering.degrees.get_priority(&2),
        Some(&NodeDegree::new(strict(2.0)))
    );
}

#[test]
fn insert_small_height_change_removes_touched_promoted_leaves_only() {
    let mut clustering = test_clustering();
    let mut touched = FxHashSet::default();

    clustering
        .insert_fresh_nodes(&[(1, strict(1.0)), (2, strict(2.0))], &mut touched)
        .unwrap();

    touched.insert(TreeIndex(1));
    touched.insert(TreeIndex(2));

    clustering
        .insert_fresh_nodes(&[(3, strict(3.0))], &mut touched)
        .unwrap();

    assert!(!touched.contains(&TreeIndex(1)));
    assert!(touched.contains(&TreeIndex(2)));
    assert_eq!(touched.len(), 1);

    assert_eq!(clustering.node_to_tree_map.get(&1), Some(&TreeIndex(3)));
    assert_eq!(clustering.node_to_tree_map.get(&2), Some(&TreeIndex(2)));
    assert_eq!(clustering.node_to_tree_map.get(&3), Some(&TreeIndex(4)));
    assert_eq!(
        clustering.tree_data.persistent.volume[TreeIndex(0)],
        volume(6.0)
    );
    assert_eq!(
        clustering.tree_data.persistent.size[TreeIndex(0)],
        NonZeroUsize::new(3).unwrap()
    );
}

#[test]
fn delete_tail_leaf_promotes_live_tail_source() {
    let mut clustering = test_clustering();
    let mut touched = FxHashSet::default();

    clustering
        .insert_fresh_nodes(
            &[(1, strict(1.0)), (2, strict(2.0)), (3, strict(3.0))],
            &mut touched,
        )
        .unwrap();

    touched.insert(TreeIndex(2));
    touched.insert(TreeIndex(3));

    clustering.delete_nodes_compact(&[3], &mut touched).unwrap();
    apply_size_volume_updates(&mut clustering, &touched);

    assert_eq!(clustering.tree_data.persistent.volume.len(), 3);
    assert_eq!(clustering.node_to_tree_map.get(&2), Some(&TreeIndex(1)));
    assert_eq!(clustering.node_to_tree_map.get(&1), Some(&TreeIndex(2)));
    assert!(!clustering.node_to_tree_map.contains_key(&3));

    assert_eq!(touched, FxHashSet::from_iter([TreeIndex(2)]));
    assert_eq!(
        clustering.tree_data.persistent.volume[TreeIndex(0)],
        volume(3.0)
    );
    assert_eq!(
        clustering.tree_data.persistent.size[TreeIndex(0)],
        NonZeroUsize::new(2).unwrap()
    );
}

#[test]
fn delete_interior_leaf_fills_hole_from_tail_source() {
    let mut clustering = test_clustering();
    let mut touched = FxHashSet::default();

    clustering
        .insert_fresh_nodes(
            &[(1, strict(1.0)), (2, strict(2.0)), (3, strict(3.0))],
            &mut touched,
        )
        .unwrap();

    clustering.delete_nodes_compact(&[1], &mut touched).unwrap();
    apply_size_volume_updates(&mut clustering, &touched);

    assert_eq!(clustering.tree_data.persistent.volume.len(), 3);
    assert_eq!(clustering.node_to_tree_map.get(&2), Some(&TreeIndex(1)));
    assert_eq!(clustering.node_to_tree_map.get(&3), Some(&TreeIndex(2)));
    assert!(!clustering.node_to_tree_map.contains_key(&1));

    assert_eq!(touched, FxHashSet::from_iter([TreeIndex(2)]));
    assert_eq!(
        clustering.tree_data.persistent.volume[TreeIndex(0)],
        volume(5.0)
    );
    assert_eq!(
        clustering.tree_data.persistent.size[TreeIndex(0)],
        NonZeroUsize::new(2).unwrap()
    );
}

#[test]
fn delete_to_single_leaf_moves_survivor_to_root() {
    let mut clustering = test_clustering();
    let mut touched = FxHashSet::default();

    clustering
        .insert_fresh_nodes(
            &[
                (1, strict(1.0)),
                (2, strict(2.0)),
                (3, strict(3.0)),
                (4, strict(4.0)),
                (5, strict(5.0)),
            ],
            &mut touched,
        )
        .unwrap();

    clustering
        .delete_nodes_compact(&[1, 2, 3, 4], &mut touched)
        .unwrap();

    assert!(touched.is_empty());
    assert_eq!(clustering.tree_data.persistent.volume.len(), 1);
    assert_eq!(clustering.node_to_tree_map.get(&5), Some(&TreeIndex(0)));
    assert_eq!(clustering.tree_to_node_map.get(&TreeIndex(0)), Some(&5));
    assert_eq!(
        clustering.tree_data.persistent.volume[TreeIndex(0)],
        volume(5.0)
    );
    assert_eq!(
        clustering.tree_data.persistent.size[TreeIndex(0)],
        NonZeroUsize::new(1).unwrap()
    );
}

#[test]
fn delete_all_nodes_clears_tree() {
    let mut clustering = test_clustering();
    let mut touched = FxHashSet::default();

    clustering
        .insert_fresh_nodes(&[(1, strict(1.0)), (2, strict(2.0))], &mut touched)
        .unwrap();
    touched.insert(TreeIndex(1));

    clustering
        .delete_nodes_compact(&[1, 2], &mut touched)
        .unwrap();

    assert!(touched.is_empty());
    assert!(clustering.node_to_tree_map.is_empty());
    assert!(clustering.tree_to_node_map.is_empty());
    assert!(clustering.degrees.is_empty());
    assert!(clustering.tree_data.persistent.volume.is_empty());
    assert!(clustering.tree_data.persistent.size.is_empty());
}

#[test]
fn apply_node_ops_handles_mixed_delete_insert_modify_batch() {
    let mut clustering = test_clustering();

    <TestClustering as DynamicClusteringAlg<usize, f64>>::apply_node_ops(
        &mut clustering,
        &[
            (1, Some(strict(1.0))),
            (2, Some(strict(2.0))),
            (3, Some(strict(3.0))),
            (4, Some(strict(4.0))),
            (5, Some(strict(5.0))),
        ],
    )
    .unwrap();
    assert_tree_consistent(&clustering);

    <TestClustering as DynamicClusteringAlg<usize, f64>>::apply_node_ops(
        &mut clustering,
        &[
            (2, None),
            (3, Some(strict(30.0))),
            (4, None),
            (6, Some(strict(6.0))),
            (7, Some(strict(7.0))),
        ],
    )
    .unwrap();

    assert_tree_consistent(&clustering);
    assert_eq!(clustering.num_leaves(), 5);
    assert!(!clustering.node_to_tree_map.contains_key(&2));
    assert!(!clustering.node_to_tree_map.contains_key(&4));

    for (node, degree) in [(1, 1.0), (3, 30.0), (5, 5.0), (6, 6.0), (7, 7.0)] {
        assert_eq!(
            clustering
                .degrees
                .get_priority(&node)
                .map(|degree| degree.into_scalar()),
            Some(degree)
        );
    }

    assert_eq!(clustering.tree_data.persistent.size[TreeIndex(0)].get(), 5);
    assert_eq!(
        clustering.tree_data.persistent.volume[TreeIndex(0)].into_scalar(),
        49.0
    );
}

#[test]
fn apply_node_ops_can_delete_all_nodes() {
    let mut clustering = test_clustering();

    <TestClustering as DynamicClusteringAlg<usize, f64>>::apply_node_ops(
        &mut clustering,
        &[(1, Some(strict(1.0))), (2, Some(strict(2.0)))],
    )
    .unwrap();

    <TestClustering as DynamicClusteringAlg<usize, f64>>::apply_node_ops(
        &mut clustering,
        &[(1, None), (2, None)],
    )
    .unwrap();

    assert_tree_consistent(&clustering);
    assert_eq!(clustering.num_leaves(), 0);
    assert!(clustering.node_to_tree_map.is_empty());
    assert!(clustering.tree_data.persistent.volume.is_empty());
}
