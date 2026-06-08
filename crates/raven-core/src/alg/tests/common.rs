use std::sync::Arc;

use rustc_hash::FxHashSet;

use crate::{
    alg::{DynamicClustering, ResizeQueryInfo},
    error::OracleError,
    types::{AlgType, Neighbourhoods, Strict, TreeIndex, Volume},
    DynamicClusteringAlg, GraphOracle,
};

pub(crate) type TestClustering = DynamicClustering<2, usize, f64>;

pub(crate) fn strict(value: f64) -> Strict<f64> {
    Strict::<f64>::new(value).unwrap()
}

pub(crate) fn volume(value: f64) -> Volume<f64> {
    Volume::from_scalar(value).unwrap()
}

pub(crate) fn test_clustering() -> TestClustering {
    let cluster_alg: AlgType<f64> = Arc::new(|_, _| (Vec::new(), 0));

    DynamicClustering::new(cluster_alg)
        .with_sigma(strict(1.0))
        .with_num_trials(1)
        .with_coreset_size(1)
        .with_sampling_seeds(1)
        .with_num_clusters(1)
        .with_prop_name("w")
}

pub(crate) fn use_zero_label_cluster_alg(clustering: &mut TestClustering) {
    clustering.cluster_alg = Arc::new(|graph, _| {
        let n = graph.symbolic().nrows();
        (vec![0; n], 1)
    });
}

pub(crate) fn apply_six_node_fixture(clustering: &mut TestClustering) {
    <TestClustering as DynamicClusteringAlg<usize, f64>>::apply_node_ops(
        clustering,
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
}

pub(crate) fn query_ready_clustering(
    resize_query_info: ResizeQueryInfo,
    num_trials: usize,
) -> TestClustering {
    let mut clustering = test_clustering()
        .with_resize_query_info(resize_query_info)
        .with_num_trials(num_trials)
        .with_coreset_size(3)
        .with_sampling_seeds(2);
    use_zero_label_cluster_alg(&mut clustering);
    apply_six_node_fixture(&mut clustering);
    clustering
}

pub(crate) fn apply_size_volume_updates(
    clustering: &mut TestClustering,
    touched: &FxHashSet<TreeIndex>,
) {
    clustering.apply_updates_from_set(touched, |other, idx| {
        TestClustering::one_step_recompute_size(idx, &mut other.tree_data.persistent.size);
        TestClustering::one_step_recompute_volume(idx, &mut other.tree_data.persistent.volume);
    });
}

pub(crate) struct EmptyOracle {
    offsets: Vec<usize>,
}

impl EmptyOracle {
    pub(crate) fn new() -> Self {
        Self {
            offsets: Vec::new(),
        }
    }

    fn empty_rows<'a>(
        &'a mut self,
        nodes: &[usize],
    ) -> Result<Neighbourhoods<'a, usize, f64>, OracleError<String>> {
        self.offsets.clear();
        self.offsets.resize(nodes.len() + 1, 0);
        Ok(Neighbourhoods::new(&[], &self.offsets))
    }
}

impl GraphOracle<usize, f64, String> for EmptyOracle {
    fn graph_neighbourhoods<'a>(
        &'a mut self,
        nodes: &[usize],
    ) -> Result<Neighbourhoods<'a, usize, f64>, OracleError<String>> {
        self.empty_rows(nodes)
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

pub(crate) fn assert_tree_consistent(clustering: &TestClustering) {
    let leaves = clustering.num_leaves();
    let total = TestClustering::total_count_for_leaves(leaves);
    let leaf_range = TestClustering::leaf_range_for_leaves(leaves);

    assert_eq!(clustering.tree_data.persistent.volume.len(), total);
    assert_eq!(clustering.tree_data.persistent.size.len(), total);
    assert_eq!(clustering.tree_data.query_time.len(), clustering.num_trials);
    for query_time in &clustering.tree_data.query_time {
        assert_eq!(query_time.timestamp.len(), total);
        assert_eq!(query_time.f_delta.len(), total);
        assert_eq!(query_time.h_b.len(), total);
        assert_eq!(query_time.h_s.len(), total);
    }

    assert_eq!(clustering.node_to_tree_map.len(), leaves);
    assert_eq!(clustering.tree_to_node_map.len(), leaves);
    assert_eq!(clustering.degrees.len(), leaves);

    for (&node, &idx) in &clustering.node_to_tree_map {
        assert!(
            leaf_range.contains(&idx.0),
            "node {node} mapped to non-leaf index {:?} outside {:?}",
            idx,
            leaf_range
        );
        assert_eq!(clustering.tree_to_node_map.get(&idx), Some(&node));

        let degree = clustering
            .degrees
            .get_priority(&node)
            .unwrap_or_else(|| panic!("missing degree for node {node}"));
        assert_eq!(
            degree.into_scalar(),
            clustering.tree_data.persistent.volume[idx].into_scalar()
        );
    }

    for (&idx, &node) in &clustering.tree_to_node_map {
        assert_eq!(clustering.node_to_tree_map.get(&node), Some(&idx));
    }

    if leaves == 0 {
        return;
    }

    let expected_volume = leaf_range
        .clone()
        .map(|idx| clustering.tree_data.persistent.volume[idx].into_scalar())
        .sum::<f64>();
    assert_eq!(
        clustering.tree_data.persistent.size[TreeIndex(0)].get(),
        leaves
    );
    assert_eq!(
        clustering.tree_data.persistent.volume[TreeIndex(0)].into_scalar(),
        expected_volume
    );
}
