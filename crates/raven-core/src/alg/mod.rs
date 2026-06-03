use priority_queue::PriorityQueue;
use rustc_hash::{FxHashMap, FxHashSet};

mod coreset_impls;
mod query;
mod sampling_impls;
mod tree_impls;
mod tree_layout;
mod workspace;

#[cfg(test)]
mod tests;

use anyhow::{Result, anyhow};

use crate::types::{
    AlgType, Contribution, FloatScalar, NodeDegree, NonStrict, NonStrictCarrierOps, Strict,
    StrictCarrierOps, TreeIndex,
};

pub(crate) use tree_layout::TreeLayout;
pub use workspace::{Persistent, QueryTime, ResizeQueryInfo, TreeData, TrialWorkspace};

/// ARITY: the maximum number of children per node in the tree
/// V: the type of the node identifiers
/// T: the numeric type of node values (e.g., f32, f64)
pub struct DynamicClustering<const ARITY: usize, V, T> {
    /// Map stable unique node Ids to tree indices
    pub node_to_tree_map: FxHashMap<V, TreeIndex>,
    /// and the reverse map:
    pub tree_to_node_map: FxHashMap<TreeIndex, V>,

    /// degree priority queue
    pub degrees: PriorityQueue<V, NodeDegree<T>>,

    /// struct to hold tree data
    pub tree_data: TreeData<ARITY, T>,

    /// sigma shift to set
    pub sigma: Strict<T>,

    /// For lazy query time updates
    pub timestamp: usize,
    /// Whether to resize query time arrays during updates or only during queries.
    pub resize_query_info: ResizeQueryInfo,
    pub num_trials: usize,

    pub coreset_size: usize,
    pub sampling_seeds: usize,

    pub num_clusters: usize,
    pub cluster_alg: AlgType<T>,
    pub prop_name: String,
}

// Holds info for coreset construction.
pub struct SamplingInfo<V, T> {
    pub x_star: V,
    pub sigma: Strict<T>,
    pub sigma_over_x_star_deg: Strict<T>,
    pub timestamp: usize,
    pub x_star_seed_set_volume_inv: Strict<T>,
    pub total_contribution_inv: Option<Contribution<T>>,

    // store the weight of each seed
    seed_weight: FxHashMap<V, Strict<T>>,
    // lazy seed map for every node
    seed_map: FxHashMap<V, V>,
}

struct ClassifiedNodeOps<V, T> {
    fresh: Vec<(V, Strict<T>)>,
    modified: Vec<(V, Strict<T>)>,
    deleted: Vec<V>,
}

impl<const ARITY: usize, V, T> DynamicClustering<ARITY, V, T>
where
    V: std::hash::Hash + Eq + Copy,
    T: FloatScalar,
    Strict<T>: StrictCarrierOps<Scalar = T> + Copy,
    NonStrict<T>: NonStrictCarrierOps<Scalar = T>,
{
    pub fn new(cluster_alg: AlgType<T>) -> Self {
        let num_trials = 1;
        Self {
            node_to_tree_map: FxHashMap::default(),
            tree_to_node_map: FxHashMap::default(),
            degrees: PriorityQueue::new(),
            tree_data: TreeData {
                persistent: Persistent {
                    size: vec![],
                    volume: vec![],
                },
                query_time: std::iter::repeat_with(QueryTime::default)
                    .take(num_trials)
                    .collect(),
            },
            sigma: Strict::from_positive_scalar(
                T::from(1000.0).expect("default sigma should convert to scalar"),
            )
            .expect("default sigma must be positive and finite"),
            timestamp: 0,
            resize_query_info: ResizeQueryInfo::Updates,
            num_trials,
            coreset_size: 1024,
            sampling_seeds: 20,
            num_clusters: 10,
            cluster_alg,
            prop_name: "unknown".to_string(),
        }
    }

    pub fn with_sigma(mut self, sigma: Strict<T>) -> Self {
        self.sigma = sigma;
        self
    }

    pub fn with_resize_query_info(mut self, resize_query_info: ResizeQueryInfo) -> Self {
        if resize_query_info == ResizeQueryInfo::Updates {
            let tree_len = self.tree_data.persistent.size.len();
            self.tree_data
                .query_time
                .iter_mut()
                .for_each(|query_time| query_time.resize(tree_len));
        }
        self.resize_query_info = resize_query_info;
        self
    }

    pub fn with_num_trials(mut self, num_trials: usize) -> Self {
        let tree_len = self.tree_data.persistent.size.len();
        let resize_during_updates = self.resize_query_info == ResizeQueryInfo::Updates;
        self.tree_data.query_time.resize_with(num_trials, || {
            let mut query_time = QueryTime::default();
            if resize_during_updates {
                query_time.resize(tree_len);
            }
            query_time
        });
        self.num_trials = num_trials;
        self
    }

    pub fn with_coreset_size(mut self, coreset_size: usize) -> Self {
        self.coreset_size = coreset_size;
        self
    }

    pub fn with_sampling_seeds(mut self, sampling_seeds: usize) -> Self {
        self.sampling_seeds = sampling_seeds;
        self
    }

    pub fn with_num_clusters(mut self, num_clusters: usize) -> Self {
        self.num_clusters = num_clusters;
        self
    }

    pub fn with_prop_name(mut self, prop_name: impl Into<String>) -> Self {
        self.prop_name = prop_name.into();
        self
    }

    fn classify_node_ops(
        &self,
        diffs: &[(V, Option<Strict<T>>)],
    ) -> Result<ClassifiedNodeOps<V, T>> {
        let mut seen = FxHashSet::default();
        let mut fresh = Vec::new();
        let mut modified = Vec::new();
        let mut deleted = Vec::new();

        for (node, new_value) in diffs {
            if !seen.insert(*node) {
                return Err(anyhow!("duplicate node operation in update batch"));
            }

            match new_value {
                Some(value) => {
                    if self.node_to_tree_map.contains_key(node) {
                        modified.push((*node, *value));
                    } else {
                        fresh.push((*node, *value));
                    }
                }
                None => {
                    if self.node_to_tree_map.contains_key(node) {
                        deleted.push(*node);
                    }
                }
            }
        }

        Ok(ClassifiedNodeOps {
            fresh,
            modified,
            deleted,
        })
    }
}
