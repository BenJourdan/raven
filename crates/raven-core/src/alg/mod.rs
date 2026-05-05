use rustc_hash::{FxHashMap, FxHashSet};
use priority_queue::PriorityQueue;

mod sampling_impls;

use std::num::NonZeroUsize;

use crate::types::{
    Volume,
    FDelta,
    TreeIndex,
    NodeDegree,
    Contribution,
    HB,
    HS,
    Strict,
    AlgType,
};






/// The main data structure that backs the dynamic clustering algorithm.
/// This forms the sampling tree for f and g.
#[derive(Default, Debug)]
pub struct TreeData<const ARITY: usize, T> {
    pub timestamp: Vec<usize>,
    pub volume: Vec<Volume<T>>,
    pub size: Vec<NonZeroUsize>,
    pub f_delta: Vec<FDelta<T>>,
    pub h_b: Vec<HB<T>>,
    pub h_s: Vec<HS<T>>,
}

/// ARITY: the maximum number of children per node in the tree
/// V: the type of the node identifiers
/// T: the numeric type of node values (e.g., f32, f64)
pub struct DynamicClustering< const ARITY: usize, V, T>{
    // Map stable unique node Ids to tree indices
    pub node_to_tree_map: FxHashMap<V, TreeIndex>,
    // and the reverse map:
    pub tree_to_node_map: FxHashMap<TreeIndex, V>,

    // degree priority queue
    pub degrees: PriorityQueue<V, NodeDegree<T>>,

    // struct to hold tree data
    pub tree_data: TreeData<ARITY, T>,

    // sigma shift to set
    pub sigma: Strict<T>,

    // For lazy query time updates
    pub timestamp: usize,

    pub update_set: FxHashSet<TreeIndex>,

    pub coreset_size: usize,
    pub sampling_seeds: usize,

    pub num_clusters: usize,
    pub cluster_alg: AlgType,
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