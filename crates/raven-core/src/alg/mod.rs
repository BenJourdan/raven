use std::time::Duration;

use priority_queue::PriorityQueue;
use rand::SeedableRng;
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

/// Controls how query-trial sampling RNGs are initialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RngMode {
    /// Seed each trial from fresh system/thread randomness.
    Random,
    /// Derive one deterministic RNG seed per trial from the supplied base seed.
    Seeded(u64),
}

impl RngMode {
    pub(crate) fn rng_for_trial(self, trial_index: usize) -> rand::rngs::StdRng {
        match self {
            Self::Random => rand::make_rng(),
            Self::Seeded(seed) => {
                rand::rngs::StdRng::seed_from_u64(Self::trial_seed(seed, trial_index))
            }
        }
    }

    fn trial_seed(seed: u64, trial_index: usize) -> u64 {
        let mut z = seed.wrapping_add((trial_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

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
    pub rng_mode: RngMode,

    pub num_clusters: usize,
    pub cluster_alg: AlgType<T>,
    pub prop_name: String,
    last_query_timing: Option<QueryTiming>,
}

#[derive(Debug, Clone, Default)]
pub struct QueryTiming {
    pub total: Duration,
    pub setup: Duration,
    pub output: Duration,
    pub trials: Vec<QueryTrialTiming>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct QueryTrialTiming {
    pub total: Duration,
    pub extract_coreset: Duration,
    pub extract_coreset_breakdown: CoresetExtractionTiming,
    pub build_coreset_graph: Duration,
    pub cluster_coreset: Duration,
    pub label_partition: Duration,
    pub label_breakdown: FullGraphLabelTiming,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FullGraphLabelTiming {
    pub total: Duration,
    pub setup: Duration,
    pub degree_lookup: Duration,
    pub coreset_lookup: Duration,
    pub coreset_row_collect: Duration,
    pub center_stats: Duration,
    pub query_lookup: Duration,
    pub query_row_collect: Duration,
    pub target_info: Duration,
    pub label_nodes: Duration,

    pub labelled_nodes: usize,
    pub coreset_nodes: usize,
    pub degree_nodes: usize,
    pub coreset_lookup_edges: usize,
    pub query_lookup_edges: usize,
}

impl FullGraphLabelTiming {
    pub fn add(&mut self, other: Self) {
        self.total += other.total;
        self.setup += other.setup;
        self.degree_lookup += other.degree_lookup;
        self.coreset_lookup += other.coreset_lookup;
        self.coreset_row_collect += other.coreset_row_collect;
        self.center_stats += other.center_stats;
        self.query_lookup += other.query_lookup;
        self.query_row_collect += other.query_row_collect;
        self.target_info += other.target_info;
        self.label_nodes += other.label_nodes;

        self.labelled_nodes += other.labelled_nodes;
        self.coreset_nodes += other.coreset_nodes;
        self.degree_nodes += other.degree_nodes;
        self.coreset_lookup_edges += other.coreset_lookup_edges;
        self.query_lookup_edges += other.query_lookup_edges;
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CoresetExtractionTiming {
    pub setup: Duration,
    pub initial_repairs: Duration,
    pub seed_sampling: Duration,
    pub seed_repairs: Duration,
    pub total_contribution: Duration,
    pub smoothed_sampling: Duration,
    pub deduplication: Duration,

    pub initial_repair_calls: usize,
    pub seed_repair_calls: usize,
    pub seed_samples: usize,
    pub smoothed_samples: usize,
    pub dedup_unique_nodes: usize,

    pub repair_calls: usize,
    pub repair_point_seed_move: Duration,
    pub repair_point_f_delta: Duration,
    pub repair_point_lookup: Duration,
    pub repair_neighbour_scan: Duration,
    pub repair_neighbour_lookup: Duration,
    pub repair_neighbour_compare: Duration,
    pub repair_neighbour_f_delta_write: Duration,
    pub repair_neighbour_seed_move: Duration,
    pub repair_neighbour_f_delta_recompute: Duration,
    pub repair_new_seed_h_update: Duration,
    pub repair_new_seed_h_write: Duration,
    pub repair_new_seed_h_recompute: Duration,
    pub repair_old_seed_prepare: Duration,
    pub repair_old_seed_lookup: Duration,
    pub repair_old_seed_rescale: Duration,
    pub repair_old_seed_h_recompute: Duration,
    pub repair_neighbours_scanned: usize,
    pub repair_neighbours_improved: usize,
    pub repair_new_seed_h_update_nodes: usize,
    pub repair_old_seed_count: usize,
    pub repair_old_seed_neighbours_scanned: usize,
    pub repair_old_seed_neighbours_rescaled: usize,
    pub repair_old_seed_h_update_nodes: usize,
}

impl CoresetExtractionTiming {
    pub fn add(&mut self, other: Self) {
        self.setup += other.setup;
        self.initial_repairs += other.initial_repairs;
        self.seed_sampling += other.seed_sampling;
        self.seed_repairs += other.seed_repairs;
        self.total_contribution += other.total_contribution;
        self.smoothed_sampling += other.smoothed_sampling;
        self.deduplication += other.deduplication;

        self.initial_repair_calls += other.initial_repair_calls;
        self.seed_repair_calls += other.seed_repair_calls;
        self.seed_samples += other.seed_samples;
        self.smoothed_samples += other.smoothed_samples;
        self.dedup_unique_nodes += other.dedup_unique_nodes;

        self.repair_calls += other.repair_calls;
        self.repair_point_seed_move += other.repair_point_seed_move;
        self.repair_point_f_delta += other.repair_point_f_delta;
        self.repair_point_lookup += other.repair_point_lookup;
        self.repair_neighbour_scan += other.repair_neighbour_scan;
        self.repair_neighbour_lookup += other.repair_neighbour_lookup;
        self.repair_neighbour_compare += other.repair_neighbour_compare;
        self.repair_neighbour_f_delta_write += other.repair_neighbour_f_delta_write;
        self.repair_neighbour_seed_move += other.repair_neighbour_seed_move;
        self.repair_neighbour_f_delta_recompute += other.repair_neighbour_f_delta_recompute;
        self.repair_new_seed_h_update += other.repair_new_seed_h_update;
        self.repair_new_seed_h_write += other.repair_new_seed_h_write;
        self.repair_new_seed_h_recompute += other.repair_new_seed_h_recompute;
        self.repair_old_seed_prepare += other.repair_old_seed_prepare;
        self.repair_old_seed_lookup += other.repair_old_seed_lookup;
        self.repair_old_seed_rescale += other.repair_old_seed_rescale;
        self.repair_old_seed_h_recompute += other.repair_old_seed_h_recompute;
        self.repair_neighbours_scanned += other.repair_neighbours_scanned;
        self.repair_neighbours_improved += other.repair_neighbours_improved;
        self.repair_new_seed_h_update_nodes += other.repair_new_seed_h_update_nodes;
        self.repair_old_seed_count += other.repair_old_seed_count;
        self.repair_old_seed_neighbours_scanned += other.repair_old_seed_neighbours_scanned;
        self.repair_old_seed_neighbours_rescaled += other.repair_old_seed_neighbours_rescaled;
        self.repair_old_seed_h_update_nodes += other.repair_old_seed_h_update_nodes;
    }

    pub fn repair_wall(self) -> Duration {
        self.initial_repairs + self.seed_repairs
    }
}

/// Query-local metadata for coreset construction.
pub struct SamplingInfo<T> {
    pub x_star_idx: TreeIndex,
    pub sigma: Strict<T>,
    pub sigma_over_x_star_deg: Strict<T>,
    pub timestamp: usize,
    pub x_star_seed_set_volume_inv: Strict<T>,
    pub total_contribution_inv: Option<Contribution<T>>,
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
            rng_mode: RngMode::Random,
            num_clusters: 10,
            cluster_alg,
            prop_name: "unknown".to_string(),
            last_query_timing: None,
        }
    }

    pub fn last_query_timing(&self) -> Option<&QueryTiming> {
        self.last_query_timing.as_ref()
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

    pub fn with_rng_mode(mut self, rng_mode: RngMode) -> Self {
        self.rng_mode = rng_mode;
        self
    }

    pub fn with_rng_seed(self, seed: u64) -> Self {
        self.with_rng_mode(RngMode::Seeded(seed))
    }

    pub fn with_random_rng(self) -> Self {
        self.with_rng_mode(RngMode::Random)
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
