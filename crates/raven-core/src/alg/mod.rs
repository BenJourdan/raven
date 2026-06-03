use priority_queue::PriorityQueue;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

mod coreset_impls;
mod sampling_impls;
mod tree_impls;

use std::num::NonZeroUsize;

use anyhow::{Result, anyhow};

use crate::{
    DynamicClusteringAlg, GraphOracle,
    error::DynamicCoresetError,
    types::{
        AlgType, Contribution, FDelta, FloatScalar, HB, HS, NodeDegree, NonStrict,
        NonStrictCarrierOps, PartitionOutput, PartitionType, Strict, StrictCarrierOps, TreeIndex,
        TrialObjective, TrialOutputMode, TrialPartition, Volume,
    },
};

/// The main data structure that backs the dynamic clustering algorithm.
/// This forms the sampling tree for f and g.
pub struct TreeData<const ARITY: usize, T> {
    pub persistent: Persistent<T>,
    pub query_time: Vec<QueryTime<T>>,
}
pub struct Persistent<T> {
    pub volume: Vec<Volume<T>>,
    pub size: Vec<NonZeroUsize>,
}
pub struct QueryTime<T> {
    pub timestamp: Vec<usize>,
    pub f_delta: Vec<FDelta<T>>,
    pub h_b: Vec<HB<T>>,
    pub h_s: Vec<HS<T>>,
}

impl<T> Default for QueryTime<T> {
    fn default() -> Self {
        Self {
            timestamp: Vec::new(),
            f_delta: Vec::new(),
            h_b: Vec::new(),
            h_s: Vec::new(),
        }
    }
}

impl<T> QueryTime<T>
where
    NonStrict<T>: NonStrictCarrierOps<Scalar = T>,
{
    pub fn clear(&mut self) {
        self.timestamp.clear();
        self.f_delta.clear();
        self.h_b.clear();
        self.h_s.clear();
    }
    pub fn truncate(&mut self, new_len: usize) {
        self.timestamp.truncate(new_len);
        self.f_delta.truncate(new_len);
        self.h_b.truncate(new_len);
        self.h_s.truncate(new_len);
    }
    pub fn resize(&mut self, new_len: usize) {
        self.timestamp.resize(new_len, 0);
        self.f_delta.resize_with(new_len, FDelta::zero);
        self.h_b.resize_with(new_len, HB::zero);
        self.h_s.resize_with(new_len, HS::zero);
    }
}

/// Should we resize the scratch arrays used for query time during updates,
/// or only during queries? At update time should help with query latency
/// but may increase update latency. Vice versa for only resizing during queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeQueryInfo {
    Updates,
    Query,
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

/// Workspace holding borrowed references for a single trial of coreset construction.
pub struct TrialWorkspace<'a, const ARITY: usize, V, T> {
    pub timestamp: usize,
    pub persistent: &'a Persistent<T>,
    pub query_time: &'a mut QueryTime<T>,
    pub tree_to_node_map: &'a FxHashMap<TreeIndex, V>,
    pub node_to_tree_map: &'a FxHashMap<V, TreeIndex>,
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

impl<const ARITY: usize, V, T> DynamicClusteringAlg<V, T> for DynamicClustering<ARITY, V, T>
where
    V: std::hash::Hash + Eq + Clone + Copy + Send + Sync,
    T: FloatScalar + Send + Sync,
    Strict<T>: StrictCarrierOps<Scalar = T> + Copy,
    NonStrict<T>: NonStrictCarrierOps<Scalar = T> + Copy,
{
    fn apply_node_ops(&mut self, diffs: &[(V, Option<Strict<T>>)]) -> Result<()> {
        let ops = self.classify_node_ops(diffs)?;
        let mut touched = FxHashSet::default();

        self.delete_nodes_compact(&ops.deleted, &mut touched)?;
        self.insert_fresh_nodes(&ops.fresh, &mut touched)?;
        self.update_modified_nodes(&ops.modified, &mut touched)?;

        self.apply_updates_from_set(&touched, |other, idx| {
            Self::one_step_recompute_size(idx, &mut other.tree_data.persistent.size);
            Self::one_step_recompute_volume(idx, &mut other.tree_data.persistent.volume);
        });

        Ok(())
    }

    fn query<O, E>(
        &mut self,
        partition: PartitionType<V>,
        trial_output_mode: TrialOutputMode,
        oracles: &mut [&mut O],
    ) -> Result<PartitionOutput<V, T>>
    where
        O: GraphOracle<V, T, E> + ?Sized + Send,
        E: std::fmt::Display,
    {
        if self.tree_data.persistent.size.is_empty() {
            return Err(DynamicCoresetError::NoData.into());
        }

        let coreset_size = NonZeroUsize::new(self.coreset_size)
            .ok_or_else(|| anyhow!("coreset_size must be non-zero"))?;
        let sampling_seeds = NonZeroUsize::new(self.sampling_seeds)
            .ok_or_else(|| anyhow!("sampling_seeds must be non-zero"))?;

        if self.num_trials == 0 {
            return Err(anyhow!("num_trials must be non-zero"));
        }

        // check we have the right number of oracles for the number of trials
        if oracles.len() != self.num_trials {
            return Err(anyhow!(
                "expected {} oracles for {} trials, but got {}",
                self.num_trials,
                self.num_trials,
                oracles.len()
            ));
        }
        if self.tree_data.query_time.len() != self.num_trials {
            return Err(anyhow!(
                "expected {} query time workspaces for {} trials, but got {}",
                self.num_trials,
                self.num_trials,
                self.tree_data.query_time.len()
            ));
        }
        // check that query time arrays are up to date
        if self.resize_query_info == ResizeQueryInfo::Query {
            let tree_len = self.tree_data.persistent.size.len();
            self.tree_data
                .query_time
                .iter_mut()
                .for_each(|query_time| query_time.resize(tree_len));
        } else {
            // if resizing during updates, we just need to check that the query time arrays are the right length
            let tree_len = self.tree_data.persistent.size.len();
            if self.tree_data.query_time.iter().any(|query_time| {
                query_time.timestamp.len() != tree_len
                    || query_time.f_delta.len() != tree_len
                    || query_time.h_b.len() != tree_len
                    || query_time.h_s.len() != tree_len
            }) {
                return Err(anyhow!(
                    "query time arrays are not the right length: expected {}, got {:?}",
                    tree_len,
                    self.tree_data
                        .query_time
                        .iter()
                        .map(|qt| (
                            qt.timestamp.len(),
                            qt.f_delta.len(),
                            qt.h_b.len(),
                            qt.h_s.len(),
                        ))
                        .collect::<Vec<_>>()
                ));
            }
        }

        // bump timestamp for this query:
        self.timestamp = self
            .timestamp
            .checked_add(1)
            .ok_or_else(|| anyhow!("query timestamp overflow"))?;
        let timestamp = self.timestamp;

        // setup workspaces for parallel coreset construction trials:
        let persistent = &self.tree_data.persistent;
        let node_to_tree_map = &self.node_to_tree_map;
        let tree_to_node_map = &self.tree_to_node_map;

        let sigma = self.sigma;
        let (&x_star, &x_star_degree) = self.degrees.peek().ok_or_else(|| {
            anyhow!("cannot query on empty graph: no x_star for coreset construction")
        })?;
        let cluster_alg = self.cluster_alg.clone();
        let requested_num_clusters = self.num_clusters;

        let node_names = match partition {
            PartitionType::All => {
                let mut nodes_by_tree_index = self
                    .node_to_tree_map
                    .iter()
                    .map(|(node, idx)| (*idx, *node))
                    .collect::<Vec<_>>();
                nodes_by_tree_index.sort_unstable_by_key(|(idx, _)| idx.0);
                let node_names = nodes_by_tree_index
                    .into_iter()
                    .map(|(_, node)| node)
                    .collect::<Vec<_>>();
                Some(node_names)
            }
            PartitionType::Subset(_) => None,
        };

        let labels_scores_clusters = self
            .tree_data
            .query_time
            .par_iter_mut()
            .zip(oracles.par_iter_mut())
            .map(|(query_time, oracle)| -> Result<_> {
                let mut context = TrialWorkspace::<ARITY, _, _> {
                    timestamp,
                    persistent,
                    query_time,
                    node_to_tree_map,
                    tree_to_node_map,
                };
                let mut coreset = context.extract_coreset_trial(
                    &mut **oracle,
                    sigma,
                    x_star,
                    x_star_degree,
                    coreset_size,
                    sampling_seeds,
                )?;
                let coreset_graph = context.build_coreset_graph(&coreset, &mut **oracle, sigma)?;
                let (coreset_labels, num_clusters) =
                    (cluster_alg)(coreset_graph.as_ref(), requested_num_clusters);

                if coreset_labels.len() != coreset.nodes.len() {
                    return Err(anyhow!(
                        "cluster algorithm returned {} labels for {} coreset nodes",
                        coreset_labels.len(),
                        coreset.nodes.len()
                    ));
                }
                coreset.coreset_labels = Some(coreset_labels);

                let (labels, scores) = match partition {
                    PartitionType::All => {
                        let (_nodes, labels, scores) = context.rust_label_full_graph(
                            &coreset,
                            num_clusters,
                            &mut **oracle,
                            node_names.as_ref().unwrap().as_slice(),
                            sigma,
                        )?;
                        (labels, scores)
                    }
                    PartitionType::Subset(nodes) => {
                        let (_nodes, labels, scores) = context.rust_label_full_graph(
                            &coreset,
                            num_clusters,
                            &mut **oracle,
                            nodes,
                            sigma,
                        )?;
                        (labels, scores)
                    }
                };

                Ok((labels, scores, num_clusters))
            })
            .collect::<Result<Vec<_>>>()?;

        if let TrialOutputMode::Winner(objective) = trial_output_mode {
            let best_idx = match objective {
                TrialObjective::KernelDistance => {
                    // choose the trial with the smallest sum of centroid distances as the winner.
                    labels_scores_clusters
                        .iter()
                        .enumerate()
                        .min_by(|(_, (_, scores_a, _)), (_, (_, scores_b, _))| {
                            let score_a = scores_a.iter().copied().sum::<T>();
                            let score_b = scores_b.iter().copied().sum::<T>();
                            score_a
                                .partial_cmp(&score_b)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx)
                        .unwrap_or(0)
                }
            };
            match partition {
                PartitionType::All => {
                    // We just return the labels and
                    let trial_part = TrialPartition {
                        trial_index: best_idx,
                        labels: labels_scores_clusters[best_idx].0.clone(),
                        scores: None, // don't need the scores since there is only one trial.
                        num_clusters: labels_scores_clusters[best_idx].2,
                    };
                    return Ok(PartitionOutput::All(node_names.unwrap(), vec![trial_part]));
                }
                PartitionType::Subset(_nodes) => {
                    // Similar but we don't need to include the nodes since the order was specified by the input subset.
                    let trial_part = TrialPartition {
                        trial_index: best_idx,
                        labels: labels_scores_clusters[best_idx].0.clone(),
                        scores: None, // don't need the scores since there is only one trial.
                        num_clusters: labels_scores_clusters[best_idx].2,
                    };
                    return Ok(PartitionOutput::Subset(vec![trial_part]));
                }
            }
        }

        // We return all the trial paritions. The caller decides what to do with them.
        match partition {
            PartitionType::All => {
                let trial_parts = labels_scores_clusters.into_iter().enumerate().map(
                    |(idx, (labels, scores, num_clusters))| TrialPartition {
                        trial_index: idx,
                        labels,
                        scores: Some(scores),
                        num_clusters,
                    },
                );
                Ok(PartitionOutput::All(
                    node_names.unwrap(),
                    trial_parts.collect(),
                ))
            }
            PartitionType::Subset(_nodes) => {
                let trial_parts = labels_scores_clusters.into_iter().enumerate().map(
                    |(idx, (labels, scores, num_clusters))| TrialPartition {
                        trial_index: idx,
                        labels,
                        scores: Some(scores),
                        num_clusters,
                    },
                );
                Ok(PartitionOutput::Subset(trial_parts.collect()))
            }
        }
    }
}
