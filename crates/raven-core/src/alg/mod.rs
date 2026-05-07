use priority_queue::PriorityQueue;
use rustc_hash::{FxHashMap, FxHashSet};

mod coreset_impls;
mod sampling_impls;
mod tree_impls;

use std::num::NonZeroUsize;

use anyhow::{Result, anyhow};

use crate::{
    CoresetNeighbours, DynamicClusteringAlg, GraphBatchNeighbours,
    types::{
        AlgType, Contribution, FDelta, FloatScalar, HB, HS, NodeDegree, NonStrict,
        NonStrictCarrierOps, PartitionOutput, PartitionType, Strict, StrictCarrierOps, TreeIndex,
        Volume,
    },
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
pub struct DynamicClustering<const ARITY: usize, V, T> {
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
    V: std::hash::Hash + Eq + Clone + Copy,
    Strict<T>: Copy,
{
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
    fn apply_node_ops<G, E>(
        &mut self,
        diffs: &[(V, Option<Strict<T>>)],
        _graph_oracle: &G,
    ) -> Result<()>
    where
        G: GraphBatchNeighbours<V, T, E> + ?Sized,
    {
        let ops = self.classify_node_ops(diffs)?;
        let mut touched = FxHashSet::default();

        self.delete_nodes_compact(&ops.deleted, &mut touched)?;
        self.insert_fresh_nodes(&ops.fresh, &mut touched)?;
        self.update_modified_nodes(&ops.modified, &mut touched)?;

        self.apply_updates_from_set(&touched, |other, idx| {
            Self::one_step_recompute_size(idx, &mut other.tree_data.size);
            Self::one_step_recompute_volume(idx, &mut other.tree_data.volume);
        });

        Ok(())
    }

    fn query<G, C, E>(
        &mut self,
        partition: PartitionType<V>,
        graph_oracle: &G,
        coreset_oracle: &C,
    ) -> Result<PartitionOutput<V>>
    where
        G: GraphBatchNeighbours<V, T, E> + ?Sized,
        C: CoresetNeighbours<V, T, E> + ?Sized,
        E: std::fmt::Display,
    {
        let coreset_size = NonZeroUsize::new(self.coreset_size)
            .ok_or_else(|| anyhow!("coreset_size must be non-zero"))?;
        let sampling_seeds = NonZeroUsize::new(self.sampling_seeds)
            .ok_or_else(|| anyhow!("sampling_seeds must be non-zero"))?;

        let mut coreset =
            self.extract_coreset(graph_oracle, coreset_oracle, coreset_size, sampling_seeds)?;
        let coreset_graph = self.build_coreset_graph(&coreset, coreset_oracle)?;
        let (coreset_labels, num_clusters) =
            (self.cluster_alg)(coreset_graph.as_ref(), self.num_clusters);

        if coreset_labels.len() != coreset.nodes.len() {
            return Err(anyhow!(
                "cluster algorithm returned {} labels for {} coreset nodes",
                coreset_labels.len(),
                coreset.nodes.len()
            ));
        }
        coreset.coreset_labels = Some(coreset_labels);

        match partition {
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

                let (nodes, labels, _) =
                    self.rust_label_full_graph(&coreset, num_clusters, graph_oracle, &node_names)?;
                Ok(PartitionOutput::All(nodes, labels, num_clusters))
            }
            PartitionType::Subset(nodes) => {
                let (_, labels, _) =
                    self.rust_label_full_graph(&coreset, num_clusters, graph_oracle, nodes)?;
                Ok(PartitionOutput::Subset(labels, num_clusters))
            }
        }
    }
}
