use std::num::NonZeroUsize;

use anyhow::{anyhow, Result};
use rayon::prelude::*;
use rustc_hash::FxHashSet;

use super::{DynamicClustering, ResizeQueryInfo, TrialWorkspace};
use crate::{
    error::DynamicCoresetError,
    types::{
        FloatScalar, NonStrict, NonStrictCarrierOps, PartitionOutput, PartitionType, Strict,
        StrictCarrierOps, TrialObjective, TrialOutputMode, TrialPartition,
    },
    DynamicClusteringAlg, GraphOracle,
};

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
        let rng_mode = self.rng_mode;

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
            .enumerate()
            .map(|(trial_index, (query_time, oracle))| -> Result<_> {
                let mut context = TrialWorkspace::<ARITY, _, _> {
                    timestamp,
                    persistent,
                    query_time,
                    node_to_tree_map,
                    tree_to_node_map,
                };
                let mut rng = rng_mode.rng_for_trial(trial_index);
                let mut coreset = context.extract_coreset_trial(
                    &mut **oracle,
                    sigma,
                    x_star,
                    x_star_degree,
                    coreset_size,
                    sampling_seeds,
                    &mut rng,
                )?;
                let mut coreset_graph =
                    context.build_coreset_graph(&coreset, &mut **oracle, sigma)?;
                let (coreset_labels, num_clusters) =
                    (cluster_alg)(&mut coreset_graph, requested_num_clusters);

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

        // We return all the trial partitions. The caller decides what to do with them.
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
