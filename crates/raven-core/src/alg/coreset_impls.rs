use std::num::NonZero;

use anyhow::{Result, anyhow};
use faer::sparse::{SparseRowMat, SymbolicSparseRowMat};
use rand::RngExt;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use super::{SamplingInfo, TreeLayout, TrialWorkspace};
use crate::{
    GraphOracle,
    error::DynamicCoresetError,
    types::{
        Contribution, EdgeWeight, FDelta, FloatScalar, HB, HS, Neighbourhoods, NodeDegree,
        NonStrict, NonStrictCarrierOps, Strict, StrictCarrierOps, TreeIndex,
    },
};

pub struct Coreset<V, T> {
    pub nodes: Vec<V>,
    pub node_indices: Vec<TreeIndex>,
    pub weights: Vec<Strict<T>>,
    pub coreset_labels: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Copy)]
struct CoresetProjectionInfo<T> {
    label: usize,
    weight: T,
    degree: T,
}

fn lookup_graph_neighbourhoods<'a, V, T, G, E>(
    oracle: &'a mut G,
    nodes: &[V],
    context: &str,
) -> Result<Neighbourhoods<'a, V, T>>
where
    G: GraphOracle<V, T, E> + ?Sized,
    E: std::fmt::Display,
{
    oracle
        .graph_neighbourhoods(nodes)
        .map_err(|e| anyhow!("{context}: {e}"))
}

fn lookup_coreset_neighbourhoods<'a, V, T, G, E>(
    oracle: &'a mut G,
    nodes: &[V],
    context: &str,
) -> Result<Neighbourhoods<'a, V, T>>
where
    G: GraphOracle<V, T, E> + ?Sized,
    E: std::fmt::Display,
{
    oracle
        .coreset_neighbourhoods(nodes)
        .map_err(|e| anyhow!("{context}: {e}"))
}

fn lookup_graph_neighbourhoods_intersecting<'a, V, T, G, E>(
    oracle: &'a mut G,
    sources: &[V],
    targets: &[V],
    context: &str,
) -> Result<Neighbourhoods<'a, V, T>>
where
    G: GraphOracle<V, T, E> + ?Sized,
    E: std::fmt::Display,
{
    oracle
        .graph_neighbourhoods_intersecting(sources, targets)
        .map_err(|e| anyhow!("{context}: {e}"))
}

fn lookup_single_graph_neighbourhood<'a, V, T, G, E>(
    oracle: &'a mut G,
    node_query: &'a [V; 1],
    context: &str,
) -> Result<&'a [(V, Strict<T>)]>
where
    G: GraphOracle<V, T, E> + ?Sized,
    E: std::fmt::Display,
{
    let neighbourhoods = lookup_graph_neighbourhoods(oracle, node_query.as_slice(), context)?;
    neighbourhoods
        .row(0)
        .ok_or_else(|| anyhow!("{context}: oracle returned no row for single-node lookup"))
}

// SamplingInfo impl

impl<V, T> SamplingInfo<V, T>
where
    V: std::hash::Hash + Eq + Clone + Copy,
    T: FloatScalar, // T must be a floating point type (either f32 or f64)
    Strict<T>: StrictCarrierOps<Scalar = T> + Copy,
    NonStrict<T>: NonStrictCarrierOps<Scalar = T> + Copy,
{
    pub fn new(
        x_star: V,
        sigma: Strict<T>,
        sigma_over_x_star_deg: Strict<T>,
        timestamp: usize,
        total_weight: Strict<T>,
    ) -> Self {
        // weights are always positive, ASSUMING a definite kernel
        // because seed sets will always contain at least the seed.
        let mut seed_weight = FxHashMap::<V, Strict<T>>::default();
        // Initially, the only seed is x^*, with seed weight equal to the total weight (volume) of the input
        seed_weight.insert(x_star, total_weight);

        let mut seed_map = FxHashMap::<V, V>::default();
        // Initially, we just have x^* maps to itself:
        seed_map.insert(x_star, x_star);

        Self {
            x_star,
            sigma,
            sigma_over_x_star_deg,
            timestamp,
            x_star_seed_set_volume_inv: Strict::<T>::from_positive_scalar(
                T::ONE / total_weight.into_scalar(),
            )
            .expect("total weight must have a positive finite reciprocal"),
            total_contribution_inv: None,
            seed_weight,
            seed_map,
        }
    }

    pub fn get_seed(&mut self, node: V) -> V {
        // return the seed of a point, defaulting to x_star if not seen before
        *self.seed_map.entry(node).or_insert(self.x_star)
    }

    pub fn set_seed(&mut self, node: V, seed: V) {
        // Overwrite any existing seed entry (get_seed initializes to x_star on first access).
        self.seed_map.insert(node, seed);
    }

    pub fn modify_seed_weight(&mut self, seed: V, diff: T) -> Result<()> {
        // increment the seed weight of seed by diff. If it is not present, insert it with this value
        let new_weight_scalar = match self.seed_weight.get(&seed) {
            Some(weight) => weight.into_scalar() + diff,
            None => diff,
        };

        let new_weight = Strict::<T>::from_positive_scalar(new_weight_scalar)
            .map_err(|e| anyhow!("seed weight update produced invalid weight: {e}"))?;
        self.seed_weight.insert(seed, new_weight);

        // keep x_star seed set volume in sync for g() smoothing term
        if seed == self.x_star {
            self.x_star_seed_set_volume_inv =
                Strict::<T>::from_positive_scalar(T::ONE / new_weight.into_scalar())
                    .map_err(|e| anyhow!("x_star seed weight reciprocal is invalid: {e}"))?;
        }

        Ok(())
    }

    pub fn get_seed_weight(&self, seed: V) -> Strict<T> {
        *self.seed_weight.get(&seed).unwrap()
    }
}

impl<const ARITY: usize, V, T> TrialWorkspace<'_, ARITY, V, T>
where
    V: std::hash::Hash + Eq + Clone + Copy,
    T: FloatScalar, // T must be a floating point type (either f32 or f64)
    Strict<T>: StrictCarrierOps<Scalar = T> + Copy,
    NonStrict<T>: NonStrictCarrierOps<Scalar = T> + Copy,
{
    fn move_seed_membership(
        info: &mut SamplingInfo<V, T>,
        node: V,
        new_seed: V,
        weight: T,
        old_seeds: &mut FxHashSet<V>,
        allow_already_in_seed: bool,
        context: &str,
    ) -> Result<()> {
        let old_seed = info.get_seed(node);
        if old_seed == new_seed {
            if allow_already_in_seed {
                info.set_seed(node, new_seed);
                return Ok(());
            }

            return Err(anyhow!(
                "{context}: node was already assigned to the new seed during seed membership update"
            ));
        }

        old_seeds.insert(old_seed);
        info.modify_seed_weight(old_seed, -weight)?;
        info.modify_seed_weight(new_seed, weight)?;
        info.set_seed(node, new_seed);

        Ok(())
    }

    fn recompute_f_delta_to_root(&mut self, source: TreeIndex) {
        let timestamp = self.timestamp;
        let query_time = &mut *self.query_time;
        TreeLayout::<ARITY>::apply_updates_from_single(source, |idx| {
            TreeLayout::<ARITY>::one_step_recompute_non_strict_with_timestamp(
                idx,
                &mut query_time.f_delta,
                &query_time.timestamp,
                timestamp,
                |_i| FDelta::zero(),
            );
            query_time.timestamp[idx] = timestamp;
        });
    }

    fn recompute_h_to_root(&mut self, source: TreeIndex) {
        let timestamp = self.timestamp;
        let persistent = self.persistent;
        let query_time = &mut *self.query_time;
        TreeLayout::<ARITY>::apply_updates_from_single(source, |idx| {
            TreeLayout::<ARITY>::one_step_recompute_non_strict_with_timestamp(
                idx,
                &mut query_time.h_b,
                &query_time.timestamp,
                timestamp,
                |i| {
                    // HB::from_scalar(persistent.volume[i].into_scalar())
                    //     .expect("volume must be non-negative")
                    unsafe { HB::from_scalar_unchecked(persistent.volume[i].into_scalar()) }
                },
            );
            TreeLayout::<ARITY>::one_step_recompute_non_strict_with_timestamp(
                idx,
                &mut query_time.h_s,
                &query_time.timestamp,
                timestamp,
                |_i| HS::zero(),
            );
            query_time.timestamp[idx] = timestamp;
        });
    }

    fn recompute_h_s_from_set(&mut self, update_set: &FxHashSet<TreeIndex>) {
        let timestamp = self.timestamp;
        let total = self.persistent.size.len();
        let query_time = &mut *self.query_time;
        TreeLayout::<ARITY>::apply_updates_from_set(total, update_set, |idx| {
            TreeLayout::<ARITY>::one_step_recompute_non_strict_with_timestamp(
                idx,
                &mut query_time.h_s,
                &query_time.timestamp,
                timestamp,
                |_i| HS::zero(),
            );
            query_time.timestamp[idx] = timestamp;
        });
    }

    pub fn extract_coreset_trial<G, E>(
        &mut self,
        graph_oracle: &mut G,
        sigma: Strict<T>,
        x_star: V,
        x_star_degree: NodeDegree<T>,
        coreset_size: NonZero<usize>,
        sampling_seeds: NonZero<usize>,
        rng: &mut impl rand::Rng,
    ) -> Result<Coreset<V, T>>
    where
        G: GraphOracle<V, T, E> + ?Sized + Send,
        E: std::fmt::Display,
    {
        // basic sanity: can't sample more seeds than leaves or build a coreset smaller than the seed set
        let root_size = self
            .persistent
            .size
            .first()
            .ok_or(DynamicCoresetError::NoData)?;
        if sampling_seeds.get() < 2 {
            return Err(anyhow!(
                "Expected at least 2 sampling seeds; got {}",
                sampling_seeds
            ));
        }

        if !(sampling_seeds < coreset_size && coreset_size < *root_size) {
            return Err(anyhow!(
                "Expected sampling_seeds < coreset_size < size(root); got {} < {} < {}",
                sampling_seeds,
                coreset_size,
                root_size
            ));
        }

        let timestamp = self.timestamp;

        debug_assert!(
            x_star_degree.into_scalar().is_finite() && x_star_degree.into_scalar() > T::ZERO,
            "x_star must have positive finite degree"
        );

        let sigma_over_x_star_deg =
            Strict::<T>::from_positive_scalar(sigma.into_scalar() / x_star_degree.into_scalar())
                .map_err(|e| anyhow!("sigma/x_star_degree was invalid: {e}"))?;

        let mut info = SamplingInfo::new(
            x_star,
            sigma,
            sigma_over_x_star_deg,
            timestamp,
            self.persistent.volume[0].0,
        );

        // sanity: leaves marked deleted should carry zero volume
        // self.assert_zero_volume_for_empty_leaves(&info);

        // first we add x_star:
        self.repair(x_star, &mut info, graph_oracle)?;

        // Now we sample a node uniformly:
        let tree_size = self.persistent.size.len();
        let num_leaves = self.node_to_tree_map.len();

        let uniform_idx = TreeIndex(rng.random_range(tree_size - num_leaves..tree_size));
        let uniform_node = *self.tree_to_node_map.get(&uniform_idx).unwrap();
        self.repair(uniform_node, &mut info, graph_oracle)?;

        let remaining_seeds = sampling_seeds.get().saturating_sub(2);
        for i in 0..remaining_seeds {
            // Sample a point according to f:
            let (node, _, _) = self.sample(&info, rng).map_err(|e| {
                anyhow!("failed sampling seed {} of {}: {e}", i + 1, remaining_seeds)
            })?;
            self.repair(node, &mut info, graph_oracle)?;
        }

        // populate total_contribution_inv
        let total_contribution = self.f(TreeIndex(0), &info);
        let total_contribution_scalar = total_contribution.into_scalar();
        debug_assert!(
            total_contribution_scalar.is_finite() && total_contribution_scalar > T::ZERO,
            "total contribution must be positive and finite"
        );
        info.total_contribution_inv = Some(
            Contribution::from_scalar(T::ONE / total_contribution_scalar).map_err(|e| {
                anyhow!("total contribution reciprocal was not non-negative finite: {e}")
            })?,
        );

        let coreset_size_f =
            T::from(coreset_size.get()).expect("coreset size should convert to scalar");
        let coreset_iterator = (0..coreset_size.get()).map(|_| {
            let (node, idx, prob) = self.sample_smoothed(&info, rng).unwrap();
            let node_deg = self.persistent.volume[idx].into_scalar();
            let weight = node_deg / (prob.into_scalar() * coreset_size_f);
            (node, idx, weight)
        });

        // Now we deduplicate the coreset:
        let mut coreset: FxHashMap<(V, TreeIndex), T> = FxHashMap::default();
        for (v, index, weight) in coreset_iterator {
            let entry = coreset.entry((v, index)).or_insert(T::ZERO);
            *entry = *entry + weight;
        }

        let mut unique_vs = Vec::with_capacity(coreset.len());
        let mut unique_indices = Vec::with_capacity(coreset.len());
        let mut weights = Vec::with_capacity(coreset.len());
        for ((v, idx), weight) in coreset {
            debug_assert!(
                weight.is_finite() && weight > T::ZERO,
                "deduplicated coreset weight must be positive finite"
            );
            unique_vs.push(v);
            unique_indices.push(idx);
            weights.push(
                Strict::<T>::from_positive_scalar(weight)
                    .map_err(|e| anyhow!("deduplicated coreset weight was invalid: {e}"))?,
            );
        }

        Ok(Coreset {
            nodes: unique_vs,
            node_indices: unique_indices,
            weights,
            coreset_labels: None,
        })
    }

    pub fn repair<G, E>(
        &mut self,
        point_added: V,
        info: &mut SamplingInfo<V, T>,
        graph_oracle: &mut G,
    ) -> Result<()>
    where
        G: GraphOracle<V, T, E> + ?Sized,
        E: std::fmt::Display,
    {
        // We implicitly add the point to the init set, update its neighbours,
        // and seed maps / seed weights.
        let point_added_index = *self.node_to_tree_map.get(&point_added).unwrap();

        let point_added_volume = self.persistent.volume[point_added_index];
        let point_added_degree = NodeDegree::from_scalar(point_added_volume.into_scalar())
            .map_err(|e| anyhow!("point-added volume could not be used as a node degree: {e}"))?;
        let point_added_weight = point_added_volume.into_scalar();
        let mut old_seeds = FxHashSet::default();

        Self::move_seed_membership(
            info,
            point_added,
            point_added,
            point_added_weight,
            &mut old_seeds,
            true,
            "repair point seed move",
        )?;

        // Zero this point's f contribution by matching f_delta to f_b.
        let f_b = self.f_b(point_added_index, info);
        self.query_time.f_delta[point_added_index] = FDelta::from_scalar(f_b.into_scalar())
            .map_err(|e| anyhow!("base contribution could not be stored as f_delta: {e}"))?;
        self.query_time.timestamp[point_added_index] = info.timestamp;
        self.recompute_f_delta_to_root(point_added_index);

        let point_query = [point_added];
        let neighbours =
            lookup_single_graph_neighbourhood(graph_oracle, &point_query, "repair point lookup")?;

        let mut filtered_neighbours = Vec::with_capacity(neighbours.len());

        for (neighbour, edge_weight) in neighbours.iter() {
            let neighbour_idx = *self.node_to_tree_map.get(neighbour).ok_or_else(|| {
                anyhow!("repair point lookup returned a neighbour missing from the tree")
            })?;
            let neighbour_volume = self.persistent.volume[neighbour_idx];

            let weighted_distance_to_point_added =
                Self::weighted_kernel_distance(point_added_degree, EdgeWeight::new(*edge_weight));
            let current_contribution = self.f(neighbour_idx, info);

            if weighted_distance_to_point_added < current_contribution {
                // Neighbour is now closer to this point.
                filtered_neighbours.push(*neighbour);

                let new_f_delta_term = (self.f_b(neighbour_idx, info).into_scalar()
                    - weighted_distance_to_point_added.into_scalar())
                .max(T::ZERO);
                self.query_time.f_delta[neighbour_idx] = FDelta::from_scalar(new_f_delta_term)
                    .map_err(|e| {
                        anyhow!("updated f_delta term was not non-negative finite: {e}")
                    })?;
                self.query_time.timestamp[neighbour_idx] = info.timestamp;
                self.recompute_f_delta_to_root(neighbour_idx);

                Self::move_seed_membership(
                    info,
                    *neighbour,
                    point_added,
                    neighbour_volume.into_scalar(),
                    &mut old_seeds,
                    // Nodes can already belong to this seed set. In
                    // particular, unseen nodes default to x_star, so repairing
                    // x_star should not try to debit and credit the same seed.
                    true,
                    "repair neighbour seed move",
                )?;
            }
        }

        let seed_weight = info.get_seed_weight(point_added);
        let seed_weight_scalar = seed_weight.into_scalar();
        debug_assert!(
            seed_weight_scalar.is_finite() && seed_weight_scalar > T::ZERO,
            "seed weight must be non-zero for h_s update"
        );

        for z in filtered_neighbours.into_iter().chain([point_added]) {
            let z_idx = *self.node_to_tree_map.get(&z).unwrap();

            self.query_time.h_b[z_idx] = HB::zero();
            let deg_z = self.persistent.volume[z_idx].into_scalar();
            self.query_time.h_s[z_idx] =
                HS::from_scalar(deg_z / seed_weight_scalar).map_err(|e| {
                    anyhow!("h_s update for new seed set was not non-negative finite: {e}")
                })?;

            self.query_time.timestamp[z_idx] = info.timestamp;
            self.recompute_h_to_root(z_idx);
        }

        // Update h_s for nodes in old seed sets whose seed-set weights changed, except x^*.
        let x_star = info.x_star;
        let timestamp = info.timestamp;

        let old_seeds_and_weights = old_seeds
            .into_iter()
            .filter(|s| *s != x_star)
            .map(|s| (s, info.get_seed_weight(s)))
            .collect::<Vec<_>>();

        if old_seeds_and_weights.is_empty() {
            return Ok(());
        }

        let mut h_s_update_set = FxHashSet::default();
        let old_seed_nodes = old_seeds_and_weights
            .iter()
            .map(|(s, _)| *s)
            .collect::<Vec<_>>();
        let old_seed_neighbour_batches =
            lookup_graph_neighbourhoods(graph_oracle, &old_seed_nodes, "old seed lookup")?;

        for ((s, seed_weight), neighbours) in old_seeds_and_weights
            .into_iter()
            .zip(old_seed_neighbour_batches.iter())
        {
            let seed_weight_scalar = seed_weight.into_scalar();
            debug_assert!(
                seed_weight_scalar.is_finite() && seed_weight_scalar > T::ZERO,
                "old seed weight must be non-zero for h_s rescale"
            );

            for (z, _) in neighbours
                .iter()
                .filter(|(neighbour, _)| info.get_seed(*neighbour) == s)
            {
                let z_idx = *self.node_to_tree_map.get(z).ok_or_else(|| {
                    anyhow!("old seed lookup returned a neighbour missing from the tree")
                })?;
                let deg_z = self.persistent.volume[z_idx].into_scalar();
                self.query_time.h_s[z_idx] =
                    HS::from_scalar(deg_z / seed_weight_scalar).map_err(|e| {
                        anyhow!("h_s rescale for old seed set was not non-negative finite: {e}")
                    })?;
                self.query_time.timestamp[z_idx] = timestamp;
                h_s_update_set.insert(z_idx);
            }
        }

        self.recompute_h_s_from_set(&h_s_update_set);

        Ok(())
    }
    pub fn build_coreset_graph<C, E>(
        &self,
        coreset: &Coreset<V, T>,
        coreset_oracle: &mut C,
        sigma: Strict<T>,
    ) -> Result<SparseRowMat<usize, T>>
    where
        C: GraphOracle<V, T, E> + ?Sized,
        E: std::fmt::Display,
    {
        let n = coreset.nodes.len();
        if coreset.node_indices.len() != n || coreset.weights.len() != n {
            return Err(anyhow!(
                "coreset graph build expected matching node/index/weight lengths; got nodes={}, indices={}, weights={}",
                n,
                coreset.node_indices.len(),
                coreset.weights.len()
            ));
        }

        let coreset_neighbourhoods = lookup_coreset_neighbourhoods(
            coreset_oracle,
            coreset.nodes.as_slice(),
            "coreset graph lookup",
        )?;

        let node_name_to_index = coreset
            .nodes
            .iter()
            .enumerate()
            .map(|(idx, name)| (*name, idx))
            .collect::<FxHashMap<V, usize>>();

        // For each coreset node i, precompute W_C[i] * D_C[i]^{-1}.
        // Here W_C is the diagonal matrix of coreset weights and D_C is the
        // degree diagonal restricted to coreset nodes.
        let weight_degree_inv = (0..n)
            .map(|idx| {
                coreset.weights[idx].into_scalar()
                    / self.persistent.volume[coreset.node_indices[idx]].into_scalar()
            })
            .collect::<Vec<_>>();

        let mut data = Vec::<T>::with_capacity(n * 200);
        let mut indices = Vec::<usize>::with_capacity(n * 200);
        let mut indptr = Vec::<usize>::with_capacity(n + 1);
        let mut nnz_per_row = Vec::<usize>::with_capacity(n);
        let mut indptr_counter = 0;

        // Build the shifted and reweighted coreset adjacency:
        //
        // A_C = W_C D_C^{-1} A_C D_C^{-1} W_C
        //     + sigma W_C D_C^{-1} W_C
        //
        // where:
        // - A_C is the adjacency matrix restricted to coreset nodes,
        // - W_C is the diagonal matrix of coreset weights,
        // - D_C is the degree diagonal restricted to coreset nodes,
        // - sigma is the regularising diagonal shift.
        for (i, (&node_name, neighbours)) in coreset
            .nodes
            .iter()
            .zip(coreset_neighbourhoods.iter())
            .enumerate()
        {
            let weight_degree_inv_i = weight_degree_inv[i];
            let mut no_diag = true;
            let mut row_entries = neighbours
                .iter()
                .filter_map(|(neighbour_name, edge_weight)| {
                    let coreset_j = *node_name_to_index.get(neighbour_name)?;
                    let edge_weight = edge_weight.into_scalar();
                    if node_name == *neighbour_name {
                        no_diag = false;
                        Some((
                            coreset_j,
                            edge_weight * weight_degree_inv_i * weight_degree_inv_i
                                + sigma.into_scalar()
                                    * coreset.weights[i].into_scalar()
                                    * weight_degree_inv_i,
                        ))
                    } else {
                        Some((
                            coreset_j,
                            edge_weight * weight_degree_inv_i * weight_degree_inv[coreset_j],
                        ))
                    }
                })
                .collect::<Vec<(usize, T)>>();

            if no_diag {
                // The oracle may omit an explicit self-loop. Add the diagonal
                // shift term in that case.
                row_entries.push((
                    i,
                    sigma.into_scalar() * coreset.weights[i].into_scalar() * weight_degree_inv_i,
                ));
            }

            row_entries.sort_unstable_by_key(|&(idx, _)| idx);

            data.extend(row_entries.iter().map(|x| x.1));
            indices.extend(row_entries.iter().map(|x| x.0));
            let nnz = row_entries.len();
            nnz_per_row.push(nnz);
            indptr.push(indptr_counter);
            indptr_counter += nnz;
        }

        indptr.push(indptr_counter);
        Ok(SparseRowMat::new(
            SymbolicSparseRowMat::<usize>::new_checked(n, n, indptr, Some(nnz_per_row), indices),
            data,
        ))
    }
    pub fn rust_label_full_graph<G, E>(
        &self,
        coreset: &Coreset<V, T>,
        num_clusters: usize,
        graph_oracle: &mut G,
        nodes: &[V],
        sigma: Strict<T>,
    ) -> Result<(Vec<V>, Vec<usize>, Vec<T>)>
    where
        G: GraphOracle<V, T, E> + ?Sized,
        E: std::fmt::Display,
        V: Send + Sync,
        T: Send + Sync,
    {
        if num_clusters == 0 {
            return Err(anyhow!(
                "full graph labelling requires at least one cluster"
            ));
        }
        if coreset.nodes.len() != coreset.weights.len() {
            return Err(anyhow!(
                "full graph labelling expected matching coreset node/weight lengths; got nodes={}, weights={}",
                coreset.nodes.len(),
                coreset.weights.len()
            ));
        }

        let shift = sigma.into_scalar();
        let coreset_labels = coreset
            .coreset_labels
            .as_ref()
            .ok_or_else(|| anyhow!("coreset labels must be set before full-graph labelling"))?;
        if coreset_labels.len() != coreset.nodes.len() {
            return Err(anyhow!(
                "full graph labelling expected one label per coreset node; got nodes={}, labels={}",
                coreset.nodes.len(),
                coreset_labels.len()
            ));
        }

        let node_names = nodes.to_vec();

        // Union of all nodes we will touch: labelled nodes plus coreset nodes.
        // This deduplicates the graph-wide neighbourhood batch lookup.
        let mut all_nodes_set: FxHashSet<V> = node_names.iter().copied().collect();
        all_nodes_set.extend(coreset.nodes.iter().copied());
        let all_nodes: Vec<V> = all_nodes_set.iter().copied().collect();

        // Precompute degree lookups to avoid touching the main data structures
        // inside parallel labelling loops.
        let mut degree_map = FxHashMap::<V, T>::default();
        for node in all_nodes.iter().copied() {
            let idx = *self
                .node_to_tree_map
                .get(&node)
                .ok_or_else(|| anyhow!("full graph labelling node was missing from the tree"))?;
            degree_map.insert(node, self.persistent.volume[idx].into_scalar());
        }

        let (center_norms, center_denoms) = {
            let coreset_neighbourhoods = lookup_graph_neighbourhoods_intersecting(
                graph_oracle,
                coreset.nodes.as_slice(),
                coreset.nodes.as_slice(),
                "full graph coreset lookup",
            )?;
            let coreset_rows = coreset_neighbourhoods.iter().collect::<Vec<_>>();
            self.compute_full_graph_center_stats(
                coreset,
                num_clusters,
                coreset_labels,
                &degree_map,
                &coreset_rows,
            )?
        };

        let labels_and_distances = {
            let query_neighbourhoods = lookup_graph_neighbourhoods_intersecting(
                graph_oracle,
                node_names.as_slice(),
                coreset.nodes.as_slice(),
                "full graph query lookup",
            )?;
            let query_rows = query_neighbourhoods.iter().collect::<Vec<_>>();
            self.label_full_graph_nodes_from_centers(
                node_names.as_slice(),
                &query_rows,
                coreset,
                coreset_labels,
                &degree_map,
                &center_norms,
                &center_denoms,
                shift,
            )?
        };

        Ok((node_names, labels_and_distances.0, labels_and_distances.1))
    }

    fn compute_full_graph_center_stats(
        &self,
        coreset: &Coreset<V, T>,
        num_clusters: usize,
        coreset_labels: &[usize],
        degree_map: &FxHashMap<V, T>,
        coreset_rows: &[&[(V, Strict<T>)]],
    ) -> Result<(Vec<T>, Vec<T>)>
    where
        V: Send + Sync,
        T: Send + Sync,
    {
        if coreset_rows.len() != coreset.nodes.len() {
            return Err(anyhow!(
                "full graph center stat build expected one row per coreset node; got nodes={}, rows={}",
                coreset.nodes.len(),
                coreset_rows.len()
            ));
        }

        let coreset_adjacency = coreset
            .nodes
            .iter()
            .copied()
            .zip(coreset_rows.iter().copied())
            .collect::<FxHashMap<_, _>>();

        // Group the coreset nodes/weights by cluster label.
        let mut coreset_grouped = std::iter::repeat_with(|| (Vec::new(), Vec::new()))
            .take(num_clusters)
            .collect::<Vec<(Vec<V>, Vec<T>)>>();
        for ((&name, &label), weight) in coreset
            .nodes
            .iter()
            .zip(coreset_labels.iter())
            .zip(coreset.weights.iter())
        {
            if label >= num_clusters {
                return Err(anyhow!(
                    "coreset label {} was outside the cluster range 0..{}",
                    label,
                    num_clusters
                ));
            }
            coreset_grouped[label].0.push(name);
            coreset_grouped[label].1.push(weight.into_scalar());
        }

        // For cluster C, the represented center is
        //   c = (sum_{i in C} w_i phi(i)) / (sum_{i in C} w_i).
        //
        // Its squared norm is
        //   ||c||^2 = denom^{-2} sum_{i,j in C} w_i w_j k(i,j)
        //
        // with the normalized graph kernel used by dyn-cc's full-graph
        // projection:
        //   k(i,j) = A_ij / (deg(i) deg(j)).
        //
        // The coreset graph construction applies the sigma diagonal shift before
        // clustering. During projection, graph oracles must not expose
        // artificial self-loops, and the diagonal shift is only retained as the
        // final ||phi(x)||^2 score constant below, where it does not affect the
        // chosen label.
        let result = coreset_grouped
            .into_par_iter()
            .map(|(indices, weights)| {
                if indices.is_empty() {
                    // Empty cluster: give it an infinite norm so it is never chosen as the default.
                    return (T::infinity(), T::ZERO);
                }

                let indices_set: FxHashSet<V> = indices.iter().copied().collect();
                let index_to_weight: FxHashMap<V, T> = indices
                    .iter()
                    .copied()
                    .zip(weights.iter().copied())
                    .collect();

                let denom: T = weights.iter().copied().sum();
                if denom == T::ZERO {
                    return (T::infinity(), T::ZERO);
                }
                let mut center_norm_sum = T::ZERO;

                indices.iter().for_each(|i| {
                    let vertex_degree =
                        *degree_map.get(i).expect("degree missing for coreset node");

                    let neighbours = coreset_adjacency.get(i).copied().unwrap_or(&[]);
                    let adjacency_contrib = neighbours.iter().fold(T::ZERO, |acc, (j, v)| {
                        if indices_set.contains(j) {
                            let neighbour_degree =
                                *degree_map.get(j).expect("degree missing for neighbour");
                            let value = v.into_scalar() / (vertex_degree * neighbour_degree);
                            acc + value * index_to_weight[i] * index_to_weight[j]
                        } else {
                            acc
                        }
                    });

                    center_norm_sum = center_norm_sum + adjacency_contrib;
                });

                (center_norm_sum / (denom * denom), denom)
            })
            .collect::<Vec<(T, T)>>();

        Ok(result.into_iter().unzip())
    }

    fn label_full_graph_nodes_from_centers(
        &self,
        node_names: &[V],
        query_rows: &[&[(V, Strict<T>)]],
        coreset: &Coreset<V, T>,
        coreset_labels: &[usize],
        degree_map: &FxHashMap<V, T>,
        center_norms: &[T],
        center_denoms: &[T],
        shift: T,
    ) -> Result<(Vec<usize>, Vec<T>)>
    where
        V: Send + Sync,
        T: Send + Sync,
    {
        if node_names.len() != query_rows.len() {
            return Err(anyhow!(
                "full graph labelling expected one row per labelled node; got nodes={}, rows={}",
                node_names.len(),
                query_rows.len()
            ));
        }
        if center_norms.len() != center_denoms.len() {
            return Err(anyhow!(
                "full graph labelling expected matching center norm/denom lengths; got norms={}, denoms={}",
                center_norms.len(),
                center_denoms.len()
            ));
        }

        // Pick the smallest finite center norm; if none are finite, fall back to cluster 0.
        let mut smallest_center_by_norm = 0usize;
        let mut smallest_center_by_norm_value = T::infinity();
        for (idx, norm) in center_norms.iter().enumerate() {
            if norm.is_finite() && *norm < smallest_center_by_norm_value {
                smallest_center_by_norm = idx;
                smallest_center_by_norm_value = *norm;
            }
        }

        let target_info = coreset
            .nodes
            .iter()
            .copied()
            .zip(coreset_labels.iter().copied())
            .zip(coreset.weights.iter())
            .map(|((node, label), weight)| {
                let degree = *degree_map
                    .get(&node)
                    .expect("degree missing for coreset neighbour");
                (
                    node,
                    CoresetProjectionInfo {
                        label,
                        weight: weight.into_scalar(),
                        degree,
                    },
                )
            })
            .collect::<FxHashMap<_, _>>();

        // For each labelled node x, compute normalized inner products
        //   <phi(x), c_l> = denom_l^{-1} sum_{j in C_l} w_j k(x,j)
        // against each represented center, then choose the center minimizing
        //   ||c_l||^2 - 2 <phi(x), c_l>.
        //
        // The final distance adds ||phi(x)||^2, plus the same sigma diagonal
        // score constant. The constant is independent of the candidate center,
        // so it is useful for comparable scores but must not influence label
        // selection.
        let labels_and_distances: (Vec<usize>, Vec<T>) = node_names
            .par_iter()
            .zip(query_rows.par_iter())
            .map(|i| {
                let (i, neighbours) = i;
                let vertex_degree = *degree_map
                    .get(i)
                    .expect("degree missing for node in labelling pass");
                let mut x_to_c_is: FxHashMap<usize, T> = FxHashMap::default();

                neighbours.iter().for_each(|(neighbour, edge_weight)| {
                    let info = target_info
                        .get(neighbour)
                        .expect("intersecting oracle returned a non-coreset neighbour");

                    let inner_prod_with_vertex =
                        edge_weight.into_scalar() / (vertex_degree * info.degree);

                    x_to_c_is
                        .entry(info.label)
                        .and_modify(|e| {
                            *e = *e + info.weight * inner_prod_with_vertex;
                        })
                        .or_insert(info.weight * inner_prod_with_vertex);
                });

                x_to_c_is.iter_mut().for_each(|(k, v)| {
                    let denom = center_denoms[*k];
                    if denom.is_finite() && denom != T::ZERO {
                        *v = *v / denom;
                    } else {
                        *v = T::ZERO;
                    }
                });

                let mut best_center_value = smallest_center_by_norm_value;
                let mut best_center = smallest_center_by_norm;

                x_to_c_is
                    .iter()
                    .filter(|(center, _)| center_norms[**center].is_finite())
                    .for_each(|(center, v)| {
                        let distance = center_norms[*center] - (T::ONE + T::ONE) * *v;
                        if distance < best_center_value {
                            best_center = *center;
                            best_center_value = distance;
                        }
                    });

                (
                    best_center,
                    best_center_value
                        + T::ONE / (vertex_degree * vertex_degree)
                        + shift / vertex_degree,
                )
            })
            .unzip();

        Ok(labels_and_distances)
    }
}
