use std::num::NonZero;
#[cfg(feature = "deep-query-timing")]
use std::time::Instant;

use anyhow::{Result, anyhow};
use faer::sparse::{SparseRowMat, SymbolicSparseRowMat};
use rand::RngExt;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use super::{
    CoresetExtractionTiming, FullGraphLabelTiming, SamplingInfo, TreeLayout, TrialWorkspace,
};
use crate::{
    GraphOracle,
    error::DynamicCoresetError,
    types::{
        Contribution, EdgeWeight, FDelta, FloatScalar, HB, HS, Neighbourhoods, NodeDegree,
        NonStrict, NonStrictCarrierOps, Strict, StrictCarrierOps, TreeIndex,
    },
};

#[cfg(feature = "deep-query-timing")]
macro_rules! timing_start {
    () => {
        Instant::now()
    };
}

#[cfg(not(feature = "deep-query-timing"))]
macro_rules! timing_start {
    () => {
        ()
    };
}

#[cfg(feature = "deep-query-timing")]
macro_rules! timing_add_elapsed {
    ($target:expr, $started:expr) => {
        $target += $started.elapsed();
    };
}

#[cfg(not(feature = "deep-query-timing"))]
macro_rules! timing_add_elapsed {
    ($target:expr, $started:expr) => {{
        let _ = &$target;
        let _ = &$started;
    }};
}

#[cfg(feature = "deep-query-timing")]
macro_rules! timing_add {
    ($target:expr, $value:expr) => {
        $target += $value;
    };
}

#[cfg(not(feature = "deep-query-timing"))]
macro_rules! timing_add {
    ($target:expr, $value:expr) => {{
        let _ = || {
            let _ = &$target;
            let _ = &$value;
        };
    }};
}

pub struct Coreset<V, T> {
    pub nodes: Vec<V>,
    pub node_indices: Vec<TreeIndex>,
    pub weights: Vec<Strict<T>>,
    pub coreset_labels: Option<Vec<usize>>,
    pub coreset_neighbourhood_data: Vec<(V, Strict<T>)>,
    pub coreset_neighbourhood_offsets: Vec<usize>,
}

impl<V, T> Coreset<V, T> {
    fn cached_coreset_neighbourhood_rows(&self) -> Option<Vec<&[(V, Strict<T>)]>> {
        if self.coreset_neighbourhood_offsets.len() != self.nodes.len() + 1 {
            return None;
        }

        let rows = (0..self.nodes.len())
            .map(|idx| {
                let start = self.coreset_neighbourhood_offsets[idx];
                let end = self.coreset_neighbourhood_offsets[idx + 1];
                self.coreset_neighbourhood_data.get(start..end)
            })
            .collect::<Option<Vec<_>>>()?;

        Some(rows)
    }
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

impl<T> SamplingInfo<T>
where
    T: FloatScalar, // T must be a floating point type (either f32 or f64)
    Strict<T>: StrictCarrierOps<Scalar = T> + Copy,
    NonStrict<T>: NonStrictCarrierOps<Scalar = T> + Copy,
{
    pub fn new(
        x_star_idx: TreeIndex,
        sigma: Strict<T>,
        sigma_over_x_star_deg: Strict<T>,
        timestamp: usize,
        total_weight: Strict<T>,
    ) -> Self {
        Self {
            x_star_idx,
            sigma,
            sigma_over_x_star_deg,
            timestamp,
            x_star_seed_set_volume_inv: Strict::<T>::from_positive_scalar(
                T::ONE / total_weight.into_scalar(),
            )
            .expect("total weight must have a positive finite reciprocal"),
            total_contribution_inv: None,
        }
    }
}

impl<const ARITY: usize, V, T> TrialWorkspace<'_, ARITY, V, T>
where
    V: std::hash::Hash + Eq + Clone + Copy,
    T: FloatScalar, // T must be a floating point type (either f32 or f64)
    Strict<T>: StrictCarrierOps<Scalar = T> + Copy,
    NonStrict<T>: NonStrictCarrierOps<Scalar = T> + Copy,
{
    pub(crate) fn initialize_sampling_state(
        &mut self,
        info: &mut SamplingInfo<T>,
        total_weight: Strict<T>,
    ) {
        self.set_seed_idx(info.x_star_idx, info.x_star_idx, info);
        self.query_time.seed_weight[info.x_star_idx] = total_weight.into_scalar();
        self.query_time.seed_weight_epoch[info.x_star_idx] = info.timestamp;
    }

    pub(crate) fn get_seed_idx(&self, node_idx: TreeIndex, info: &SamplingInfo<T>) -> TreeIndex {
        if self.query_time.seed_owner_epoch[node_idx] == info.timestamp {
            self.query_time.seed_owner[node_idx]
        } else {
            info.x_star_idx
        }
    }

    pub(crate) fn set_seed_idx(
        &mut self,
        node_idx: TreeIndex,
        seed_idx: TreeIndex,
        info: &SamplingInfo<T>,
    ) {
        self.query_time.seed_owner[node_idx] = seed_idx;
        self.query_time.seed_owner_epoch[node_idx] = info.timestamp;
    }

    pub(crate) fn get_seed_weight(&self, seed_idx: TreeIndex, info: &SamplingInfo<T>) -> Strict<T> {
        debug_assert_eq!(
            self.query_time.seed_weight_epoch[seed_idx], info.timestamp,
            "seed weight should have been initialized before lookup"
        );
        Strict::<T>::from_positive_scalar(self.query_time.seed_weight[seed_idx])
            .expect("seed weight should be positive and finite")
    }

    pub(crate) fn modify_seed_weight(
        &mut self,
        seed_idx: TreeIndex,
        diff: T,
        info: &mut SamplingInfo<T>,
    ) -> Result<()> {
        let current = if self.query_time.seed_weight_epoch[seed_idx] == info.timestamp {
            self.query_time.seed_weight[seed_idx]
        } else {
            T::ZERO
        };
        let new_weight = Strict::<T>::from_positive_scalar(current + diff)
            .map_err(|e| anyhow!("seed weight update produced invalid weight: {e}"))?;

        self.query_time.seed_weight[seed_idx] = new_weight.into_scalar();
        self.query_time.seed_weight_epoch[seed_idx] = info.timestamp;

        if seed_idx == info.x_star_idx {
            info.x_star_seed_set_volume_inv =
                Strict::<T>::from_positive_scalar(T::ONE / new_weight.into_scalar())
                    .map_err(|e| anyhow!("x_star seed weight reciprocal is invalid: {e}"))?;
        }

        Ok(())
    }

    fn next_old_seed_seen_epoch(&mut self) -> usize {
        self.query_time.old_seed_seen_epoch = self
            .query_time
            .old_seed_seen_epoch
            .checked_add(1)
            .unwrap_or_else(|| {
                self.query_time.old_seed_seen.fill(0);
                1
            });
        self.query_time.old_seed_seen_epoch
    }

    fn push_old_seed_once(
        &mut self,
        old_seed_idx: TreeIndex,
        seen_epoch: usize,
        old_seeds: &mut Vec<TreeIndex>,
    ) {
        if self.query_time.old_seed_seen[old_seed_idx] == seen_epoch {
            return;
        }
        self.query_time.old_seed_seen[old_seed_idx] = seen_epoch;
        old_seeds.push(old_seed_idx);
    }

    fn move_seed_membership(
        &mut self,
        info: &mut SamplingInfo<T>,
        node_idx: TreeIndex,
        new_seed_idx: TreeIndex,
        weight: T,
        old_seeds: &mut Vec<TreeIndex>,
        old_seed_seen_epoch: usize,
        allow_already_in_seed: bool,
        context: &str,
    ) -> Result<()> {
        let old_seed_idx = self.get_seed_idx(node_idx, info);
        if old_seed_idx == new_seed_idx {
            if allow_already_in_seed {
                self.set_seed_idx(node_idx, new_seed_idx, info);
                return Ok(());
            }

            return Err(anyhow!(
                "{context}: node was already assigned to the new seed during seed membership update"
            ));
        }

        self.push_old_seed_once(old_seed_idx, old_seed_seen_epoch, old_seeds);
        self.modify_seed_weight(old_seed_idx, -weight, info)?;
        self.modify_seed_weight(new_seed_idx, weight, info)?;
        self.set_seed_idx(node_idx, new_seed_idx, info);

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

    fn recompute_f_delta_from_updates(&mut self, update_indices: &[TreeIndex]) {
        let timestamp = self.timestamp;
        let total = self.persistent.size.len();
        let query_time = &mut *self.query_time;
        let mut current = std::mem::take(&mut query_time.tree_update_current);
        let mut next = std::mem::take(&mut query_time.tree_update_next);
        let mut seen = std::mem::take(&mut query_time.tree_update_seen);
        let mut seen_epoch = query_time.tree_update_seen_epoch;

        TreeLayout::<ARITY>::apply_updates_from_slice_with_marker(
            total,
            update_indices,
            &mut current,
            &mut next,
            &mut seen,
            &mut seen_epoch,
            |idx| {
                TreeLayout::<ARITY>::one_step_recompute_non_strict_with_timestamp(
                    idx,
                    &mut query_time.f_delta,
                    &query_time.timestamp,
                    timestamp,
                    |_i| FDelta::zero(),
                );
                query_time.timestamp[idx] = timestamp;
            },
        );

        query_time.tree_update_current = current;
        query_time.tree_update_next = next;
        query_time.tree_update_seen = seen;
        query_time.tree_update_seen_epoch = seen_epoch;
    }

    fn recompute_h_from_updates(&mut self, update_indices: &[TreeIndex]) {
        let timestamp = self.timestamp;
        let total = self.persistent.size.len();
        let persistent = self.persistent;
        let query_time = &mut *self.query_time;
        let mut current = std::mem::take(&mut query_time.tree_update_current);
        let mut next = std::mem::take(&mut query_time.tree_update_next);
        let mut seen = std::mem::take(&mut query_time.tree_update_seen);
        let mut seen_epoch = query_time.tree_update_seen_epoch;

        TreeLayout::<ARITY>::apply_updates_from_slice_with_marker(
            total,
            update_indices,
            &mut current,
            &mut next,
            &mut seen,
            &mut seen_epoch,
            |idx| {
                TreeLayout::<ARITY>::one_step_recompute_h_pair_with_timestamp(
                    idx,
                    &mut query_time.h_b,
                    &mut query_time.h_s,
                    &persistent.volume,
                    &query_time.timestamp,
                    timestamp,
                );
                query_time.timestamp[idx] = timestamp;
            },
        );

        query_time.tree_update_current = current;
        query_time.tree_update_next = next;
        query_time.tree_update_seen = seen;
        query_time.tree_update_seen_epoch = seen_epoch;
    }

    fn recompute_h_s_from_updates(&mut self, update_indices: &[TreeIndex]) {
        let timestamp = self.timestamp;
        let total = self.persistent.size.len();
        let query_time = &mut *self.query_time;
        let mut current = std::mem::take(&mut query_time.tree_update_current);
        let mut next = std::mem::take(&mut query_time.tree_update_next);
        let mut seen = std::mem::take(&mut query_time.tree_update_seen);
        let mut seen_epoch = query_time.tree_update_seen_epoch;

        TreeLayout::<ARITY>::apply_updates_from_slice_with_marker(
            total,
            update_indices,
            &mut current,
            &mut next,
            &mut seen,
            &mut seen_epoch,
            |idx| {
                TreeLayout::<ARITY>::one_step_recompute_non_strict_with_timestamp(
                    idx,
                    &mut query_time.h_s,
                    &query_time.timestamp,
                    timestamp,
                    |_i| HS::zero(),
                );
                query_time.timestamp[idx] = timestamp;
            },
        );

        query_time.tree_update_current = current;
        query_time.tree_update_next = next;
        query_time.tree_update_seen = seen;
        query_time.tree_update_seen_epoch = seen_epoch;
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
        let mut timing = CoresetExtractionTiming::default();
        self.extract_coreset_trial_timed(
            graph_oracle,
            sigma,
            x_star,
            x_star_degree,
            coreset_size,
            sampling_seeds,
            rng,
            &mut timing,
        )
    }

    pub(crate) fn extract_coreset_trial_timed<G, E>(
        &mut self,
        graph_oracle: &mut G,
        sigma: Strict<T>,
        x_star: V,
        x_star_degree: NodeDegree<T>,
        coreset_size: NonZero<usize>,
        sampling_seeds: NonZero<usize>,
        rng: &mut impl rand::Rng,
        timing: &mut CoresetExtractionTiming,
    ) -> Result<Coreset<V, T>>
    where
        G: GraphOracle<V, T, E> + ?Sized + Send,
        E: std::fmt::Display,
    {
        let setup_started = timing_start!();
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
        let x_star_idx = *self
            .node_to_tree_map
            .get(&x_star)
            .ok_or_else(|| anyhow!("x_star was missing from the tree"))?;

        let mut info = SamplingInfo::new(
            x_star_idx,
            sigma,
            sigma_over_x_star_deg,
            timestamp,
            self.persistent.volume[0].0,
        );
        self.initialize_sampling_state(&mut info, self.persistent.volume[0].0);
        timing_add_elapsed!(timing.setup, setup_started);

        // sanity: leaves marked deleted should carry zero volume
        // self.assert_zero_volume_for_empty_leaves(&info);

        // first we add x_star:
        let initial_repairs_started = timing_start!();
        self.repair_timed(x_star, &mut info, graph_oracle, timing)?;
        timing_add!(timing.initial_repair_calls, 1);

        // Now we sample a node uniformly:
        let tree_size = self.persistent.size.len();
        let num_leaves = self.node_to_tree_map.len();

        let uniform_idx = TreeIndex(rng.random_range(tree_size - num_leaves..tree_size));
        let uniform_node = *self.tree_to_node_map.get(&uniform_idx).unwrap();
        self.repair_timed(uniform_node, &mut info, graph_oracle, timing)?;
        timing_add!(timing.initial_repair_calls, 1);
        timing_add_elapsed!(timing.initial_repairs, initial_repairs_started);

        let remaining_seeds = sampling_seeds.get().saturating_sub(2);
        for i in 0..remaining_seeds {
            // Sample a point according to f:
            let seed_sample_started = timing_start!();
            let (node, _, _) = self.sample(&info, rng).map_err(|e| {
                anyhow!("failed sampling seed {} of {}: {e}", i + 1, remaining_seeds)
            })?;
            timing_add_elapsed!(timing.seed_sampling, seed_sample_started);
            timing_add!(timing.seed_samples, 1);

            let seed_repair_started = timing_start!();
            self.repair_timed(node, &mut info, graph_oracle, timing)?;
            timing_add_elapsed!(timing.seed_repairs, seed_repair_started);
            timing_add!(timing.seed_repair_calls, 1);
        }

        // populate total_contribution_inv
        let total_contribution_started = timing_start!();
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
        timing_add_elapsed!(timing.total_contribution, total_contribution_started);

        let coreset_size_f =
            T::from(coreset_size.get()).expect("coreset size should convert to scalar");
        let smoothed_sampling_started = timing_start!();
        let coreset_samples = (0..coreset_size.get())
            .map(|_| {
                let (node, idx, prob) = self.sample_smoothed(&info, rng).unwrap();
                let node_deg = self.persistent.volume[idx].into_scalar();
                let weight = node_deg / (prob.into_scalar() * coreset_size_f);
                (node, idx, weight)
            })
            .collect::<Vec<_>>();
        timing_add_elapsed!(timing.smoothed_sampling, smoothed_sampling_started);
        timing_add!(timing.smoothed_samples, coreset_samples.len());

        // Now we deduplicate the coreset:
        let dedup_started = timing_start!();
        let mut coreset: FxHashMap<(V, TreeIndex), T> = FxHashMap::default();
        for (v, index, weight) in coreset_samples {
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
        timing_add!(timing.dedup_unique_nodes, unique_vs.len());
        timing_add_elapsed!(timing.deduplication, dedup_started);

        Ok(Coreset {
            nodes: unique_vs,
            node_indices: unique_indices,
            weights,
            coreset_labels: None,
            coreset_neighbourhood_data: Vec::new(),
            coreset_neighbourhood_offsets: Vec::new(),
        })
    }

    pub fn repair<G, E>(
        &mut self,
        point_added: V,
        info: &mut SamplingInfo<T>,
        graph_oracle: &mut G,
    ) -> Result<()>
    where
        G: GraphOracle<V, T, E> + ?Sized,
        E: std::fmt::Display,
    {
        let mut timing = CoresetExtractionTiming::default();
        self.repair_timed(point_added, info, graph_oracle, &mut timing)
    }

    fn repair_timed<G, E>(
        &mut self,
        point_added: V,
        info: &mut SamplingInfo<T>,
        graph_oracle: &mut G,
        timing: &mut CoresetExtractionTiming,
    ) -> Result<()>
    where
        G: GraphOracle<V, T, E> + ?Sized,
        E: std::fmt::Display,
    {
        timing_add!(timing.repair_calls, 1);
        // We implicitly add the point to the init set, update its neighbours,
        // and seed maps / seed weights.
        let point_added_index = *self.node_to_tree_map.get(&point_added).unwrap();

        let point_added_volume = self.persistent.volume[point_added_index];
        let point_added_degree = NodeDegree::from_scalar(point_added_volume.into_scalar())
            .map_err(|e| anyhow!("point-added volume could not be used as a node degree: {e}"))?;
        let point_added_weight = point_added_volume.into_scalar();
        let mut old_seeds = Vec::new();
        let old_seed_seen_epoch = self.next_old_seed_seen_epoch();

        let point_seed_move_started = timing_start!();
        self.move_seed_membership(
            info,
            point_added_index,
            point_added_index,
            point_added_weight,
            &mut old_seeds,
            old_seed_seen_epoch,
            true,
            "repair point seed move",
        )?;
        timing_add_elapsed!(timing.repair_point_seed_move, point_seed_move_started);

        // Zero this point's f contribution by matching f_delta to f_b.
        let point_f_delta_started = timing_start!();
        let f_b = self.f_b(point_added_index, info);
        self.query_time.f_delta[point_added_index] = FDelta::from_scalar(f_b.into_scalar())
            .map_err(|e| anyhow!("base contribution could not be stored as f_delta: {e}"))?;
        self.query_time.timestamp[point_added_index] = info.timestamp;
        self.recompute_f_delta_to_root(point_added_index);
        timing_add_elapsed!(timing.repair_point_f_delta, point_f_delta_started);

        let point_query = [point_added];
        let point_lookup_started = timing_start!();
        let neighbours =
            lookup_single_graph_neighbourhood(graph_oracle, &point_query, "repair point lookup")?;
        timing_add_elapsed!(timing.repair_point_lookup, point_lookup_started);

        let mut filtered_neighbour_indices = Vec::with_capacity(neighbours.len());
        let mut f_delta_update_nodes = Vec::with_capacity(neighbours.len());

        let neighbour_scan_started = timing_start!();
        timing_add!(timing.repair_neighbours_scanned, neighbours.len());
        for (neighbour, edge_weight) in neighbours.iter() {
            let neighbour_lookup_started = timing_start!();
            let neighbour_idx = *self.node_to_tree_map.get(neighbour).ok_or_else(|| {
                anyhow!("repair point lookup returned a neighbour missing from the tree")
            })?;
            timing_add_elapsed!(timing.repair_neighbour_lookup, neighbour_lookup_started);

            let neighbour_compare_started = timing_start!();
            let weighted_distance_to_point_added =
                Self::weighted_kernel_distance(point_added_degree, EdgeWeight::new(*edge_weight));
            let current_contribution = self.f(neighbour_idx, info);
            timing_add_elapsed!(timing.repair_neighbour_compare, neighbour_compare_started);

            if weighted_distance_to_point_added < current_contribution {
                // Neighbour is now closer to this point.
                filtered_neighbour_indices.push(neighbour_idx);
                timing_add!(timing.repair_neighbours_improved, 1);

                let f_delta_write_started = timing_start!();
                let new_f_delta_term = (self.f_b(neighbour_idx, info).into_scalar()
                    - weighted_distance_to_point_added.into_scalar())
                .max(T::ZERO);
                self.query_time.f_delta[neighbour_idx] = FDelta::from_scalar(new_f_delta_term)
                    .map_err(|e| {
                        anyhow!("updated f_delta term was not non-negative finite: {e}")
                    })?;
                self.query_time.timestamp[neighbour_idx] = info.timestamp;
                f_delta_update_nodes.push(neighbour_idx);
                timing_add_elapsed!(timing.repair_neighbour_f_delta_write, f_delta_write_started);

                let seed_move_started = timing_start!();
                let neighbour_volume = self.persistent.volume[neighbour_idx];
                self.move_seed_membership(
                    info,
                    neighbour_idx,
                    point_added_index,
                    neighbour_volume.into_scalar(),
                    &mut old_seeds,
                    old_seed_seen_epoch,
                    // Nodes can already belong to this seed set. In
                    // particular, unseen nodes default to x_star, so repairing
                    // x_star should not try to debit and credit the same seed.
                    true,
                    "repair neighbour seed move",
                )?;
                timing_add_elapsed!(timing.repair_neighbour_seed_move, seed_move_started);
            }
        }
        let f_delta_recompute_started = timing_start!();
        self.recompute_f_delta_from_updates(&f_delta_update_nodes);
        timing_add_elapsed!(
            timing.repair_neighbour_f_delta_recompute,
            f_delta_recompute_started
        );
        timing_add_elapsed!(timing.repair_neighbour_scan, neighbour_scan_started);

        let seed_weight = self.get_seed_weight(point_added_index, info);
        let seed_weight_scalar = seed_weight.into_scalar();
        debug_assert!(
            seed_weight_scalar.is_finite() && seed_weight_scalar > T::ZERO,
            "seed weight must be non-zero for h_s update"
        );

        let new_seed_h_update_started = timing_start!();
        let new_seed_h_write_started = timing_start!();
        let mut h_update_nodes = Vec::with_capacity(filtered_neighbour_indices.len() + 1);
        for z_idx in filtered_neighbour_indices
            .into_iter()
            .chain([point_added_index])
        {
            self.query_time.h_b[z_idx] = HB::zero();
            let deg_z = self.persistent.volume[z_idx].into_scalar();
            self.query_time.h_s[z_idx] =
                HS::from_scalar(deg_z / seed_weight_scalar).map_err(|e| {
                    anyhow!("h_s update for new seed set was not non-negative finite: {e}")
                })?;

            self.query_time.timestamp[z_idx] = info.timestamp;
            h_update_nodes.push(z_idx);
        }
        timing_add!(timing.repair_new_seed_h_update_nodes, h_update_nodes.len());
        timing_add_elapsed!(timing.repair_new_seed_h_write, new_seed_h_write_started);

        let new_seed_h_recompute_started = timing_start!();
        self.recompute_h_from_updates(&h_update_nodes);
        timing_add_elapsed!(
            timing.repair_new_seed_h_recompute,
            new_seed_h_recompute_started
        );
        timing_add_elapsed!(timing.repair_new_seed_h_update, new_seed_h_update_started);

        // Update h_s for nodes in old seed sets whose seed-set weights changed, except x^*.
        let timestamp = info.timestamp;

        let old_seed_prepare_started = timing_start!();
        let old_seeds_and_weights = old_seeds
            .into_iter()
            .filter(|s| *s != info.x_star_idx)
            .map(|s| (s, self.get_seed_weight(s, info)))
            .collect::<Vec<_>>();
        timing_add!(timing.repair_old_seed_count, old_seeds_and_weights.len());

        if old_seeds_and_weights.is_empty() {
            timing_add_elapsed!(timing.repair_old_seed_prepare, old_seed_prepare_started);
            return Ok(());
        }

        let mut h_s_update_nodes = Vec::new();
        let old_seed_nodes = old_seeds_and_weights
            .iter()
            .map(|(s, _)| {
                self.tree_to_node_map
                    .get(s)
                    .copied()
                    .ok_or_else(|| anyhow!("old seed was missing from the tree"))
            })
            .collect::<Result<Vec<_>>>()?;
        timing_add_elapsed!(timing.repair_old_seed_prepare, old_seed_prepare_started);

        let old_seed_lookup_started = timing_start!();
        let old_seed_neighbour_batches =
            lookup_graph_neighbourhoods(graph_oracle, &old_seed_nodes, "old seed lookup")?;
        timing_add_elapsed!(timing.repair_old_seed_lookup, old_seed_lookup_started);

        let old_seed_rescale_started = timing_start!();
        for ((s, seed_weight), neighbours) in old_seeds_and_weights
            .into_iter()
            .zip(old_seed_neighbour_batches.iter())
        {
            timing_add!(timing.repair_old_seed_neighbours_scanned, neighbours.len());
            let seed_weight_scalar = seed_weight.into_scalar();
            debug_assert!(
                seed_weight_scalar.is_finite() && seed_weight_scalar > T::ZERO,
                "old seed weight must be non-zero for h_s rescale"
            );

            for (z, _) in neighbours.iter() {
                let z_idx = *self.node_to_tree_map.get(z).ok_or_else(|| {
                    anyhow!("old seed lookup returned a neighbour missing from the tree")
                })?;
                if self.get_seed_idx(z_idx, info) != s {
                    continue;
                }

                timing_add!(timing.repair_old_seed_neighbours_rescaled, 1);
                let deg_z = self.persistent.volume[z_idx].into_scalar();
                self.query_time.h_s[z_idx] =
                    HS::from_scalar(deg_z / seed_weight_scalar).map_err(|e| {
                        anyhow!("h_s rescale for old seed set was not non-negative finite: {e}")
                    })?;
                self.query_time.timestamp[z_idx] = timestamp;
                h_s_update_nodes.push(z_idx);
            }
        }
        timing_add_elapsed!(timing.repair_old_seed_rescale, old_seed_rescale_started);
        timing_add!(
            timing.repair_old_seed_h_update_nodes,
            h_s_update_nodes.len()
        );

        let h_s_recompute_started = timing_start!();
        self.recompute_h_s_from_updates(&h_s_update_nodes);
        timing_add_elapsed!(timing.repair_old_seed_h_recompute, h_s_recompute_started);

        Ok(())
    }
    pub fn build_coreset_graph<C, E>(
        &self,
        coreset: &mut Coreset<V, T>,
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

        coreset.coreset_neighbourhood_data.clear();
        coreset.coreset_neighbourhood_offsets.clear();
        coreset.coreset_neighbourhood_offsets.reserve(n + 1);
        coreset.coreset_neighbourhood_offsets.push(0);

        let mut coreset_neighbourhood_index_data = Vec::<(usize, Strict<T>)>::new();
        let mut current_row = 0usize;
        let mut visitor_error = None::<String>;
        let visited_edges = coreset_oracle
            .visit_coreset_neighbourhoods_with_target_indices(
                coreset.nodes.as_slice(),
                |row_idx, target_idx, node, weight| {
                    if visitor_error.is_some() {
                        return;
                    }
                    if row_idx >= n {
                        visitor_error = Some(format!(
                            "coreset graph lookup returned row index {row_idx} for {n} rows"
                        ));
                        return;
                    }
                    if target_idx >= n {
                        visitor_error = Some(format!(
                            "coreset graph lookup returned target index {target_idx} for {n} targets"
                        ));
                        return;
                    }
                    if row_idx < current_row {
                        visitor_error = Some(format!(
                            "coreset graph lookup visited row {row_idx} after row {current_row}"
                        ));
                        return;
                    }

                    while current_row < row_idx {
                        coreset
                            .coreset_neighbourhood_offsets
                            .push(coreset.coreset_neighbourhood_data.len());
                        current_row += 1;
                    }

                    coreset.coreset_neighbourhood_data.push((node, weight));
                    coreset_neighbourhood_index_data.push((target_idx, weight));
                },
            )
            .map_err(|e| anyhow!("coreset graph lookup: {e}"))?;

        if let Some(error) = visitor_error {
            return Err(anyhow!("{error}"));
        }
        debug_assert_eq!(visited_edges, coreset_neighbourhood_index_data.len());
        debug_assert_eq!(
            coreset.coreset_neighbourhood_data.len(),
            coreset_neighbourhood_index_data.len()
        );

        while coreset.coreset_neighbourhood_offsets.len() < n + 1 {
            coreset
                .coreset_neighbourhood_offsets
                .push(coreset.coreset_neighbourhood_data.len());
        }

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
        for i in 0..n {
            let weight_degree_inv_i = weight_degree_inv[i];
            let mut no_diag = true;
            let row_start = coreset.coreset_neighbourhood_offsets[i];
            let row_end = coreset.coreset_neighbourhood_offsets[i + 1];
            let mut row_entries = coreset_neighbourhood_index_data[row_start..row_end]
                .iter()
                .map(|(coreset_j, edge_weight)| {
                    let edge_weight = edge_weight.into_scalar();
                    if i == *coreset_j {
                        no_diag = false;
                        (
                            *coreset_j,
                            edge_weight * weight_degree_inv_i * weight_degree_inv_i
                                + sigma.into_scalar()
                                    * coreset.weights[i].into_scalar()
                                    * weight_degree_inv_i,
                        )
                    } else {
                        (
                            *coreset_j,
                            edge_weight * weight_degree_inv_i * weight_degree_inv[*coreset_j],
                        )
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
        let mut timing = FullGraphLabelTiming::default();
        self.rust_label_full_graph_timed(
            coreset,
            num_clusters,
            graph_oracle,
            nodes,
            sigma,
            &mut timing,
        )
    }

    pub(crate) fn rust_label_full_graph_timed<G, E>(
        &self,
        coreset: &Coreset<V, T>,
        num_clusters: usize,
        graph_oracle: &mut G,
        nodes: &[V],
        sigma: Strict<T>,
        timing: &mut FullGraphLabelTiming,
    ) -> Result<(Vec<V>, Vec<usize>, Vec<T>)>
    where
        G: GraphOracle<V, T, E> + ?Sized,
        E: std::fmt::Display,
        V: Send + Sync,
        T: Send + Sync,
    {
        let total_started = timing_start!();
        *timing = FullGraphLabelTiming::default();

        let setup_started = timing_start!();
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
        timing_add!(timing.labelled_nodes, node_names.len());
        timing_add!(timing.coreset_nodes, coreset.nodes.len());

        // Union of all nodes we will touch: labelled nodes plus coreset nodes.
        // This deduplicates the graph-wide neighbourhood batch lookup.
        let mut all_nodes_set: FxHashSet<V> = node_names.iter().copied().collect();
        all_nodes_set.extend(coreset.nodes.iter().copied());
        let all_nodes: Vec<V> = all_nodes_set.iter().copied().collect();
        timing_add!(timing.degree_nodes, all_nodes.len());
        timing_add_elapsed!(timing.setup, setup_started);

        // Precompute degree lookups to avoid touching the main data structures
        // inside parallel labelling loops.
        let degree_lookup_started = timing_start!();
        let mut degree_map = FxHashMap::<V, T>::default();
        for node in all_nodes.iter().copied() {
            let idx = *self
                .node_to_tree_map
                .get(&node)
                .ok_or_else(|| anyhow!("full graph labelling node was missing from the tree"))?;
            degree_map.insert(node, self.persistent.volume[idx].into_scalar());
        }
        timing_add_elapsed!(timing.degree_lookup, degree_lookup_started);

        let (center_norms, center_denoms) =
            if let Some(coreset_rows) = coreset.cached_coreset_neighbourhood_rows() {
                let coreset_row_collect_started = timing_start!();
                timing_add!(
                    timing.coreset_lookup_edges,
                    coreset_rows.iter().map(|row| row.len()).sum::<usize>()
                );
                timing_add_elapsed!(timing.coreset_row_collect, coreset_row_collect_started);

                let center_stats_started = timing_start!();
                let center_stats = self.compute_full_graph_center_stats(
                    coreset,
                    num_clusters,
                    coreset_labels,
                    &degree_map,
                    &coreset_rows,
                )?;
                timing_add_elapsed!(timing.center_stats, center_stats_started);
                center_stats
            } else {
                let coreset_lookup_started = timing_start!();
                let coreset_neighbourhoods = lookup_graph_neighbourhoods_intersecting(
                    graph_oracle,
                    coreset.nodes.as_slice(),
                    coreset.nodes.as_slice(),
                    "full graph coreset lookup",
                )?;
                timing_add_elapsed!(timing.coreset_lookup, coreset_lookup_started);

                let coreset_row_collect_started = timing_start!();
                let coreset_rows = coreset_neighbourhoods.iter().collect::<Vec<_>>();
                timing_add!(
                    timing.coreset_lookup_edges,
                    coreset_rows.iter().map(|row| row.len()).sum::<usize>()
                );
                timing_add_elapsed!(timing.coreset_row_collect, coreset_row_collect_started);

                let center_stats_started = timing_start!();
                let center_stats = self.compute_full_graph_center_stats(
                    coreset,
                    num_clusters,
                    coreset_labels,
                    &degree_map,
                    &coreset_rows,
                )?;
                timing_add_elapsed!(timing.center_stats, center_stats_started);
                center_stats
            };

        let labels_and_distances = self.label_full_graph_nodes_from_centers_timed(
            graph_oracle,
            node_names.as_slice(),
            coreset,
            coreset_labels,
            &degree_map,
            &center_norms,
            &center_denoms,
            shift,
            timing,
        )?;

        timing_add_elapsed!(timing.total, total_started);
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

    fn label_full_graph_nodes_from_centers_timed<G, E>(
        &self,
        graph_oracle: &mut G,
        node_names: &[V],
        coreset: &Coreset<V, T>,
        coreset_labels: &[usize],
        degree_map: &FxHashMap<V, T>,
        center_norms: &[T],
        center_denoms: &[T],
        shift: T,
        timing: &mut FullGraphLabelTiming,
    ) -> Result<(Vec<usize>, Vec<T>)>
    where
        G: GraphOracle<V, T, E> + ?Sized,
        E: std::fmt::Display,
        V: Send + Sync,
        T: Send + Sync,
    {
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

        let num_centers = center_norms.len();
        let target_info_started = timing_start!();
        let mut target_labels = Vec::with_capacity(coreset.nodes.len());
        let mut target_weights = Vec::with_capacity(coreset.nodes.len());
        let mut target_degrees = Vec::with_capacity(coreset.nodes.len());
        for ((node, label), weight) in coreset
            .nodes
            .iter()
            .copied()
            .zip(coreset_labels.iter().copied())
            .zip(coreset.weights.iter())
        {
            if label >= num_centers {
                return Err(anyhow!(
                    "coreset label {} was outside the cluster range 0..{}",
                    label,
                    num_centers
                ));
            }
            let degree = *degree_map
                .get(&node)
                .expect("degree missing for coreset neighbour");
            target_labels.push(label);
            target_weights.push(weight.into_scalar());
            target_degrees.push(degree);
        }
        timing_add_elapsed!(timing.target_info, target_info_started);

        let labelled_degrees = node_names
            .iter()
            .map(|node| {
                degree_map
                    .get(node)
                    .copied()
                    .ok_or_else(|| anyhow!("degree missing for node in labelling pass"))
            })
            .collect::<Result<Vec<_>>>()?;

        // For each labelled node x, compute normalized inner products
        //   <phi(x), c_l> = denom_l^{-1} sum_{j in C_l} w_j k(x,j)
        // against each represented center, then choose the center minimizing
        //   ||c_l||^2 - 2 <phi(x), c_l>.
        //
        // The final distance adds ||phi(x)||^2, plus the same sigma diagonal
        // score constant. The constant is independent of the candidate center,
        // so it is useful for comparable scores but must not influence label
        // selection.
        let mut center_scores = vec![T::ZERO; node_names.len() * num_centers];

        let query_lookup_started = timing_start!();
        let query_lookup_edges = graph_oracle
            .visit_graph_neighbourhoods_intersecting_with_target_indices(
                node_names,
                coreset.nodes.as_slice(),
                |row_idx, target_idx, _neighbour, edge_weight| {
                    debug_assert!(row_idx < node_names.len());
                    debug_assert!(target_idx < target_labels.len());

                    let vertex_degree = labelled_degrees[row_idx];
                    let inner_prod_with_vertex =
                        edge_weight.into_scalar() / (vertex_degree * target_degrees[target_idx]);
                    let label = target_labels[target_idx];
                    let offset = row_idx * num_centers + label;
                    center_scores[offset] =
                        center_scores[offset] + target_weights[target_idx] * inner_prod_with_vertex;
                },
            )
            .map_err(|e| anyhow!("full graph query lookup: {e}"))?;
        timing_add_elapsed!(timing.query_lookup, query_lookup_started);
        timing_add!(timing.query_lookup_edges, query_lookup_edges);

        let label_nodes_started = timing_start!();
        let labels_and_distances: (Vec<usize>, Vec<T>) = (0..node_names.len())
            .into_par_iter()
            .map(|row_idx| {
                let vertex_degree = labelled_degrees[row_idx];
                let row_start = row_idx * num_centers;
                let row_scores = &center_scores[row_start..row_start + num_centers];
                let mut best_center_value = smallest_center_by_norm_value;
                let mut best_center = smallest_center_by_norm;

                for center in 0..num_centers {
                    if !center_norms[center].is_finite() {
                        continue;
                    }
                    let denom = center_denoms[center];
                    let inner_prod_with_center = if denom.is_finite() && denom != T::ZERO {
                        row_scores[center] / denom
                    } else {
                        T::ZERO
                    };
                    let distance =
                        center_norms[center] - (T::ONE + T::ONE) * inner_prod_with_center;
                    if distance < best_center_value {
                        best_center = center;
                        best_center_value = distance;
                    }
                }

                (
                    best_center,
                    best_center_value
                        + T::ONE / (vertex_degree * vertex_degree)
                        + shift / vertex_degree,
                )
            })
            .unzip();
        timing_add_elapsed!(timing.label_nodes, label_nodes_started);

        Ok(labels_and_distances)
    }
}
