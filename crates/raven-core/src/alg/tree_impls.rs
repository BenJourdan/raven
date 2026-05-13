use anyhow::{Result, anyhow};
use rustc_hash::FxHashSet;
use std::{num::NonZero, ops::Range};

use super::DynamicClustering;
use crate::types::{
    FDelta, FloatScalar, HB, HS, NodeDegree, NonStrict, NonStrictCarrierOps, Strict,
    StrictCarrierOps, TreeIndex, Volume,
};

impl<const ARITY: usize, V, T> DynamicClustering<ARITY, V, T>
where
    V: std::hash::Hash + Eq + Clone + Copy,
    T: FloatScalar, // T must be a floating point type (either f32 or f64)
    Strict<T>: StrictCarrierOps<Scalar = T> + Copy,
    NonStrict<T>: NonStrictCarrierOps<Scalar = T> + Copy,
{
    pub fn parent_index(&self, child_index: TreeIndex) -> Option<TreeIndex> {
        if child_index.0 == 0 {
            None
        } else {
            Some((child_index - TreeIndex(1)) / ARITY)
        }
    }

    pub fn child_index(&self, parent_index: TreeIndex, child_index: usize) -> TreeIndex {
        TreeIndex(parent_index.0 * ARITY + 1 + child_index)
    }

    pub fn which_child(child_idex: TreeIndex) -> usize {
        (child_idex.0 - 1) % ARITY
    }

    pub fn num_leaves(&self) -> usize {
        let n = self.node_to_tree_map.len();
        debug_assert!(
            self.tree_to_node_map.len() == n && self.degrees.len() == n,
            "Inconsistent number of nodes in DynamicClustering data structures"
        );
        n
    }

    pub fn num_internal_nodes(&self) -> usize {
        Self::internal_count_for_leaves(self.num_leaves())
    }

    pub fn num_internal_nodes_from_leaves(num_leaves: usize) -> usize {
        Self::internal_count_for_leaves(num_leaves)
    }

    pub(crate) fn internal_count_for_leaves(num_leaves: usize) -> usize {
        debug_assert!(ARITY > 1, "ARITY must be at least 2");

        // In a d-ary tree with n leaves and every internal node having at most d
        // children, the compact heap layout needs
        //   I(n) = ceil((n - 1) / (d - 1))
        // internal nodes for n > 1. The integer expression below is the same
        // quantity, written to avoid floating point arithmetic.
        if num_leaves <= 1 {
            0
        } else {
            (num_leaves - 2) / (ARITY - 1) + 1
        }
    }

    pub fn num_total_nodes(&self) -> usize {
        self.num_leaves() + self.num_internal_nodes()
    }

    pub fn num_total_nodes_from_leaves(num_leaves: usize) -> usize {
        Self::total_count_for_leaves(num_leaves)
    }

    pub(crate) fn total_count_for_leaves(num_leaves: usize) -> usize {
        num_leaves + Self::internal_count_for_leaves(num_leaves)
    }

    pub(crate) fn leaf_start_for_leaves(num_leaves: usize) -> usize {
        Self::internal_count_for_leaves(num_leaves)
    }

    pub(crate) fn leaf_range_for_leaves(num_leaves: usize) -> Range<usize> {
        let leaf_start = Self::leaf_start_for_leaves(num_leaves);
        leaf_start..leaf_start + num_leaves
    }

    pub(crate) fn is_leaf_index_for_leaves(idx: TreeIndex, num_leaves: usize) -> bool {
        Self::leaf_range_for_leaves(num_leaves).contains(&idx.0)
    }

    fn rebuild_from_leaves(&mut self, leaf_start: usize, leaf_end: usize) {
        // Precondition: [leaf_start, leaf_end) is a contiguous block of leaves
        // (possibly spanning the last 2 levels).

        let total = self.num_total_nodes();

        if leaf_start >= leaf_end || leaf_start == 0 {
            return;
        }

        let sum_non_empty_volumes = |volumes: &[Volume<T>]| -> Volume<T> {
            debug_assert!(
                !volumes.is_empty(),
                "volume aggregation requires at least one child"
            );

            let total: T = volumes.iter().map(|volume| volume.into_scalar()).sum();
            Volume::from_scalar(total)
                .expect("sum of positive finite volumes was zero or overflowed")
        };

        // --- compute bottom-level start index ---

        let n = total as f64;
        let d = ARITY as f64;

        // For a full d-ary tree:
        // N = (d^(h+1) - 1)/(d-1) -> h = log_d((d-1)N + 1) - 1
        // We floor h here; for a complete tree this gives the deepest *full* level,
        // and the "bottom" level is either h or h+1, but the boundary
        // (first index of deepest level) is still:
        //   l_bottom_start = (d^h - 1)/(d-1)
        let h = (((d - 1.0) * n + 1.0).log(d)).floor() as u32 - 1;
        let l_bottom_start = (ARITY.pow(h) - 1) / (ARITY - 1);

        // Split into:
        //  - bottom_range: indices on the deepest level
        //  - top_range: indices on the level above
        //
        // Either range may be empty.
        let mut bottom_range = leaf_start.max(l_bottom_start)..leaf_end.max(l_bottom_start);
        let mut top_range = leaf_start.min(l_bottom_start)..leaf_end.min(l_bottom_start);

        // Invariants we maintain:
        //  - bottom_range and top_range are each either empty or entirely one level.
        //  - bottom_range (if non-empty) is strictly deeper than top_range (if non-empty).

        while !bottom_range.is_empty() || !top_range.is_empty() {
            // --- 1. process bottom_range (deepest level) ---

            if !bottom_range.is_empty() {
                let child_start = bottom_range.start;
                let child_end = bottom_range.end;

                if child_start == 0 {
                    // We've reached the root.
                    bottom_range = 0..0;
                    continue;
                }

                // Compute parent range for this level
                let parent_start = self.parent_index(TreeIndex(child_start)).unwrap().0;
                let parent_end = self.parent_index(TreeIndex(child_end - 1)).unwrap().0 + 1;

                // Update parents from their children.
                // This loop is *per-level* and parallelisable.
                for p in parent_start..parent_end {
                    let p_idx = TreeIndex(p);
                    let c_start = self.child_index(p_idx, 0).0;
                    let c_end = (c_start + ARITY).min(total);

                    let size: NonZero<usize> = self.tree_data.size[c_start..c_end]
                        .iter()
                        .map(|&x| x.get())
                        .sum::<usize>()
                        .try_into()
                        .expect("Size should be nonzero since parent has at least one child");
                    let volume = sum_non_empty_volumes(&self.tree_data.volume[c_start..c_end]);

                    self.tree_data.size[p] = size;
                    self.tree_data.volume[p] = volume;
                }

                // Now our "bottom" frontier moves up one level
                bottom_range = parent_start..parent_end;
            }

            // --- 2. possibly merge with top_range ---

            if !bottom_range.is_empty() && !top_range.is_empty() {
                // If the new bottom_range overlaps with the existing top_range,
                // they are now on the same level: merge them.
                if bottom_range.end >= top_range.start && bottom_range.start <= top_range.end {
                    let new_start = bottom_range.start.min(top_range.start);
                    let new_end = bottom_range.end.max(top_range.end);
                    top_range = new_start..new_end;
                    bottom_range = 0..0; // empty
                }
            }

            // --- 3. process top_range (next level up) ---

            if !top_range.is_empty() {
                let child_start = top_range.start;
                let child_end = top_range.end;

                if child_start == 0 {
                    // We're at the root: update it directly and finish.
                    let p_idx = TreeIndex(0);
                    let c_start = self.child_index(p_idx, 0).0;
                    let c_end = (c_start + ARITY).min(total);

                    let size: NonZero<usize> = self.tree_data.size[c_start..c_end]
                        .iter()
                        .map(|&x| x.get())
                        .sum::<usize>()
                        .try_into()
                        .expect("Size should be nonzero since parent has at least one child");

                    let volume = sum_non_empty_volumes(&self.tree_data.volume[c_start..c_end]);

                    self.tree_data.size[0] = size;
                    self.tree_data.volume[0] = volume;
                    break;
                }

                let parent_start = self.parent_index(TreeIndex(child_start)).unwrap().0;
                let parent_end = self.parent_index(TreeIndex(child_end - 1)).unwrap().0 + 1;

                for p in parent_start..parent_end {
                    let p_idx = TreeIndex(p);
                    let c_start = self.child_index(p_idx, 0).0;
                    let c_end = (c_start + ARITY).min(total);

                    let size: NonZero<usize> = self.tree_data.size[c_start..c_end]
                        .iter()
                        .map(|&x| x.get())
                        .sum::<usize>()
                        .try_into()
                        .expect("Size should be nonzero since parent has at least one child");

                    let volume = sum_non_empty_volumes(&self.tree_data.volume[c_start..c_end]);

                    self.tree_data.size[p] = size;
                    self.tree_data.volume[p] = volume;
                }

                // Move top frontier up one level
                top_range = parent_start..parent_end;
            }
        }

        // Ensure the root reflects the final child aggregates.
        if total > 1 {
            let root_idx = TreeIndex(0);
            let child_start = self.child_index(root_idx, 0).0;
            if child_start < total {
                let child_end = (child_start + ARITY).min(total);
                let size: NonZero<usize> = self.tree_data.size[child_start..child_end]
                    .iter()
                    .map(|&x| x.get())
                    .sum::<usize>()
                    .try_into()
                    .expect("Size should be nonzero since parent has at least one child");
                let volume = sum_non_empty_volumes(&self.tree_data.volume[child_start..child_end]);
                self.tree_data.size[0] = size;
                self.tree_data.volume[0] = volume;
            }
        }
    }

    pub(crate) fn delete_nodes_compact(
        &mut self,
        deleted: &[V],
        touched: &mut FxHashSet<TreeIndex>,
    ) -> Result<()> {
        if deleted.is_empty() {
            return Ok(());
        }

        let old_leaves = self.num_leaves();
        let old_internal = self.num_internal_nodes();
        let old_total = old_internal + old_leaves;

        let mut deleted_pairs = Vec::with_capacity(deleted.len());
        let mut deleted_indices = FxHashSet::default();

        for &node in deleted {
            let idx = *self
                .node_to_tree_map
                .get(&node)
                .ok_or_else(|| anyhow!("deleted node was missing from the tree"))?;

            if !Self::is_leaf_index_for_leaves(idx, old_leaves) {
                return Err(anyhow!("deleted node was not stored at a leaf index"));
            }

            if !deleted_indices.insert(idx) {
                return Err(anyhow!("duplicate node deletion in update batch"));
            }

            deleted_pairs.push((node, idx));
        }

        let removed = deleted_pairs.len();
        let new_leaves = old_leaves
            .checked_sub(removed)
            .ok_or_else(|| anyhow!("more deleted nodes than current leaves"))?;

        if new_leaves == 0 {
            self.node_to_tree_map.clear();
            self.tree_to_node_map.clear();
            self.degrees.clear();
            self.tree_data.timestamp.clear();
            self.tree_data.volume.clear();
            self.tree_data.size.clear();
            self.tree_data.f_delta.clear();
            self.tree_data.h_b.clear();
            self.tree_data.h_s.clear();
            touched.clear();
            return Ok(());
        }

        let new_internal = Self::internal_count_for_leaves(new_leaves);
        let new_total = new_internal + new_leaves;

        debug_assert!(new_internal <= old_internal);
        debug_assert!(new_total <= old_total);

        // Old layout:
        //   internal slots: [0, I0)
        //   leaf slots:     [I0, N0)
        //
        // New layout:
        //   internal slots: [0, I1)
        //   leaf slots:     [I1, N1)
        //
        // Because I1 <= I0 and N1 <= N0, final leaf slots split into:
        //   promoted slots: [I1, min(I0, N1))
        //     These were internal slots before deletion and must be populated
        //     from live old leaves at the tail.
        //   old leaf slots: [I0, N1)
        //     Live nodes here keep the same index; deleted nodes here are holes.
        //
        // The truncated suffix [N1, N0) contains exactly enough live leaves to
        // fill promoted slots plus holes in the kept old-leaf range.
        let promoted_start = new_internal;
        let promoted_end = old_internal.min(new_total);
        let kept_old_leaf_start = old_internal;
        let kept_old_leaf_end = new_total;

        let mut destinations = Vec::new();
        destinations.extend((promoted_start..promoted_end).map(TreeIndex));

        let kept_holes = deleted_pairs
            .iter()
            .map(|(_, idx)| *idx)
            .filter(|idx| idx.0 >= kept_old_leaf_start && idx.0 < kept_old_leaf_end)
            .collect::<Vec<_>>();
        destinations.extend(kept_holes);

        let source_start = old_internal.max(new_total);
        let sources = (source_start..old_total)
            .map(TreeIndex)
            .filter(|idx| !deleted_indices.contains(idx))
            .collect::<Vec<_>>();

        if sources.len() != destinations.len() {
            return Err(anyhow!(
                "deletion compaction invariant failed: {} sources for {} destinations",
                sources.len(),
                destinations.len()
            ));
        }

        for (node, idx) in &deleted_pairs {
            let removed_idx = self
                .node_to_tree_map
                .remove(node)
                .ok_or_else(|| anyhow!("deleted node disappeared during compaction"))?;
            debug_assert_eq!(removed_idx, *idx);
            self.tree_to_node_map.remove(idx);
            self.degrees.remove(node);
        }

        let leaf_size = NonZero::new(1).expect("1 is non-zero");
        let mut moves = Vec::with_capacity(sources.len());

        for (source, dest) in sources.iter().copied().zip(destinations.iter().copied()) {
            let node = self
                .tree_to_node_map
                .remove(&source)
                .ok_or_else(|| anyhow!("live tail source was missing from the tree map"))?;

            self.tree_data.volume[dest] = self.tree_data.volume[source];
            self.tree_data.size[dest] = leaf_size;

            self.tree_to_node_map.insert(dest, node);
            self.node_to_tree_map.insert(node, dest);
            moves.push((source, dest));
        }

        // Existing touched indices are old-layout leaf indices. For moved tail
        // leaves, probe the touched set directly: if the old source was pending,
        // translate it to the destination unless the promoted-range rebuild
        // already covers that destination.
        let mut moved_touched = Vec::new();
        for &(source, dest) in &moves {
            if touched.remove(&source) && (dest.0 < promoted_start || dest.0 >= promoted_end) {
                moved_touched.push(dest);
            }
        }

        // What remains in touched should be same-place live leaves. Drop deleted
        // holes and old tail indices that are about to be truncated, then add
        // the moved destinations that still need a batch-level recompute.
        touched.retain(|idx| {
            !deleted_indices.contains(idx) && idx.0 >= old_internal && idx.0 < new_total
        });
        touched.extend(moved_touched);

        // A destination in the kept old-leaf range is a sparse hole filled from
        // the tail, so its ancestors still need the batch-level recompute.
        for dest in destinations
            .iter()
            .copied()
            .filter(|dest| dest.0 >= old_internal)
        {
            touched.insert(dest);
        }

        self.tree_data.timestamp.truncate(new_total);
        self.tree_data.volume.truncate(new_total);
        self.tree_data.size.truncate(new_total);
        self.tree_data.f_delta.truncate(new_total);
        self.tree_data.h_b.truncate(new_total);
        self.tree_data.h_s.truncate(new_total);

        if promoted_start < promoted_end {
            self.rebuild_from_leaves(promoted_start, promoted_end);
        }

        // Truncating [N1, N0) can shorten the child list of the last remaining
        // internal node even when no surviving leaf value at that parent moved.
        // Touch the final leaf to recompute that boundary path.
        if new_leaves > 1 {
            let boundary = TreeIndex(new_total - 1);
            if boundary.0 < promoted_start || boundary.0 >= promoted_end {
                touched.insert(boundary);
            }
        }

        debug_assert_eq!(self.num_leaves(), new_leaves);
        debug_assert_eq!(self.num_internal_nodes(), new_internal);
        debug_assert!(self.tree_to_node_map.keys().all(|idx| idx.0 < new_total));
        debug_assert!(
            self.node_to_tree_map
                .values()
                .all(|idx| Self::is_leaf_index_for_leaves(*idx, new_leaves))
        );

        Ok(())
    }

    pub(crate) fn insert_fresh_nodes(
        &mut self,
        fresh: &[(V, Strict<T>)],
        touched: &mut FxHashSet<TreeIndex>,
    ) -> Result<()> {
        debug_assert!(ARITY > 1, "ARITY must be at least 2");

        let added = fresh.len();
        if added == 0 {
            return Ok(());
        }

        for (node, _) in fresh {
            if self.node_to_tree_map.contains_key(node) {
                return Err(anyhow!("fresh node already exists in the tree"));
            }
        }

        let old_leaves = self.num_leaves();
        let old_internal = self.num_internal_nodes();
        let old_total = old_internal + old_leaves;

        let new_leaves = old_leaves + added;
        let new_internal = Self::internal_count_for_leaves(new_leaves);
        let new_total = new_internal + new_leaves;

        let leaf_size = NonZero::new(1).expect("1 is non-zero");
        let volume_filler = Volume::from_scalar(T::ONE).unwrap();

        self.tree_data.timestamp.resize(new_total, 0);
        self.tree_data.volume.resize(new_total, volume_filler);
        self.tree_data.size.resize(new_total, leaf_size);
        self.tree_data.f_delta.resize(new_total, FDelta::zero());
        self.tree_data.h_b.resize(new_total, HB::zero());
        self.tree_data.h_s.resize(new_total, HS::zero());

        let old_leaf_start = Self::leaf_start_for_leaves(old_leaves);
        let old_leaf_end = old_leaf_start + old_leaves;

        if new_internal >= old_total {
            // Big height jump:
            // all old leaves move from [I_old, I_old + n_old) to
            // [I_new, I_new + n_old). The number of moved leaves can be large
            // only when the batch itself is large enough to force a new level.
            //
            // Since rebuild_from_leaves runs over the complete final leaf block,
            // every pending size/volume path is recomputed locally. Any touched
            // indices accumulated by deletion are old-layout indices, so discard
            // the whole pending set after the rebuild.
            let new_leaf_start = new_internal;

            self.tree_data
                .volume
                .copy_within(old_leaf_start..old_leaf_end, new_leaf_start);

            for i in 0..old_leaves {
                let old_idx = TreeIndex(old_leaf_start + i);
                let node = self.tree_to_node_map.remove(&old_idx).unwrap();
                let new_idx = TreeIndex(new_leaf_start + i);
                self.tree_to_node_map.insert(new_idx, node);
                self.node_to_tree_map.insert(node, new_idx);
            }

            let start_new = new_leaf_start + old_leaves;
            let end_new = start_new + added;
            for (i, (node, degree)) in fresh.iter().enumerate() {
                let idx = TreeIndex(start_new + i);
                self.tree_data.volume[idx] = Volume::new(*degree);
                self.tree_data.size[idx] = leaf_size;
                self.tree_data.timestamp[idx] = 0;
                self.tree_data.f_delta[idx] = FDelta::zero();
                self.tree_data.h_b[idx] = HB::zero();
                self.tree_data.h_s[idx] = HS::zero();
                self.tree_to_node_map.insert(idx, *node);
                self.node_to_tree_map.insert(*node, idx);
                self.degrees.push(*node, NodeDegree::new(*degree));
            }

            self.rebuild_from_leaves(new_leaf_start, end_new);
            touched.clear();
        } else {
            // Small height change:
            // exactly [I_old, I_new) old leaves become internal slots. Move
            // those O(I_new - I_old) leaves to the old suffix, then append the
            // fresh leaves after them.
            //
            // The local rebuild covers precisely the moved leaves plus the fresh
            // leaves: [old_total, end_new). If any previously touched leaf lived
            // in the promoted prefix [I_old, I_new), its old index is no longer a
            // leaf and its new path is already covered by this local rebuild, so
            // remove it from the pending touched set.
            let src_start = old_internal;
            let src_end = new_internal;
            let promoted = src_end - src_start;

            let dest_start = old_total;
            let dest_end = dest_start + promoted;

            self.tree_data
                .volume
                .copy_within(src_start..src_end, dest_start);

            for i in 0..promoted {
                let old_idx = TreeIndex(src_start + i);
                let node = self.tree_to_node_map.remove(&old_idx).unwrap();
                let new_idx = TreeIndex(dest_start + i);
                self.tree_to_node_map.insert(new_idx, node);
                self.node_to_tree_map.insert(node, new_idx);
            }

            let start_new = dest_end;
            let end_new = start_new + added;
            debug_assert_eq!(end_new, new_internal + new_leaves);

            for (i, (node, degree)) in fresh.iter().enumerate() {
                let idx = TreeIndex(start_new + i);
                self.tree_data.volume[idx] = Volume::new(*degree);
                self.tree_data.size[idx] = leaf_size;
                self.tree_data.timestamp[idx] = 0;
                self.tree_data.f_delta[idx] = FDelta::zero();
                self.tree_data.h_b[idx] = HB::zero();
                self.tree_data.h_s[idx] = HS::zero();
                self.tree_to_node_map.insert(idx, *node);
                self.node_to_tree_map.insert(*node, idx);
                self.degrees.push(*node, NodeDegree::new(*degree));
            }

            self.rebuild_from_leaves(old_total, end_new);
            touched.retain(|idx| idx.0 < src_start || idx.0 >= src_end);
        }

        debug_assert_eq!(self.num_leaves(), new_leaves);
        debug_assert_eq!(self.num_internal_nodes(), new_internal);
        Ok(())
    }

    pub(crate) fn update_modified_nodes(
        &mut self,
        modified: &[(V, Strict<T>)],
        touched: &mut FxHashSet<TreeIndex>,
    ) -> Result<()> {
        for (node, degree) in modified {
            let idx = *self
                .node_to_tree_map
                .get(node)
                .ok_or_else(|| anyhow!("modified node was missing from the tree"))?;
            debug_assert!(Self::is_leaf_index_for_leaves(idx, self.num_leaves()));

            self.tree_data.volume[idx] = Volume::new(*degree);
            self.degrees.push(*node, NodeDegree::new(*degree));
            touched.insert(idx);
        }

        Ok(())
    }

    pub fn apply_updates_from_set<F: Fn(&mut Self, TreeIndex)>(
        &mut self,
        update_set: &FxHashSet<TreeIndex>,
        f: F,
    ) {
        if update_set.is_empty() {
            return;
        }

        let mut current = FxHashSet::default();
        let mut bottom = FxHashSet::default();

        let total = self.num_total_nodes();
        let n = total as f64;
        let d = ARITY as f64;

        // For a full d-ary tree:
        // N = (d^(h+1) - 1)/(d-1) -> h = log_d((d-1)N + 1) - 1
        // We floor h here; for a complete tree this gives the deepest *full* level,
        // and the "bottom" level is either h or h+1, but the boundary
        // (first index of deepest level) is still:
        //   l_bottom_start = (d^h - 1)/(d-1)
        let h = (((d - 1.0) * n + 1.0).log(d)).floor() as u32 - 1;
        let l_bottom_start = (ARITY.pow(h) - 1) / (ARITY - 1);

        for &idx in update_set.iter() {
            if idx.0 >= l_bottom_start {
                bottom.insert(idx);
            } else {
                current.insert(idx);
            }
        }

        // process bottom set first, then merge with top set
        let bottom_parents: FxHashSet<TreeIndex> = bottom
            .into_iter()
            .filter_map(|child_idx| self.parent_index(child_idx))
            .collect();
        for p_idx in bottom_parents.iter() {
            f(self, *p_idx);
        }

        current.extend(bottom_parents.into_iter());

        // process until current is empty
        while !current.is_empty() {
            current = current
                .into_iter()
                .filter_map(|child_idx| self.parent_index(child_idx))
                .collect::<FxHashSet<_>>();
            for p_idx in current.iter() {
                f(self, *p_idx);
            }
        }
    }

    pub fn apply_updates_from_single<F: Fn(&mut Self, TreeIndex)>(
        &mut self,
        source: TreeIndex,
        f: F,
    ) {
        let mut maybe_parent = self.parent_index(source);
        while let Some(parent) = maybe_parent {
            f(self, parent);
            maybe_parent = self.parent_index(parent);
        }
    }

    #[inline(always)]
    pub fn one_step_recompute<X>(parent: TreeIndex, tree: &mut [X])
    where
        X: for<'a> std::iter::Sum<&'a X>,
    {
        let start = parent.0 * ARITY + 1;
        let end = (start + ARITY).min(tree.len());
        tree[parent] = tree[start..end].iter().sum();
    }

    #[inline(always)]
    pub fn one_step_recompute_size(parent: TreeIndex, tree: &mut [NonZero<usize>]) {
        let start = parent.0 * ARITY + 1;
        if start >= tree.len() {
            return;
        }
        let end = (start + ARITY).min(tree.len());
        tree[parent] = tree[start..end]
            .iter()
            .map(|&size| size.get())
            .sum::<usize>()
            .try_into()
            .expect("parent with at least one child must have non-zero size");
    }

    #[inline(always)]
    pub fn one_step_recompute_volume(parent: TreeIndex, tree: &mut [Volume<T>]) {
        let start = parent.0 * ARITY + 1;
        if start >= tree.len() {
            return;
        }
        let end = (start + ARITY).min(tree.len());
        let total: T = tree[start..end]
            .iter()
            .map(|volume| volume.into_scalar())
            .sum();
        tree[parent] =
            Volume::from_scalar(total).expect("sum of positive finite volumes must stay positive");
    }

    #[inline(always)]
    pub fn one_step_recompute_with_timestamp<X, F>(
        parent: TreeIndex,
        tree: &mut [X],
        timestamps: &[usize],
        cur_timestamp: usize,
        mut fallback: F,
    ) where
        X: Copy + std::iter::Sum<X>,
        F: FnMut(TreeIndex) -> X,
    {
        let start = parent.0 * ARITY + 1;
        if start >= tree.len() {
            return;
        }
        let end = (start + ARITY).min(tree.len());

        let total = (start..end)
            .map(|idx| {
                if timestamps[idx] == cur_timestamp {
                    tree[idx]
                } else {
                    fallback(TreeIndex(idx))
                }
            })
            .sum();

        tree[parent] = total;
    }
}

#[cfg(test)]
mod tests {
    use super::DynamicClustering;
    use crate::{
        DynamicClusteringAlg, GraphOracle,
        alg::TreeData,
        error::DynamicCoresetError,
        error::OracleError,
        types::{AlgType, NodeDegree, PartitionOutput, PartitionType, Strict, TreeIndex, Volume},
    };
    use priority_queue::PriorityQueue;
    use rustc_hash::{FxHashMap, FxHashSet};
    use std::{num::NonZeroUsize, sync::Arc};

    type TestClustering = DynamicClustering<2, usize, f64>;

    fn strict(value: f64) -> Strict<f64> {
        Strict::<f64>::new(value).unwrap()
    }

    fn volume(value: f64) -> Volume<f64> {
        Volume::from_scalar(value).unwrap()
    }

    fn test_clustering() -> TestClustering {
        let cluster_alg: AlgType<f64> = Arc::new(|_, _| (Vec::new(), 0));

        DynamicClustering {
            node_to_tree_map: FxHashMap::default(),
            tree_to_node_map: FxHashMap::default(),
            degrees: PriorityQueue::new(),
            tree_data: TreeData {
                timestamp: vec![],
                volume: vec![],
                size: vec![],
                f_delta: vec![],
                h_b: vec![],
                h_s: vec![],
            },
            sigma: strict(1.0),
            timestamp: 0,
            coreset_size: 1,
            sampling_seeds: 1,
            num_clusters: 1,
            cluster_alg,
            prop_name: String::from("w"),
        }
    }

    fn apply_size_volume_updates(clustering: &mut TestClustering, touched: &FxHashSet<TreeIndex>) {
        clustering.apply_updates_from_set(touched, |other, idx| {
            TestClustering::one_step_recompute_size(idx, &mut other.tree_data.size);
            TestClustering::one_step_recompute_volume(idx, &mut other.tree_data.volume);
        });
    }

    struct EmptyOracle;

    impl EmptyOracle {
        fn new() -> Self {
            Self
        }

        fn empty_rows<'a>(
            &mut self,
            nodes: &'a [usize],
        ) -> Result<Vec<&'a [(usize, Strict<f64>)]>, OracleError<String>> {
            static EMPTY_NEIGHBOURS: &[(usize, Strict<f64>)] = &[];

            Ok(vec![EMPTY_NEIGHBOURS; nodes.len()])
        }
    }

    impl GraphOracle<usize, f64, String> for EmptyOracle {
        fn graph_neighbourhoods<'a>(
            &'a mut self,
            nodes: &'a [usize],
        ) -> Result<Vec<&'a [(usize, Strict<f64>)]>, OracleError<String>> {
            self.empty_rows(nodes)
        }

        fn coreset_neighbourhoods<'a>(
            &'a mut self,
            nodes: &'a [usize],
        ) -> Result<Vec<&'a [(usize, Strict<f64>)]>, OracleError<String>> {
            self.empty_rows(nodes)
        }
    }

    fn assert_tree_consistent(clustering: &TestClustering) {
        let leaves = clustering.num_leaves();
        let total = TestClustering::total_count_for_leaves(leaves);
        let leaf_range = TestClustering::leaf_range_for_leaves(leaves);

        assert_eq!(clustering.tree_data.timestamp.len(), total);
        assert_eq!(clustering.tree_data.volume.len(), total);
        assert_eq!(clustering.tree_data.size.len(), total);
        assert_eq!(clustering.tree_data.f_delta.len(), total);
        assert_eq!(clustering.tree_data.h_b.len(), total);
        assert_eq!(clustering.tree_data.h_s.len(), total);

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
                clustering.tree_data.volume[idx].into_scalar()
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
            .map(|idx| clustering.tree_data.volume[idx].into_scalar())
            .sum::<f64>();
        assert_eq!(clustering.tree_data.size[TreeIndex(0)].get(), leaves);
        assert_eq!(
            clustering.tree_data.volume[TreeIndex(0)].into_scalar(),
            expected_volume
        );
    }

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
        assert_eq!(clustering.tree_data.volume[TreeIndex(0)], volume(3.0));
        assert_eq!(
            clustering.tree_data.size[TreeIndex(0)],
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
        assert_eq!(clustering.tree_data.volume[TreeIndex(0)], volume(6.0));
        assert_eq!(
            clustering.tree_data.size[TreeIndex(0)],
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

        assert_eq!(clustering.tree_data.volume.len(), 3);
        assert_eq!(clustering.node_to_tree_map.get(&2), Some(&TreeIndex(1)));
        assert_eq!(clustering.node_to_tree_map.get(&1), Some(&TreeIndex(2)));
        assert!(!clustering.node_to_tree_map.contains_key(&3));

        assert_eq!(touched, FxHashSet::from_iter([TreeIndex(2)]));
        assert_eq!(clustering.tree_data.volume[TreeIndex(0)], volume(3.0));
        assert_eq!(
            clustering.tree_data.size[TreeIndex(0)],
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

        assert_eq!(clustering.tree_data.volume.len(), 3);
        assert_eq!(clustering.node_to_tree_map.get(&2), Some(&TreeIndex(1)));
        assert_eq!(clustering.node_to_tree_map.get(&3), Some(&TreeIndex(2)));
        assert!(!clustering.node_to_tree_map.contains_key(&1));

        assert_eq!(touched, FxHashSet::from_iter([TreeIndex(2)]));
        assert_eq!(clustering.tree_data.volume[TreeIndex(0)], volume(5.0));
        assert_eq!(
            clustering.tree_data.size[TreeIndex(0)],
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
        assert_eq!(clustering.tree_data.volume.len(), 1);
        assert_eq!(clustering.node_to_tree_map.get(&5), Some(&TreeIndex(0)));
        assert_eq!(clustering.tree_to_node_map.get(&TreeIndex(0)), Some(&5));
        assert_eq!(clustering.tree_data.volume[TreeIndex(0)], volume(5.0));
        assert_eq!(
            clustering.tree_data.size[TreeIndex(0)],
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
        assert!(clustering.tree_data.volume.is_empty());
        assert!(clustering.tree_data.size.is_empty());
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

        assert_eq!(clustering.tree_data.size[TreeIndex(0)].get(), 5);
        assert_eq!(
            clustering.tree_data.volume[TreeIndex(0)].into_scalar(),
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
        assert!(clustering.tree_data.volume.is_empty());
    }

    #[test]
    fn query_empty_tree_returns_no_data_error() {
        let mut clustering = test_clustering();
        let mut oracle = EmptyOracle::new();

        let err = <TestClustering as DynamicClusteringAlg<usize, f64>>::query::<_, String>(
            &mut clustering,
            PartitionType::All,
            &mut oracle,
        )
        .unwrap_err();

        assert!(matches!(
            err.downcast_ref::<DynamicCoresetError>(),
            Some(DynamicCoresetError::NoData)
        ));
    }

    #[test]
    fn query_runs_after_mixed_node_updates() {
        let mut clustering = test_clustering();
        clustering.coreset_size = 3;
        clustering.sampling_seeds = 2;
        clustering.cluster_alg = Arc::new(|graph, _| {
            let n = graph.symbolic().nrows();
            (vec![0; n], 1)
        });
        let mut oracle = EmptyOracle::new();

        <TestClustering as DynamicClusteringAlg<usize, f64>>::apply_node_ops(
            &mut clustering,
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

        <TestClustering as DynamicClusteringAlg<usize, f64>>::apply_node_ops(
            &mut clustering,
            &[(2, None), (3, Some(strict(30.0))), (7, Some(strict(7.0)))],
        )
        .unwrap();

        let output = <TestClustering as DynamicClusteringAlg<usize, f64>>::query::<_, String>(
            &mut clustering,
            PartitionType::All,
            &mut oracle,
        )
        .unwrap();

        match output {
            PartitionOutput::All(nodes, labels, num_clusters) => {
                assert_eq!(num_clusters, 1);
                assert_eq!(nodes.len(), clustering.num_leaves());
                assert_eq!(labels.len(), nodes.len());
                assert!(labels.iter().all(|label| *label == 0));
                assert!(!nodes.contains(&2));
                assert!(nodes.contains(&7));
            }
            PartitionOutput::Subset(_, _) => panic!("expected all-node partition output"),
        }

        assert_tree_consistent(&clustering);
    }
}
