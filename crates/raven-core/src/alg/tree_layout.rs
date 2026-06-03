use rustc_hash::FxHashSet;

use crate::types::TreeIndex;

/// Shared zero-sized helper struct for tree layout calculations.
pub(crate) struct TreeLayout<const ARITY: usize>;

impl<const ARITY: usize> TreeLayout<ARITY> {
    #[inline(always)]
    pub fn parent_index(child: TreeIndex) -> Option<TreeIndex> {
        if child.0 == 0 {
            None
        } else {
            Some((child - TreeIndex(1)) / ARITY)
        }
    }

    #[inline(always)]
    pub fn child_index(parent: TreeIndex, child: usize) -> TreeIndex {
        TreeIndex(parent.0 * ARITY + 1 + child)
    }

    pub fn apply_updates_from_set(
        total: usize,
        update_set: &FxHashSet<TreeIndex>,
        mut update: impl FnMut(TreeIndex),
    ) {
        if update_set.is_empty() {
            return;
        }

        let mut current = FxHashSet::default();
        let mut bottom = FxHashSet::default();
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

        for &idx in update_set {
            if idx.0 >= l_bottom_start {
                bottom.insert(idx);
            } else {
                current.insert(idx);
            }
        }

        // process bottom set first, then merge with top set
        let bottom_parents: FxHashSet<TreeIndex> =
            bottom.into_iter().filter_map(Self::parent_index).collect();
        for &parent in &bottom_parents {
            update(parent);
        }

        current.extend(bottom_parents);

        // process until current is empty
        while !current.is_empty() {
            current = current.into_iter().filter_map(Self::parent_index).collect();
            for &parent in &current {
                update(parent);
            }
        }
    }

    pub fn apply_updates_from_single(source: TreeIndex, mut update: impl FnMut(TreeIndex)) {
        let mut parent = Self::parent_index(source);
        while let Some(idx) = parent {
            update(idx);
            parent = Self::parent_index(idx);
        }
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

        tree[parent] = (start..end)
            .map(|idx| {
                if timestamps[idx] == cur_timestamp {
                    tree[idx]
                } else {
                    fallback(TreeIndex(idx))
                }
            })
            .sum();
    }
}
