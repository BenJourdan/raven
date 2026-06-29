use num_traits::Float as _;
use rustc_hash::FxHashSet;

use crate::types::{
    FDelta, FloatScalar, HB, HS, NonStrict, NonStrictCarrierOps, Strict, StrictCarrierOps,
    TreeIndex, Volume,
};

/// Query-time tree values whose invariants are preserved by summing children.
///
/// This is deliberately narrower than `Sum`: these scratch quantities are
/// non-negative finite by construction, so hot tree recomputation can validate
/// the invariant in debug builds and avoid repeated checked wrapper creation in
/// release builds.
pub(crate) trait NonStrictTreeValue: Copy {
    type Scalar: FloatScalar;

    fn into_scalar(self) -> Self::Scalar;

    /// # Safety
    /// `x` must be non-negative and finite.
    unsafe fn from_scalar_unchecked(x: Self::Scalar) -> Self;
}

macro_rules! impl_non_strict_tree_value {
    ($($name:ident),* $(,)?) => {
        $(
            impl<T> NonStrictTreeValue for $name<T>
            where
                T: FloatScalar,
                NonStrict<T>: NonStrictCarrierOps<Scalar = T>,
            {
                type Scalar = T;

                #[inline(always)]
                fn into_scalar(self) -> Self::Scalar {
                    $name::<T>::into_scalar(self)
                }

                #[inline(always)]
                unsafe fn from_scalar_unchecked(x: Self::Scalar) -> Self {
                    unsafe { $name::<T>::from_scalar_unchecked(x) }
                }
            }
        )*
    };
}

impl_non_strict_tree_value!(FDelta, HB, HS);

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

        let mut current = Vec::with_capacity(update_set.len());
        let mut next = Vec::with_capacity(update_set.len());
        let mut seen = vec![0; total];
        let mut seen_epoch = 0;
        Self::apply_updates_from_iter_with_marker(
            total,
            update_set.iter().copied(),
            &mut current,
            &mut next,
            &mut seen,
            &mut seen_epoch,
            &mut update,
        );
    }

    pub fn apply_updates_from_slice_with_marker(
        total: usize,
        update_indices: &[TreeIndex],
        current: &mut Vec<TreeIndex>,
        next: &mut Vec<TreeIndex>,
        seen: &mut Vec<usize>,
        seen_epoch: &mut usize,
        update: impl FnMut(TreeIndex),
    ) {
        if update_indices.is_empty() {
            current.clear();
            next.clear();
            return;
        }

        Self::apply_updates_from_iter_with_marker(
            total,
            update_indices.iter().copied(),
            current,
            next,
            seen,
            seen_epoch,
            update,
        );
    }

    fn apply_updates_from_iter_with_marker(
        total: usize,
        update_indices: impl IntoIterator<Item = TreeIndex>,
        current: &mut Vec<TreeIndex>,
        next: &mut Vec<TreeIndex>,
        seen: &mut Vec<usize>,
        seen_epoch: &mut usize,
        mut update: impl FnMut(TreeIndex),
    ) {
        current.clear();
        next.clear();
        if seen.len() < total {
            seen.resize(total, 0);
        }

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

        let current_epoch = Self::next_marker_epoch(seen, seen_epoch);
        let bottom_epoch = Self::next_marker_epoch(seen, seen_epoch);
        for idx in update_indices {
            if idx.0 >= l_bottom_start {
                if let Some(parent) = Self::parent_index(idx) {
                    Self::push_once(parent, next, seen, bottom_epoch);
                }
            } else {
                Self::push_once(idx, current, seen, current_epoch);
            }
        }

        // process bottom set first, then merge with top set
        for &parent in next.iter() {
            update(parent);
        }

        current.extend_from_slice(next);

        // process until current is empty
        while !current.is_empty() {
            next.clear();
            let mut parent_epoch = None;
            for parent in current.iter().filter_map(|&idx| Self::parent_index(idx)) {
                let epoch =
                    *parent_epoch.get_or_insert_with(|| Self::next_marker_epoch(seen, seen_epoch));
                Self::push_once(parent, next, seen, epoch);
            }

            if next.is_empty() {
                break;
            }

            for &parent in next.iter() {
                update(parent);
            }

            std::mem::swap(current, next);
        }
    }

    #[inline(always)]
    fn next_marker_epoch(seen: &mut [usize], seen_epoch: &mut usize) -> usize {
        *seen_epoch = seen_epoch.checked_add(1).unwrap_or_else(|| {
            seen.fill(0);
            1
        });
        *seen_epoch
    }

    #[inline(always)]
    fn push_once(idx: TreeIndex, indices: &mut Vec<TreeIndex>, seen: &mut [usize], epoch: usize) {
        if seen[idx.0] == epoch {
            return;
        }

        seen[idx.0] = epoch;
        indices.push(idx);
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

    #[inline(always)]
    pub fn one_step_recompute_non_strict_with_timestamp<X, F>(
        parent: TreeIndex,
        tree: &mut [X],
        timestamps: &[usize],
        cur_timestamp: usize,
        mut fallback: F,
    ) where
        X: NonStrictTreeValue,
        F: FnMut(TreeIndex) -> X,
    {
        let start = parent.0 * ARITY + 1;
        if start >= tree.len() {
            return;
        }
        let end = (start + ARITY).min(tree.len());

        let mut total = X::Scalar::ZERO;
        for idx in start..end {
            let value = if timestamps[idx] == cur_timestamp {
                tree[idx]
            } else {
                fallback(TreeIndex(idx))
            };
            total = total + value.into_scalar();
        }

        debug_assert!(
            total.is_finite() && total >= X::Scalar::ZERO,
            "sum of non-strict tree values must stay non-negative and finite"
        );

        // SAFETY: all inputs are non-negative finite query-time quantities.
        // Addition preserves non-negativity; debug builds check finiteness.
        tree[parent] = unsafe { X::from_scalar_unchecked(total) };
    }

    #[inline(always)]
    pub fn one_step_recompute_h_pair_with_timestamp<T>(
        parent: TreeIndex,
        h_b: &mut [HB<T>],
        h_s: &mut [HS<T>],
        volumes: &[Volume<T>],
        timestamps: &[usize],
        cur_timestamp: usize,
    ) where
        T: FloatScalar,
        Strict<T>: StrictCarrierOps<Scalar = T>,
        NonStrict<T>: NonStrictCarrierOps<Scalar = T>,
    {
        let start = parent.0 * ARITY + 1;
        if start >= h_b.len() {
            return;
        }
        debug_assert_eq!(h_b.len(), h_s.len());
        debug_assert_eq!(h_b.len(), volumes.len());
        debug_assert_eq!(h_b.len(), timestamps.len());

        let end = (start + ARITY).min(h_b.len());
        let mut h_b_total = T::ZERO;
        let mut h_s_total = T::ZERO;

        for idx in start..end {
            if timestamps[idx] == cur_timestamp {
                h_b_total = h_b_total + h_b[idx].into_scalar();
                h_s_total = h_s_total + h_s[idx].into_scalar();
            } else {
                h_b_total = h_b_total + volumes[idx].into_scalar();
            }
        }

        debug_assert!(
            h_b_total.is_finite() && h_b_total >= T::ZERO,
            "sum of h_b values must stay non-negative and finite"
        );
        debug_assert!(
            h_s_total.is_finite() && h_s_total >= T::ZERO,
            "sum of h_s values must stay non-negative and finite"
        );

        // SAFETY: all inputs are non-negative finite query-time quantities.
        // Addition preserves non-negativity; debug builds check finiteness.
        h_b[parent] = unsafe { HB::from_scalar_unchecked(h_b_total) };
        h_s[parent] = unsafe { HS::from_scalar_unchecked(h_s_total) };
    }
}
