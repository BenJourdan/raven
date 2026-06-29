use std::num::NonZeroUsize;

use rustc_hash::FxHashMap;

use crate::types::{
    FDelta, FloatScalar, HB, HS, NonStrict, NonStrictCarrierOps, TreeIndex, Volume,
};

/// The tree data backing the dynamic clustering algorithm.
pub struct TreeData<const ARITY: usize, T> {
    pub persistent: Persistent<T>,
    pub query_time: Vec<QueryTime<T>>,
}

/// Tree data shared across all query trials.
pub struct Persistent<T> {
    pub volume: Vec<Volume<T>>,
    pub size: Vec<NonZeroUsize>,
}

/// Scratch tree data local to a single query trial.
pub struct QueryTime<T> {
    pub timestamp: Vec<usize>,
    pub f_delta: Vec<FDelta<T>>,
    pub h_b: Vec<HB<T>>,
    pub h_s: Vec<HS<T>>,
    pub seed_owner: Vec<TreeIndex>,
    pub seed_owner_epoch: Vec<usize>,
    pub seed_weight: Vec<T>,
    pub seed_weight_epoch: Vec<usize>,
    pub old_seed_seen: Vec<usize>,
    pub old_seed_seen_epoch: usize,
    pub tree_update_current: Vec<TreeIndex>,
    pub tree_update_next: Vec<TreeIndex>,
    pub tree_update_seen: Vec<usize>,
    pub tree_update_seen_epoch: usize,
}

impl<T> Default for QueryTime<T> {
    fn default() -> Self {
        Self {
            timestamp: Vec::new(),
            f_delta: Vec::new(),
            h_b: Vec::new(),
            h_s: Vec::new(),
            seed_owner: Vec::new(),
            seed_owner_epoch: Vec::new(),
            seed_weight: Vec::new(),
            seed_weight_epoch: Vec::new(),
            old_seed_seen: Vec::new(),
            old_seed_seen_epoch: 0,
            tree_update_current: Vec::new(),
            tree_update_next: Vec::new(),
            tree_update_seen: Vec::new(),
            tree_update_seen_epoch: 0,
        }
    }
}

impl<T> QueryTime<T>
where
    T: FloatScalar,
    NonStrict<T>: NonStrictCarrierOps<Scalar = T>,
{
    pub fn clear(&mut self) {
        self.timestamp.clear();
        self.f_delta.clear();
        self.h_b.clear();
        self.h_s.clear();
        self.seed_owner.clear();
        self.seed_owner_epoch.clear();
        self.seed_weight.clear();
        self.seed_weight_epoch.clear();
        self.old_seed_seen.clear();
        self.old_seed_seen_epoch = 0;
        self.tree_update_current.clear();
        self.tree_update_next.clear();
        self.tree_update_seen.clear();
        self.tree_update_seen_epoch = 0;
    }

    pub fn truncate(&mut self, new_len: usize) {
        self.timestamp.truncate(new_len);
        self.f_delta.truncate(new_len);
        self.h_b.truncate(new_len);
        self.h_s.truncate(new_len);
        self.seed_owner.truncate(new_len);
        self.seed_owner_epoch.truncate(new_len);
        self.seed_weight.truncate(new_len);
        self.seed_weight_epoch.truncate(new_len);
        self.old_seed_seen.truncate(new_len);
        self.tree_update_seen.truncate(new_len);
    }

    pub fn resize(&mut self, new_len: usize) {
        self.timestamp.resize(new_len, 0);
        self.f_delta.resize_with(new_len, FDelta::zero);
        self.h_b.resize_with(new_len, HB::zero);
        self.h_s.resize_with(new_len, HS::zero);
        self.seed_owner.resize(new_len, TreeIndex(0));
        self.seed_owner_epoch.resize(new_len, 0);
        self.seed_weight.resize(new_len, T::ZERO);
        self.seed_weight_epoch.resize(new_len, 0);
        self.old_seed_seen.resize(new_len, 0);
        self.tree_update_seen.resize(new_len, 0);
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

/// Workspace holding borrowed references for a single trial of coreset construction.
pub struct TrialWorkspace<'a, const ARITY: usize, V, T> {
    pub timestamp: usize,
    pub persistent: &'a Persistent<T>,
    pub query_time: &'a mut QueryTime<T>,
    pub tree_to_node_map: &'a FxHashMap<TreeIndex, V>,
    pub node_to_tree_map: &'a FxHashMap<V, TreeIndex>,
}
