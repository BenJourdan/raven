use std::num::NonZeroUsize;

use rustc_hash::FxHashMap;

use crate::types::{FDelta, NonStrict, NonStrictCarrierOps, TreeIndex, Volume, HB, HS};

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

/// Workspace holding borrowed references for a single trial of coreset construction.
pub struct TrialWorkspace<'a, const ARITY: usize, V, T> {
    pub timestamp: usize,
    pub persistent: &'a Persistent<T>,
    pub query_time: &'a mut QueryTime<T>,
    pub tree_to_node_map: &'a FxHashMap<TreeIndex, V>,
    pub node_to_tree_map: &'a FxHashMap<V, TreeIndex>,
}
