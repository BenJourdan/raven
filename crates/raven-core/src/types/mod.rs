pub mod alg;
pub mod float;
pub mod int;
pub mod pow2;

pub use alg::*;
pub use float::*;
pub use int::*;
pub use pow2::*;

/// Specify whether the query is for all
/// nodes, or just a subset of nodes.
pub enum PartitionType<'a, V> {
    All,
    Subset(&'a [V]),
}

/// Specify the output format for a partition query.
/// All: return a vector of all nodes and their indices + num clusters.
/// Subset: return a vector of the indices of the subset nodes + num clusters.
#[derive(Debug)]
pub enum PartitionOutput<V> {
    All(Vec<V>, Vec<usize>, usize),
    Subset(Vec<usize>, usize),
}

#[derive(Debug, Clone)]
pub enum EdgeDeletionResult {
    BothNodesStillConnected,
    OneNodeDisconnected(String),
    BothNodesDisconnected(String, String),
}
