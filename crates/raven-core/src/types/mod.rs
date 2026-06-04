pub mod alg;
pub mod float;
pub mod int;
pub mod pow2;

pub use alg::*;
pub use float::*;
pub use int::*;
pub use pow2::*;

/// Type alias for a slice of positively weighted nodes.
pub type WeightedNodes<V, T> = [(V, Strict<T>)];

/// Specify whether the query is for all
/// nodes, or just a subset of nodes.
pub enum PartitionType<'a, V> {
    All,
    Subset(&'a [V]),
}

/// The output of a query trial.
/// Includes the trial index, the labels,
/// scores (if used), and the number of clusters.
/// The ordering of labels and scores respects the input node order.
#[derive(Debug)]
pub struct TrialPartition<T> {
    pub trial_index: usize,
    pub labels: Vec<usize>,
    pub scores: Option<Vec<T>>,
    pub num_clusters: usize,
}
/// Specify the output format for a partition query.
/// All: return labels and scores for all nodes, along with the trial partitions.
/// Subset: return labels and scores only for the specified subset of nodes,
/// along with the trial partitions.
#[derive(Debug)]
pub enum PartitionOutput<V, T> {
    All(Vec<V>, Vec<TrialPartition<T>>),
    Subset(Vec<TrialPartition<T>>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrialOutputMode {
    AllTrials,
    Winner(TrialObjective),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrialObjective {
    KernelDistance,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EdgeDeletionResult {
    BothNodesStillConnected,
    OneNodeDisconnected(String),
    BothNodesDisconnected(String, String),
}
