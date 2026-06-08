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

/// Borrowed neighbourhood rows stored as one flat data slice plus row offsets.
///
/// Row `i` occupies `data[offsets[i]..offsets[i + 1]]`. A well-formed value has
/// `offsets.len() == number_of_rows + 1`, starts at offset `0`, ends at
/// `data.len()`, and has monotonically increasing offsets.
#[derive(Debug, Clone, Copy)]
pub struct Neighbourhoods<'a, V, T> {
    data: &'a WeightedNodes<V, T>,
    offsets: &'a [usize],
}

impl<'a, V, T> Neighbourhoods<'a, V, T> {
    pub fn new(data: &'a WeightedNodes<V, T>, offsets: &'a [usize]) -> Self {
        debug_assert!(Self::well_formed(data, offsets));
        Self { data, offsets }
    }

    pub fn data(&self) -> &'a WeightedNodes<V, T> {
        self.data
    }

    pub fn offsets(&self) -> &'a [usize] {
        self.offsets
    }

    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_well_formed(&self) -> bool {
        Self::well_formed(self.data, self.offsets)
    }

    pub fn row(&self, index: usize) -> Option<&'a WeightedNodes<V, T>> {
        let start = *self.offsets.get(index)?;
        let end = *self.offsets.get(index + 1)?;
        if start > end {
            return None;
        }
        self.data.get(start..end)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &'a WeightedNodes<V, T>> + '_ {
        (0..self.len()).map(|index| {
            self.row(index)
                .expect("neighbourhood offsets should be well-formed")
        })
    }

    fn well_formed(data: &WeightedNodes<V, T>, offsets: &[usize]) -> bool {
        offsets.first() == Some(&0)
            && offsets.last() == Some(&data.len())
            && offsets.windows(2).all(|window| window[0] <= window[1])
    }
}

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
