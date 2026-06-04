pub mod alg;
pub mod error;
pub mod types;

use types::{PartitionOutput, PartitionType, WeightedNodes};

use crate::types::{Strict, TrialOutputMode};

/// Batch neighbourhood oracle used by the dynamic clustering algorithm.
///
/// `graph_neighbourhoods` returns complete adjacency rows for graph-wide
/// lookups. `coreset_neighbourhoods` treats its input batch as the complete
/// coreset node set for the current query; each returned row must contain only
/// neighbours that also appear in that same input batch.
/// Each Query trial gets its own oracle instance.
/// All oracles passed to the same query should observe the same
/// graph snapshot state.
pub trait GraphOracle<V, T, E> {
    /// Query the neighbourhoods of a batch of nodes.
    /// Returned outer vector length must match the input batch size
    /// and the order must match nodes.
    /// returned row slices may borrow from self owned data.
    fn graph_neighbourhoods<'a>(
        &'a mut self,
        nodes: &'a [V],
    ) -> Result<Vec<&'a WeightedNodes<V, T>>, error::OracleError<E>>;

    fn coreset_neighbourhoods<'a>(
        &'a mut self,
        nodes: &'a [V],
    ) -> Result<Vec<&'a WeightedNodes<V, T>>, error::OracleError<E>>;
}

/// A trait for dynamic clustering algorithms.
pub trait DynamicClusteringAlg<V, T> {
    /// Apply a batch of node updates to the data structure.
    fn apply_node_ops(&mut self, diffs: &[(V, Option<Strict<T>>)]) -> anyhow::Result<()>;

    /// Query the current clustering with a partition type.
    /// oracles length must match the number of trials.
    /// Trials may run in parallel.
    fn query<O, E>(
        &mut self,
        partition: PartitionType<V>,
        trial_output_mode: TrialOutputMode,
        oracles: &mut [&mut O],
    ) -> anyhow::Result<PartitionOutput<V, T>>
    where
        O: GraphOracle<V, T, E> + ?Sized + Send,
        E: std::fmt::Display;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imports_work() {
        alg::TreeData::<4, types::Strict<f64>> {
            persistent: alg::Persistent {
                size: vec![],
                volume: vec![],
            },
            query_time: vec![],
        };

        let x = types::Strict::<f64>::new(1.0).unwrap();
        let _y: Option<types::Strict<f64>> = Some(x);
    }
}
