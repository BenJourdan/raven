pub mod alg;
#[cfg(feature = "clustering")]
pub mod clustering;
pub mod error;
pub mod metrics;
pub mod types;

use types::{Neighbourhoods, PartitionOutput, PartitionType};

use crate::types::{Strict, TrialOutputMode};

/// Batch neighbourhood oracle used by the dynamic clustering algorithm.
///
/// `graph_neighbourhoods` returns complete adjacency rows for graph-wide
/// lookups. `graph_neighbourhoods_intersecting` returns graph adjacency rows
/// filtered to a target node set. `coreset_neighbourhoods` treats its input
/// batch as the complete coreset node set for the current query; each returned
/// row must contain only neighbours that also appear in that same input batch.
/// Each query trial gets its own oracle instance.
/// All oracles passed to the same query should observe the same
/// graph snapshot state.
pub trait GraphOracle<V, T, E> {
    /// Query the neighbourhoods of a batch of nodes.
    ///
    /// Returned rows must match the input batch length and order. Rows are
    /// represented as one flat data slice plus offsets, and may borrow from
    /// oracle-owned scratch storage. Those borrowed rows only need to remain
    /// valid until the next method call on the same oracle handle.
    fn graph_neighbourhoods<'a>(
        &'a mut self,
        nodes: &[V],
    ) -> Result<Neighbourhoods<'a, V, T>, error::OracleError<E>>;

    /// Query source neighbourhood rows filtered to a target node set.
    ///
    /// Returned rows must match the source batch length and order. Targets are
    /// only used as a filter: missing source nodes should be reported as graph
    /// errors, but missing target nodes do not need to be validated.
    fn graph_neighbourhoods_intersecting<'a>(
        &'a mut self,
        sources: &[V],
        targets: &[V],
    ) -> Result<Neighbourhoods<'a, V, T>, error::OracleError<E>>;

    fn coreset_neighbourhoods<'a>(
        &'a mut self,
        nodes: &[V],
    ) -> Result<Neighbourhoods<'a, V, T>, error::OracleError<E>>;
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
