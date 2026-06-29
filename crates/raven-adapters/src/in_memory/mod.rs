//! In-process graph storage and borrowed query oracle handles.

mod graph;
mod index;
mod oracle;
pub mod workloads;

pub use graph::{InMemoryGraphError, InMemoryUndirectedGraph, NodeOpsBuffer};
pub use index::{InMemoryIndex, InMemoryIndexError, InMemoryOracleTiming};
pub use oracle::{DenseInMemoryOracle, InMemoryOracle};

#[cfg(test)]
mod tests;
