//! In-process graph storage and borrowed query oracle handles.

mod graph;
mod oracle;
pub mod workloads;

pub use graph::{InMemoryGraphError, InMemoryUndirectedGraph, NodeOpsBuffer};
pub use oracle::InMemoryOracle;

#[cfg(test)]
mod tests;
