//! Building blocks for future IPC and network-backed graph adapters.
//!
//! Application code is expected to own graph-update ingestion. For each
//! queryable remote graph snapshot it can pass Raven a [`RemoteSnapshotNodeOps`]
//! value, apply those node operations to the core data structure, and then use
//! [`RemoteGraphClient`] to create oracle handles that read from the same
//! [`SnapshotId`].
//!
//! Transport-specific code only needs to implement [`RemoteGraphBackend`]:
//! given a snapshot and node ids, fetch the requested neighbourhood rows.
//! [`RemoteGraphClient`] and [`RemoteOracle`] handle the synchronous oracle
//! plumbing required by `raven-core`.
//!
//! ```ignore
//! // Application/event-stream code owns update ingestion and degree diffs.
//! let snapshot_node_ops = RemoteSnapshotNodeOps::new(snapshot, node_ops);
//! clustering.apply_node_ops(snapshot_node_ops.node_ops())?;
//!
//! // The remote backend owns transport-specific row fetching.
//! let remote_snapshot = client.snapshot(snapshot_node_ops.snapshot);
//! let output = remote_snapshot.with_oracles::<NodeId, f64, _>(num_trials, |oracles| {
//!     clustering.query(partition, trial_output_mode, oracles)
//! })?;
//! # Ok::<_, anyhow::Error>(())
//! ```

pub mod backend;
pub mod cache;
pub mod client;
#[cfg(feature = "memgraph")]
pub mod memgraph;
pub mod oracle;
pub mod snapshot;

pub use backend::{OwnedNeighbourhoods, RemoteGraphBackend, RemoteGraphError};
pub use client::{RemoteGraphClient, RemoteGraphSnapshot};
#[cfg(feature = "memgraph")]
pub use memgraph::{
    MemgraphBackend, MemgraphBackendError, MemgraphCacheConfig, MemgraphDecodeError,
    MemgraphFilterStrategy, MemgraphNeighbourhoodRow, MemgraphNodeId, MemgraphQueries,
    MemgraphRowError,
};
pub use oracle::RemoteOracle;
pub use snapshot::{RemoteSnapshotNodeOps, SnapshotId};

#[cfg(test)]
mod tests;
