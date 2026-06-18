use std::{fmt, future::Future};

use raven_core::types::{Neighbourhoods, Strict};

use super::SnapshotId;

#[derive(Debug)]
pub struct OwnedNeighbourhoods<V, T> {
    pub data: Vec<(V, Strict<T>)>,
    pub offsets: Vec<usize>,
}

impl<V, T> OwnedNeighbourhoods<V, T> {
    pub fn as_borrowed(&self) -> Neighbourhoods<'_, V, T> {
        Neighbourhoods::new(&self.data, &self.offsets)
    }

    pub fn empty() -> Self {
        Self {
            data: Vec::new(),
            offsets: vec![0],
        }
    }
}

/// Async row-fetching interface implemented by transport-specific backends.
///
/// All methods must observe the requested [`SnapshotId`]. Returned rows use the
/// same flat `data + offsets` representation as
/// [`raven_core::types::Neighbourhoods`], with one row per requested source node
/// in the same order.
///
/// `Error` should be a backend-specific error type. Concrete adapters can use
/// it to combine transport failures, remote service errors, and graph contract
/// errors. [`RemoteGraphError`] is only a small convenience enum for graph-level
/// failures that many backends may want to wrap.
pub trait RemoteGraphBackend<V, T>: Send + Sync {
    type Error;

    /// Fetch complete graph adjacency rows for `nodes`.
    ///
    /// Returned rows must match `nodes.len()` and preserve input order. Missing
    /// source nodes should be reported as graph/backend errors. Returned
    /// neighbours should represent the remote graph snapshot visible at
    /// `snapshot`.
    fn graph_neighbourhoods(
        &self,
        snapshot: SnapshotId,
        nodes: Vec<V>,
    ) -> impl Future<Output = Result<OwnedNeighbourhoods<V, T>, Self::Error>> + Send;

    /// Fetch graph adjacency rows for `sources`, filtered to `targets`.
    ///
    /// Returned rows must match `sources.len()` and preserve source order. Every
    /// returned neighbour must be a member of `targets`. Missing sources should
    /// be reported as graph/backend errors, but `targets` are only a filter:
    /// missing target nodes do not need to be validated.
    fn graph_neighbourhoods_intersecting(
        &self,
        snapshot: SnapshotId,
        sources: Vec<V>,
        targets: Vec<V>,
    ) -> impl Future<Output = Result<OwnedNeighbourhoods<V, T>, Self::Error>> + Send;

    /// Fetch coreset-induced neighbourhood rows.
    ///
    /// The input `nodes` are the complete coreset node set for one trial. The
    /// returned rows must match `nodes.len()` and preserve input order. Returned
    /// neighbours should only include nodes from that same input set.
    fn coreset_neighbourhoods(
        &self,
        snapshot: SnapshotId,
        nodes: Vec<V>,
    ) -> impl Future<Output = Result<OwnedNeighbourhoods<V, T>, Self::Error>> + Send;
}

/// Graph-level errors that a concrete remote backend may wrap.
///
/// This enum intentionally does not try to represent transport failures,
/// protocol errors, or remote service failures. Those should live in the
/// backend's associated `Error` type, usually alongside or wrapping this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteGraphError {
    MissingNode,
    MissingEdge,
    SelfLoop,
}

impl fmt::Display for RemoteGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingNode => write!(f, "node was missing from the remote graph"),
            Self::MissingEdge => write!(f, "edge was missing from the remote graph"),
            Self::SelfLoop => write!(f, "self-loops are not supported by the remote graph"),
        }
    }
}

impl std::error::Error for RemoteGraphError {}
