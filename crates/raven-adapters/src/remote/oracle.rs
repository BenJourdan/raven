//! Query oracle handles for future IPC and network graph backends.
//!
//! Remote oracle handles are expected to own per-trial scratch storage while
//! sharing a client and cache with other handles.

use std::sync::Arc;

use raven_core::{GraphOracle, error::OracleError, types::Neighbourhoods};

use super::{OwnedNeighbourhoods, RemoteGraphBackend, SnapshotId};

pub struct RemoteOracle<B, V, T> {
    backend: Arc<B>,
    runtime: tokio::runtime::Handle,
    snapshot: SnapshotId,
    rows: OwnedNeighbourhoods<V, T>,
}

impl<B, V, T> RemoteOracle<B, V, T> {
    pub fn new(backend: Arc<B>, runtime: tokio::runtime::Handle, snapshot: SnapshotId) -> Self {
        Self {
            backend,
            runtime,
            snapshot,
            rows: OwnedNeighbourhoods::empty(),
        }
    }

    pub fn snapshot(&self) -> SnapshotId {
        self.snapshot
    }
}

impl<B, V, T> GraphOracle<V, T, B::Error> for RemoteOracle<B, V, T>
where
    V: Copy,
    B: RemoteGraphBackend<V, T>,
{
    fn graph_neighbourhoods<'a>(
        &'a mut self,
        nodes: &[V],
    ) -> Result<Neighbourhoods<'a, V, T>, OracleError<B::Error>> {
        self.rows = self
            .runtime
            .block_on(
                self.backend
                    .graph_neighbourhoods(self.snapshot, nodes.to_vec()),
            )
            .map_err(OracleError::GraphError)?;
        Ok(self.rows.as_borrowed())
    }

    fn graph_neighbourhoods_intersecting<'a>(
        &'a mut self,
        sources: &[V],
        targets: &[V],
    ) -> Result<Neighbourhoods<'a, V, T>, OracleError<B::Error>> {
        self.rows = self
            .runtime
            .block_on(self.backend.graph_neighbourhoods_intersecting(
                self.snapshot,
                sources.to_vec(),
                targets.to_vec(),
            ))
            .map_err(OracleError::GraphError)?;
        Ok(self.rows.as_borrowed())
    }

    fn coreset_neighbourhoods<'a>(
        &'a mut self,
        nodes: &[V],
    ) -> Result<Neighbourhoods<'a, V, T>, OracleError<B::Error>> {
        self.rows = self
            .runtime
            .block_on(
                self.backend
                    .coreset_neighbourhoods(self.snapshot, nodes.to_vec()),
            )
            .map_err(OracleError::CoresetError)?;
        Ok(self.rows.as_borrowed())
    }
}
