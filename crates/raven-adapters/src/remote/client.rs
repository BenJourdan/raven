//! Client handles for remote graph backends.
//!
//! Users provide a transport-specific backend implementation. The client wraps
//! that backend with the Tokio runtime handle needed to build synchronous
//! `GraphOracle` handles for Raven's core query path.

use std::sync::Arc;

use super::{RemoteOracle, SnapshotId};

pub struct RemoteGraphClient<B> {
    backend: Arc<B>,
    runtime: tokio::runtime::Handle,
}

impl<B> Clone for RemoteGraphClient<B> {
    fn clone(&self) -> Self {
        Self {
            backend: self.backend.clone(),
            runtime: self.runtime.clone(),
        }
    }
}

impl<B> RemoteGraphClient<B> {
    pub fn new(backend: Arc<B>, runtime: tokio::runtime::Handle) -> Self {
        Self { backend, runtime }
    }

    pub fn from_backend(backend: B, runtime: tokio::runtime::Handle) -> Self {
        Self::new(Arc::new(backend), runtime)
    }

    pub fn backend(&self) -> &Arc<B> {
        &self.backend
    }

    pub fn runtime(&self) -> &tokio::runtime::Handle {
        &self.runtime
    }

    pub fn snapshot(&self, snapshot: SnapshotId) -> RemoteGraphSnapshot<B> {
        RemoteGraphSnapshot {
            backend: self.backend.clone(),
            runtime: self.runtime.clone(),
            snapshot,
        }
    }

    pub fn oracle<V, T>(&self, snapshot: SnapshotId) -> RemoteOracle<B, V, T> {
        RemoteOracle::new(self.backend.clone(), self.runtime.clone(), snapshot)
    }

    pub fn oracles<V, T>(&self, snapshot: SnapshotId, count: usize) -> Vec<RemoteOracle<B, V, T>> {
        (0..count).map(|_| self.oracle::<V, T>(snapshot)).collect()
    }

    /// Create temporary oracle handles for one snapshot and pass mutable
    /// references to a closure.
    ///
    /// This hides the small ownership dance needed by Raven's query API, which
    /// expects `&mut [&mut O]` while each remote oracle must own its scratch
    /// storage for the duration of the query.
    pub fn with_oracles<V, T, R>(
        &self,
        snapshot: SnapshotId,
        count: usize,
        f: impl FnOnce(&mut [&mut RemoteOracle<B, V, T>]) -> R,
    ) -> R {
        self.snapshot(snapshot).with_oracles(count, f)
    }
}

pub struct RemoteGraphSnapshot<B> {
    backend: Arc<B>,
    runtime: tokio::runtime::Handle,
    snapshot: SnapshotId,
}

impl<B> Clone for RemoteGraphSnapshot<B> {
    fn clone(&self) -> Self {
        Self {
            backend: self.backend.clone(),
            runtime: self.runtime.clone(),
            snapshot: self.snapshot,
        }
    }
}

impl<B> RemoteGraphSnapshot<B> {
    pub fn id(&self) -> SnapshotId {
        self.snapshot
    }

    pub fn oracle<V, T>(&self) -> RemoteOracle<B, V, T> {
        RemoteOracle::new(self.backend.clone(), self.runtime.clone(), self.snapshot)
    }

    pub fn oracles<V, T>(&self, count: usize) -> Vec<RemoteOracle<B, V, T>> {
        (0..count).map(|_| self.oracle::<V, T>()).collect()
    }

    /// Create temporary oracle handles for this snapshot and pass mutable
    /// references to a closure.
    ///
    /// The owned oracle handles live until the closure returns, so the closure
    /// may call Raven's synchronous query API directly and return its result.
    pub fn with_oracles<V, T, R>(
        &self,
        count: usize,
        f: impl FnOnce(&mut [&mut RemoteOracle<B, V, T>]) -> R,
    ) -> R {
        let mut oracles = self.oracles::<V, T>(count);
        let mut oracle_refs = oracles.iter_mut().collect::<Vec<_>>();
        f(&mut oracle_refs)
    }
}
