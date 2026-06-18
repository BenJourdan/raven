use raven_core::types::Strict;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SnapshotId(pub u64);

/// Node-degree diffs that correspond to one queryable remote graph snapshot.
///
/// The returned node operations are ready to pass to
/// [`raven_core::DynamicClusteringAlg::apply_node_ops`]. `None` means the node
/// is now absent from the positive-degree graph snapshot. Application code is
/// responsible for ensuring these node operations describe the same snapshot
/// that remote oracles will later query.
#[derive(Debug, Clone)]
pub struct RemoteSnapshotNodeOps<V, T> {
    pub snapshot: SnapshotId,
    pub node_ops: Vec<(V, Option<Strict<T>>)>,
}

impl<V, T> RemoteSnapshotNodeOps<V, T> {
    pub fn new(snapshot: SnapshotId, node_ops: Vec<(V, Option<Strict<T>>)>) -> Self {
        Self { snapshot, node_ops }
    }

    pub fn node_ops(&self) -> &[(V, Option<Strict<T>>)] {
        &self.node_ops
    }

    pub fn into_node_ops(self) -> Vec<(V, Option<Strict<T>>)> {
        self.node_ops
    }
}
