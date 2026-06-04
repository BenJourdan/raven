use std::{
    collections::{HashMap, HashSet},
    fmt,
    hash::Hash,
    num::NonZeroUsize,
};

use raven_core::types::{FloatScalar, Strict, StrictCarrierOps};

pub(super) type AdjacencyMap<V, T> = HashMap<V, HashMap<V, Strict<T>>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InMemoryGraphError {
    MissingNode,
    MissingEdge,
    SelfLoop,
}

impl fmt::Display for InMemoryGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingNode => write!(f, "node was missing from the in-memory graph"),
            Self::MissingEdge => write!(f, "edge was missing from the in-memory graph"),
            Self::SelfLoop => write!(f, "self-loops are not supported by the in-memory graph"),
        }
    }
}

impl std::error::Error for InMemoryGraphError {}

/// A small undirected graph adapter for local tests and in-process use.
///
/// The graph owns update state and lends query-only [`super::InMemoryOracle`]
/// handles. Each oracle handle borrows the same graph snapshot and owns its own
/// scratch rows, so separate handles can be used by independent query trials.
#[derive(Debug, Clone)]
pub struct InMemoryUndirectedGraph<V, T> {
    pub(super) graph: AdjacencyMap<V, T>,
    node_ops: NodeOpsBuffer<V>,
}

#[derive(Debug, Clone)]
pub struct NodeOpsBuffer<V> {
    nodes: Vec<V>,
    seen: HashSet<V>,
    capacity: Option<NonZeroUsize>,
}

impl<V> Default for NodeOpsBuffer<V> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            seen: HashSet::new(),
            capacity: None,
        }
    }
}

impl<V> NodeOpsBuffer<V>
where
    V: Eq + Hash + Copy,
{
    pub fn with_capacity(capacity: NonZeroUsize) -> Self {
        Self {
            capacity: Some(capacity),
            ..Self::default()
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.capacity
            .is_some_and(|capacity| self.len() >= capacity.get())
    }

    fn touch(&mut self, node: V) {
        if self.seen.insert(node) {
            self.nodes.push(node);
        }
    }

    fn take_nodes(&mut self) -> Vec<V> {
        self.seen.clear();
        std::mem::take(&mut self.nodes)
    }
}

impl<V, T> Default for InMemoryUndirectedGraph<V, T> {
    fn default() -> Self {
        Self {
            graph: HashMap::new(),
            node_ops: NodeOpsBuffer::default(),
        }
    }
}

impl<V, T> InMemoryUndirectedGraph<V, T>
where
    V: Eq + Hash + Copy,
    T: FloatScalar,
    Strict<T>: StrictCarrierOps<Scalar = T> + Copy,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_node_ops_capacity(capacity: NonZeroUsize) -> Self {
        Self {
            node_ops: NodeOpsBuffer::with_capacity(capacity),
            ..Self::default()
        }
    }

    pub fn contains_node(&self, node: V) -> bool {
        self.graph.contains_key(&node)
    }

    pub fn node_ops_buffer_len(&self) -> usize {
        self.node_ops.len()
    }

    pub fn node_ops_buffer_is_empty(&self) -> bool {
        self.node_ops.is_empty()
    }

    pub fn node_ops_buffer_is_full(&self) -> bool {
        self.node_ops.is_full()
    }

    /// Insert, update, or delete an undirected edge.
    ///
    /// `Some(weight)` creates missing endpoints and stores the same weight in
    /// both adjacency directions. `None` removes the edge and drops endpoints
    /// that become isolated, since the core tree only stores connected nodes.
    pub fn update_edge(
        &mut self,
        u: V,
        v: V,
        weight: Option<Strict<T>>,
    ) -> Result<(), InMemoryGraphError> {
        if u == v {
            return Err(InMemoryGraphError::SelfLoop);
        }

        match weight {
            Some(weight) => {
                self.graph.entry(u).or_default().insert(v, weight);
                self.graph.entry(v).or_default().insert(u, weight);
            }
            None => {
                let removed_uv = self
                    .graph
                    .get_mut(&u)
                    .and_then(|row| row.remove(&v))
                    .is_some();
                let removed_vu = self
                    .graph
                    .get_mut(&v)
                    .and_then(|row| row.remove(&u))
                    .is_some();

                if !removed_uv && !removed_vu {
                    return Err(InMemoryGraphError::MissingEdge);
                }

                self.remove_if_isolated(u);
                self.remove_if_isolated(v);
            }
        }

        self.node_ops.touch(u);
        self.node_ops.touch(v);
        Ok(())
    }

    pub fn neighbours(&self, node: V) -> Option<Vec<(V, Strict<T>)>> {
        self.collect_neighbours(node)
    }

    pub fn degree(&self, node: V) -> Option<Strict<T>> {
        let total = self
            .graph
            .get(&node)?
            .values()
            .map(|weight| weight.into_scalar())
            .sum::<T>();
        Strict::<T>::from_positive_scalar(total).ok()
    }
    /// Flush buffered node degree diffs for the core algorithm.
    ///
    /// Nodes with no positive degree or no current adjacency row are returned
    /// as deletions because the core tree stores only connected nodes.
    pub fn flush_node_ops(&mut self) -> Vec<(V, Option<Strict<T>>)> {
        let nodes = self.node_ops.take_nodes();
        nodes
            .into_iter()
            .map(|node| (node, self.degree(node)))
            .collect()
    }

    fn remove_if_isolated(&mut self, node: V) {
        if self.graph.get(&node).is_some_and(HashMap::is_empty) {
            self.graph.remove(&node);
        }
    }

    pub(super) fn collect_neighbours(&self, node: V) -> Option<Vec<(V, Strict<T>)>> {
        self.graph.get(&node).map(|row| {
            row.iter()
                .map(|(&neighbour, &weight)| (neighbour, weight))
                .collect()
        })
    }
}
