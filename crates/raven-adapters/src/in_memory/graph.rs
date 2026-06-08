use std::{fmt, hash::Hash, num::NonZeroUsize};

use raven_core::types::{FloatScalar, Strict, StrictCarrierOps};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

pub(super) type AdjacencyMap<V, T> = FxHashMap<V, AdjacencyRow<V, T>>;

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
    adjacency_row_capacity: Option<NonZeroUsize>,
    degree_rebuild_threshold: Option<NonZeroUsize>,
}

#[derive(Debug, Clone)]
pub(super) struct AdjacencyRow<V, T> {
    degree: CachedDegree<T>,
    pub(super) neighbours: FxHashMap<V, Strict<T>>,
}

#[derive(Debug, Clone, Copy)]
struct CachedDegree<T> {
    value: T,
    dirty_updates: usize,
}

#[derive(Debug, Clone)]
pub struct NodeOpsBuffer<V> {
    nodes: Vec<V>,
    seen: FxHashSet<V>,
    capacity: Option<NonZeroUsize>,
}

impl<V> Default for NodeOpsBuffer<V> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            seen: FxHashSet::default(),
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
            nodes: Vec::with_capacity(capacity.get()),
            seen: FxHashSet::with_capacity_and_hasher(capacity.get(), FxBuildHasher),
            capacity: Some(capacity),
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
            graph: FxHashMap::default(),
            node_ops: NodeOpsBuffer::default(),
            adjacency_row_capacity: None,
            degree_rebuild_threshold: None,
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

    pub fn with_capacity(
        capacity: NonZeroUsize,
        expected_edges_per_node: NonZeroUsize,
        degree_rebuild_threshold: NonZeroUsize,
    ) -> Self {
        Self {
            graph: FxHashMap::with_capacity_and_hasher(capacity.get(), FxBuildHasher),
            node_ops: NodeOpsBuffer::with_capacity(capacity),
            adjacency_row_capacity: Some(expected_edges_per_node),
            degree_rebuild_threshold: Some(degree_rebuild_threshold),
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
                let adjacency_row_capacity = self.adjacency_row_capacity;
                let cache_enabled = self.degree_rebuild_threshold.is_some();

                self.graph
                    .entry(u)
                    .or_insert_with(|| AdjacencyRow::with_capacity(adjacency_row_capacity))
                    .insert(v, weight, cache_enabled);
                self.graph
                    .entry(v)
                    .or_insert_with(|| AdjacencyRow::with_capacity(adjacency_row_capacity))
                    .insert(u, weight, cache_enabled);
            }
            None => {
                let cache_enabled = self.degree_rebuild_threshold.is_some();
                let removed_uv = self
                    .graph
                    .get_mut(&u)
                    .and_then(|row| row.remove(v, cache_enabled));
                let removed_vu = self
                    .graph
                    .get_mut(&v)
                    .and_then(|row| row.remove(u, cache_enabled));

                if removed_uv.or(removed_vu).is_none() {
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

    pub fn degree(&mut self, node: V) -> Option<Strict<T>> {
        let degree = self.degree_scalar(node)?;
        Strict::<T>::from_positive_scalar(degree)
            .ok()
            .or_else(|| self.rebuild_degree(node).and_then(strict_degree))
    }

    pub fn edge_weight(&self, u: V, v: V) -> Option<Strict<T>> {
        self.graph.get(&u)?.neighbours.get(&v).copied()
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
        if self.graph.get(&node).is_some_and(AdjacencyRow::is_isolated) {
            self.graph.remove(&node);
        }
    }

    fn degree_scalar(&mut self, node: V) -> Option<T> {
        if self.degree_rebuild_threshold.is_none() {
            return self.exact_degree_scalar(node);
        }

        let threshold = self.degree_rebuild_threshold.expect("checked above").get();
        let row = self.graph.get_mut(&node)?;

        if row.degree.dirty_updates >= threshold {
            Some(row.rebuild_degree())
        } else {
            Some(row.degree.value)
        }
    }

    fn exact_degree_scalar(&self, node: V) -> Option<T> {
        Some(self.graph.get(&node)?.exact_degree())
    }

    fn rebuild_degree(&mut self, node: V) -> Option<T> {
        Some(self.graph.get_mut(&node)?.rebuild_degree())
    }

    pub(super) fn collect_neighbours(&self, node: V) -> Option<Vec<(V, Strict<T>)>> {
        self.graph.get(&node).map(|row| {
            row.neighbours
                .iter()
                .map(|(&neighbour, &weight)| (neighbour, weight))
                .collect()
        })
    }
}

impl<V, T> AdjacencyRow<V, T>
where
    V: Eq + Hash + Copy,
    T: FloatScalar,
    Strict<T>: StrictCarrierOps<Scalar = T> + Copy,
{
    fn with_capacity(capacity: Option<NonZeroUsize>) -> Self {
        Self {
            degree: CachedDegree::zero(),
            neighbours: capacity.map_or_else(FxHashMap::default, |capacity| {
                FxHashMap::with_capacity_and_hasher(capacity.get(), FxBuildHasher)
            }),
        }
    }

    fn insert(
        &mut self,
        neighbour: V,
        weight: Strict<T>,
        update_cached_degree: bool,
    ) -> Option<Strict<T>> {
        let old_weight = self.neighbours.insert(neighbour, weight);

        if update_cached_degree {
            let old_weight = old_weight
                .map(|weight| weight.into_scalar())
                .unwrap_or(T::ZERO);
            self.degree.add_delta(weight.into_scalar() - old_weight);
        }

        old_weight
    }

    fn remove(&mut self, neighbour: V, update_cached_degree: bool) -> Option<Strict<T>> {
        let removed_weight = self.neighbours.remove(&neighbour);

        if update_cached_degree {
            if let Some(weight) = removed_weight {
                self.degree.add_delta(-weight.into_scalar());
            }
        }

        removed_weight
    }

    fn is_isolated(&self) -> bool {
        self.neighbours.is_empty()
    }

    fn exact_degree(&self) -> T {
        self.neighbours
            .values()
            .map(|weight| weight.into_scalar())
            .sum::<T>()
    }

    fn rebuild_degree(&mut self) -> T {
        let value = self.exact_degree();
        self.degree = CachedDegree {
            value,
            dirty_updates: 0,
        };
        value
    }
}

impl<T> CachedDegree<T>
where
    T: FloatScalar,
{
    fn zero() -> Self {
        Self {
            value: T::ZERO,
            dirty_updates: 0,
        }
    }

    fn add_delta(&mut self, delta: T) {
        if delta == T::ZERO {
            return;
        }

        self.value = self.value + delta;
        self.dirty_updates = self.dirty_updates.saturating_add(1);
    }
}

fn strict_degree<T>(degree: T) -> Option<Strict<T>>
where
    T: FloatScalar,
    Strict<T>: StrictCarrierOps<Scalar = T>,
{
    Strict::<T>::from_positive_scalar(degree).ok()
}
