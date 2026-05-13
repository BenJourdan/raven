use std::{
    collections::{HashMap, HashSet},
    fmt,
    hash::Hash,
    num::NonZeroUsize,
};

use raven_core::{
    GraphOracle,
    error::OracleError,
    types::{FloatScalar, Strict, StrictCarrierOps},
};

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
/// The graph oracle returns complete adjacency rows. The coreset oracle treats
/// its input batch as the complete current coreset and filters each adjacency
/// row to neighbours also present in that batch.
#[derive(Debug, Clone)]
pub struct InMemoryUndirectedGraph<V, T> {
    graph: HashMap<V, HashMap<V, Strict<T>>>,
    node_ops: NodeOpsBuffer<V>,
    scratch_rows: Vec<Vec<(V, Strict<T>)>>,
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
            scratch_rows: Vec::new(),
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

    fn collect_neighbours(&self, node: V) -> Option<Vec<(V, Strict<T>)>> {
        self.graph.get(&node).map(|row| {
            row.iter()
                .map(|(&neighbour, &weight)| (neighbour, weight))
                .collect()
        })
    }
}

impl<V, T> GraphOracle<V, T, InMemoryGraphError> for InMemoryUndirectedGraph<V, T>
where
    V: Eq + Hash + Copy,
    T: FloatScalar,
    Strict<T>: StrictCarrierOps<Scalar = T> + Copy,
{
    fn graph_neighbourhoods<'a>(
        &'a mut self,
        nodes: &'a [V],
    ) -> Result<Vec<&'a [(V, Strict<T>)]>, OracleError<InMemoryGraphError>> {
        self.scratch_rows.clear();
        self.scratch_rows.reserve(nodes.len());

        for node in nodes {
            let row = self
                .collect_neighbours(*node)
                .ok_or(OracleError::GraphError(InMemoryGraphError::MissingNode))?;
            self.scratch_rows.push(row);
        }

        Ok(self.scratch_rows.iter().map(Vec::as_slice).collect())
    }

    fn coreset_neighbourhoods<'a>(
        &'a mut self,
        nodes: &'a [V],
    ) -> Result<Vec<&'a [(V, Strict<T>)]>, OracleError<InMemoryGraphError>> {
        let coreset_members = nodes.iter().copied().collect::<HashSet<_>>();

        self.scratch_rows.clear();
        self.scratch_rows.reserve(nodes.len());

        for node in nodes {
            let mut row = self
                .collect_neighbours(*node)
                .ok_or(OracleError::CoresetError(InMemoryGraphError::MissingNode))?;
            row.retain(|(neighbour, _)| coreset_members.contains(neighbour));
            self.scratch_rows.push(row);
        }

        Ok(self.scratch_rows.iter().map(Vec::as_slice).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raven_core::{
        DynamicClusteringAlg,
        alg::{DynamicClustering, TreeData},
        types::{AlgType, PartitionOutput, PartitionType},
    };
    use std::sync::Arc;

    type TestClustering = DynamicClustering<2, usize, f64>;

    fn strict(value: f64) -> Strict<f64> {
        Strict::<f64>::new(value).unwrap()
    }

    fn test_clustering() -> TestClustering {
        let cluster_alg: AlgType<f64> = Arc::new(|graph, _| {
            let n = graph.symbolic().nrows();
            (vec![0; n], 1)
        });

        DynamicClustering {
            node_to_tree_map: Default::default(),
            tree_to_node_map: Default::default(),
            degrees: Default::default(),
            tree_data: TreeData {
                timestamp: vec![],
                volume: vec![],
                size: vec![],
                f_delta: vec![],
                h_b: vec![],
                h_s: vec![],
            },
            sigma: strict(1.0),
            timestamp: 0,
            coreset_size: 3,
            sampling_seeds: 2,
            num_clusters: 1,
            cluster_alg,
            prop_name: String::from("w"),
        }
    }

    #[test]
    fn graph_oracle_returns_full_adjacency_rows() {
        let mut graph = InMemoryUndirectedGraph::<usize, f64>::new();
        graph.update_edge(1, 2, Some(strict(1.5))).unwrap();
        graph.update_edge(1, 3, Some(strict(2.5))).unwrap();

        let rows = graph.graph_neighbourhoods(&[1, 2]).unwrap();

        assert_eq!(rows.len(), 2);
        assert!(rows[0].contains(&(2, strict(1.5))));
        assert!(rows[0].contains(&(3, strict(2.5))));
        assert_eq!(rows[1], &[(1, strict(1.5))]);
    }

    #[test]
    fn reversed_edge_updates_the_same_undirected_relationship() {
        let mut graph = InMemoryUndirectedGraph::<usize, f64>::new();
        graph.update_edge(1, 2, Some(strict(1.5))).unwrap();
        graph.update_edge(2, 1, Some(strict(2.5))).unwrap();

        assert_eq!(graph.degree(1), Some(strict(2.5)));
        assert_eq!(graph.degree(2), Some(strict(2.5)));

        let rows = graph.graph_neighbourhoods(&[1, 2]).unwrap();
        assert_eq!(rows[0], &[(2, strict(2.5))]);
        assert_eq!(rows[1], &[(1, strict(2.5))]);
    }

    #[test]
    fn coreset_oracle_filters_to_input_batch() {
        let mut graph = InMemoryUndirectedGraph::<usize, f64>::new();
        graph.update_edge(1, 2, Some(strict(1.0))).unwrap();
        graph.update_edge(1, 3, Some(strict(2.0))).unwrap();
        graph.update_edge(2, 3, Some(strict(3.0))).unwrap();

        let rows = graph.coreset_neighbourhoods(&[1, 3]).unwrap();

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], &[(3, strict(2.0))]);
        assert_eq!(rows[1], &[(1, strict(2.0))]);
    }

    #[test]
    fn flush_node_ops_reports_updated_and_deleted_nodes() {
        let mut graph = InMemoryUndirectedGraph::<usize, f64>::new();
        graph.update_edge(1, 2, Some(strict(4.0))).unwrap();

        assert_eq!(graph.degree(1), Some(strict(4.0)));

        let initial_ops = graph.flush_node_ops();
        assert_eq!(
            initial_ops,
            vec![(1, Some(strict(4.0))), (2, Some(strict(4.0)))]
        );
        assert!(graph.node_ops_buffer_is_empty());

        graph.update_edge(1, 2, None).unwrap();

        assert_eq!(graph.flush_node_ops(), vec![(1, None), (2, None)]);
    }

    #[test]
    fn node_ops_buffer_tracks_unique_touched_nodes_until_flush() {
        let mut graph = InMemoryUndirectedGraph::<usize, f64>::with_node_ops_capacity(
            NonZeroUsize::new(3).unwrap(),
        );

        graph.update_edge(1, 2, Some(strict(1.0))).unwrap();
        assert_eq!(graph.node_ops_buffer_len(), 2);
        assert!(!graph.node_ops_buffer_is_full());

        graph.update_edge(2, 3, Some(strict(1.0))).unwrap();
        assert_eq!(graph.node_ops_buffer_len(), 3);
        assert!(graph.node_ops_buffer_is_full());

        let ops = graph.flush_node_ops();
        assert_eq!(ops.len(), 3);
        assert!(graph.node_ops_buffer_is_empty());
    }

    #[test]
    fn graph_updates_flush_into_core_queries() {
        let mut graph = InMemoryUndirectedGraph::<usize, f64>::new();
        graph.update_edge(1, 2, Some(strict(1.0))).unwrap();
        graph.update_edge(2, 3, Some(strict(1.0))).unwrap();
        graph.update_edge(3, 4, Some(strict(1.0))).unwrap();
        graph.update_edge(4, 5, Some(strict(1.0))).unwrap();
        graph.update_edge(5, 6, Some(strict(1.0))).unwrap();

        let mut clustering = test_clustering();
        let initial_ops = graph.flush_node_ops();
        clustering.apply_node_ops(&initial_ops).unwrap();

        let output = clustering.query(PartitionType::All, &mut graph).unwrap();
        match output {
            PartitionOutput::All(nodes, labels, num_clusters) => {
                assert_eq!(num_clusters, 1);
                assert_eq!(nodes.len(), 6);
                assert_eq!(labels.len(), nodes.len());
                assert!(nodes.contains(&1));
                assert!(nodes.contains(&6));
            }
            PartitionOutput::Subset(_, _) => panic!("expected all-node query output"),
        }

        graph.update_edge(5, 6, None).unwrap();
        graph.update_edge(4, 7, Some(strict(2.0))).unwrap();

        let update_ops = graph.flush_node_ops();
        clustering.apply_node_ops(&update_ops).unwrap();

        let output = clustering.query(PartitionType::All, &mut graph).unwrap();
        match output {
            PartitionOutput::All(nodes, labels, num_clusters) => {
                assert_eq!(num_clusters, 1);
                assert_eq!(nodes.len(), 6);
                assert_eq!(labels.len(), nodes.len());
                assert!(nodes.contains(&7));
                assert!(!nodes.contains(&6));
            }
            PartitionOutput::Subset(_, _) => panic!("expected all-node query output"),
        }
    }
}
