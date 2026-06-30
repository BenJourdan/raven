#[cfg(feature = "deep-query-timing")]
use std::time::Instant;
use std::{fmt, hash::Hash, num::NonZeroUsize, time::Duration};

use raven_core::{
    DynamicClusteringAlg,
    alg::{DynamicClustering, QueryTiming},
    types::{
        FloatScalar, NodeIdentity, NonStrict, NonStrictCarrierOps, PartitionOutput, PartitionType,
        Strict, StrictCarrierOps, TrialOutputMode,
    },
};
#[cfg(feature = "deep-query-timing")]
use raven_core::{GraphOracle, error::OracleError, types::Neighbourhoods};
use rustc_hash::{FxBuildHasher, FxHashMap};

use super::{InMemoryGraphError, graph::InMemoryUndirectedGraph};

pub struct InMemoryIndex<const ARITY: usize, V, T> {
    interner: NodeInterner<V>,
    graph: InMemoryUndirectedGraph<NodeIdentity, T>,
    clustering: DynamicClustering<ARITY, NodeIdentity, T>,
    last_oracle_timing: Option<InMemoryOracleTiming>,
}

#[derive(Debug, Clone)]
pub enum InMemoryIndexError {
    Graph(InMemoryGraphError),
    Core(String),
    UnknownNode,
    MissingExternalMapping(NodeIdentity),
}

impl fmt::Display for InMemoryIndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(err) => write!(f, "{err}"),
            Self::Core(err) => write!(f, "raven core operation failed: {err}"),
            Self::UnknownNode => write!(f, "node was not present in the in-memory index"),
            Self::MissingExternalMapping(node) => {
                write!(f, "internal node {node} had no external mapping")
            }
        }
    }
}

impl std::error::Error for InMemoryIndexError {}

impl From<InMemoryGraphError> for InMemoryIndexError {
    fn from(value: InMemoryGraphError) -> Self {
        Self::Graph(value)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct InMemoryOracleTiming {
    pub graph_calls: usize,
    pub graph_sources: usize,
    pub graph_edges: usize,
    pub graph_time: Duration,
    pub intersecting_calls: usize,
    pub intersecting_sources: usize,
    pub intersecting_targets: usize,
    pub intersecting_edges: usize,
    pub intersecting_time: Duration,
    pub coreset_calls: usize,
    pub coreset_sources: usize,
    pub coreset_edges: usize,
    pub coreset_time: Duration,
}

impl InMemoryOracleTiming {
    pub fn add(&mut self, other: Self) {
        self.graph_calls += other.graph_calls;
        self.graph_sources += other.graph_sources;
        self.graph_edges += other.graph_edges;
        self.graph_time += other.graph_time;

        self.intersecting_calls += other.intersecting_calls;
        self.intersecting_sources += other.intersecting_sources;
        self.intersecting_targets += other.intersecting_targets;
        self.intersecting_edges += other.intersecting_edges;
        self.intersecting_time += other.intersecting_time;

        self.coreset_calls += other.coreset_calls;
        self.coreset_sources += other.coreset_sources;
        self.coreset_edges += other.coreset_edges;
        self.coreset_time += other.coreset_time;
    }

    pub fn total_time(self) -> Duration {
        self.graph_time + self.intersecting_time + self.coreset_time
    }
}

#[derive(Debug)]
#[cfg(feature = "deep-query-timing")]
struct MeasuredOracle<O> {
    inner: O,
    timing: InMemoryOracleTiming,
}

#[cfg(feature = "deep-query-timing")]
impl<O> MeasuredOracle<O> {
    fn new(inner: O) -> Self {
        Self {
            inner,
            timing: InMemoryOracleTiming::default(),
        }
    }
}

#[cfg(feature = "deep-query-timing")]
impl<T, E, O> GraphOracle<NodeIdentity, T, E> for MeasuredOracle<O>
where
    O: GraphOracle<NodeIdentity, T, E>,
{
    fn graph_neighbourhoods<'a>(
        &'a mut self,
        nodes: &[NodeIdentity],
    ) -> Result<Neighbourhoods<'a, NodeIdentity, T>, OracleError<E>> {
        let started = Instant::now();
        let result = self.inner.graph_neighbourhoods(nodes);
        let elapsed = started.elapsed();
        let edges = result
            .as_ref()
            .map(Neighbourhoods::data)
            .map_or(0, |data| data.len());

        self.timing.graph_calls += 1;
        self.timing.graph_sources += nodes.len();
        self.timing.graph_edges += edges;
        self.timing.graph_time += elapsed;

        result
    }

    fn graph_neighbourhoods_intersecting<'a>(
        &'a mut self,
        sources: &[NodeIdentity],
        targets: &[NodeIdentity],
    ) -> Result<Neighbourhoods<'a, NodeIdentity, T>, OracleError<E>> {
        let started = Instant::now();
        let result = self
            .inner
            .graph_neighbourhoods_intersecting(sources, targets);
        let elapsed = started.elapsed();
        let edges = result
            .as_ref()
            .map(Neighbourhoods::data)
            .map_or(0, |data| data.len());

        self.timing.intersecting_calls += 1;
        self.timing.intersecting_sources += sources.len();
        self.timing.intersecting_targets += targets.len();
        self.timing.intersecting_edges += edges;
        self.timing.intersecting_time += elapsed;

        result
    }

    fn visit_graph_neighbourhoods_intersecting<F>(
        &mut self,
        sources: &[NodeIdentity],
        targets: &[NodeIdentity],
        mut visit: F,
    ) -> Result<usize, OracleError<E>>
    where
        F: FnMut(usize, NodeIdentity, Strict<T>),
        NodeIdentity: Copy,
        Strict<T>: Copy,
    {
        let started = Instant::now();
        let result = self.inner.visit_graph_neighbourhoods_intersecting(
            sources,
            targets,
            |row_idx, node, weight| {
                visit(row_idx, node, weight);
            },
        );
        let elapsed = started.elapsed();
        let edges = result.as_ref().copied().unwrap_or(0);

        self.timing.intersecting_calls += 1;
        self.timing.intersecting_sources += sources.len();
        self.timing.intersecting_targets += targets.len();
        self.timing.intersecting_edges += edges;
        self.timing.intersecting_time += elapsed;

        result
    }

    fn visit_graph_neighbourhoods_intersecting_with_target_indices<F>(
        &mut self,
        sources: &[NodeIdentity],
        targets: &[NodeIdentity],
        mut visit: F,
    ) -> Result<usize, OracleError<E>>
    where
        F: FnMut(usize, usize, NodeIdentity, Strict<T>),
        NodeIdentity: Copy + Eq + Hash,
        Strict<T>: Copy,
    {
        let started = Instant::now();
        let result = self
            .inner
            .visit_graph_neighbourhoods_intersecting_with_target_indices(
                sources,
                targets,
                |row_idx, target_idx, node, weight| {
                    visit(row_idx, target_idx, node, weight);
                },
            );
        let elapsed = started.elapsed();
        let edges = result.as_ref().copied().unwrap_or(0);

        self.timing.intersecting_calls += 1;
        self.timing.intersecting_sources += sources.len();
        self.timing.intersecting_targets += targets.len();
        self.timing.intersecting_edges += edges;
        self.timing.intersecting_time += elapsed;

        result
    }

    fn coreset_neighbourhoods<'a>(
        &'a mut self,
        nodes: &[NodeIdentity],
    ) -> Result<Neighbourhoods<'a, NodeIdentity, T>, OracleError<E>> {
        let started = Instant::now();
        let result = self.inner.coreset_neighbourhoods(nodes);
        let elapsed = started.elapsed();
        let edges = result
            .as_ref()
            .map(Neighbourhoods::data)
            .map_or(0, |data| data.len());

        self.timing.coreset_calls += 1;
        self.timing.coreset_sources += nodes.len();
        self.timing.coreset_edges += edges;
        self.timing.coreset_time += elapsed;

        result
    }

    fn visit_coreset_neighbourhoods_with_target_indices<F>(
        &mut self,
        nodes: &[NodeIdentity],
        mut visit: F,
    ) -> Result<usize, OracleError<E>>
    where
        F: FnMut(usize, usize, NodeIdentity, Strict<T>),
        NodeIdentity: Copy + Eq + Hash,
        Strict<T>: Copy,
    {
        let started = Instant::now();
        let result = self.inner.visit_coreset_neighbourhoods_with_target_indices(
            nodes,
            |row_idx, target_idx, node, weight| {
                visit(row_idx, target_idx, node, weight);
            },
        );
        let elapsed = started.elapsed();
        let edges = result.as_ref().copied().unwrap_or(0);

        self.timing.coreset_calls += 1;
        self.timing.coreset_sources += nodes.len();
        self.timing.coreset_edges += edges;
        self.timing.coreset_time += elapsed;

        result
    }
}

#[derive(Debug, Clone)]
pub(super) struct NodeInterner<V> {
    to_internal: FxHashMap<V, NodeIdentity>,
    to_external: Vec<Option<V>>,
    free: Vec<NodeIdentity>,
}

impl<V> Default for NodeInterner<V> {
    fn default() -> Self {
        Self {
            to_internal: FxHashMap::default(),
            to_external: Vec::new(),
            free: Vec::new(),
        }
    }
}

impl<V> NodeInterner<V>
where
    V: Eq + Hash + Clone,
{
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn with_capacity(capacity: NonZeroUsize) -> Self {
        Self {
            to_internal: FxHashMap::with_capacity_and_hasher(capacity.get(), FxBuildHasher),
            to_external: Vec::with_capacity(capacity.get()),
            free: Vec::new(),
        }
    }

    pub(super) fn get(&self, value: &V) -> Option<NodeIdentity> {
        self.to_internal.get(value).copied()
    }

    pub(super) fn external(&self, identity: NodeIdentity) -> Option<&V> {
        self.to_external.get(identity.index())?.as_ref()
    }

    pub(super) fn intern(&mut self, value: V) -> NodeIdentity {
        if let Some(identity) = self.get(&value) {
            return identity;
        }

        let identity = self.free.pop().unwrap_or_else(|| {
            let identity = NodeIdentity::from(self.to_external.len());
            self.to_external.push(None);
            identity
        });

        self.to_external[identity.index()] = Some(value.clone());
        self.to_internal.insert(value, identity);
        identity
    }

    pub(super) fn release(&mut self, identity: NodeIdentity) -> Option<V> {
        let external = self.to_external.get_mut(identity.index())?.take()?;
        self.to_internal.remove(&external);
        self.free.push(identity);
        Some(external)
    }
}

impl<const ARITY: usize, V, T> InMemoryIndex<ARITY, V, T>
where
    V: Eq + Hash + Clone,
    T: FloatScalar + Send + Sync,
    Strict<T>: StrictCarrierOps<Scalar = T> + Copy,
    NonStrict<T>: NonStrictCarrierOps<Scalar = T> + Copy,
{
    pub fn new(clustering: DynamicClustering<ARITY, NodeIdentity, T>) -> Self {
        Self {
            interner: NodeInterner::new(),
            graph: InMemoryUndirectedGraph::new(),
            clustering,
            last_oracle_timing: None,
        }
    }

    pub fn with_capacity(
        clustering: DynamicClustering<ARITY, NodeIdentity, T>,
        node_capacity: NonZeroUsize,
        expected_edges_per_node: NonZeroUsize,
        degree_rebuild_threshold: NonZeroUsize,
    ) -> Self {
        Self {
            interner: NodeInterner::with_capacity(node_capacity),
            graph: InMemoryUndirectedGraph::with_capacity(
                node_capacity,
                expected_edges_per_node,
                degree_rebuild_threshold,
            ),
            clustering,
            last_oracle_timing: None,
        }
    }

    pub fn update_edge(
        &mut self,
        u: V,
        v: V,
        weight: Option<Strict<T>>,
    ) -> Result<(), InMemoryIndexError> {
        if u == v {
            return Err(InMemoryGraphError::SelfLoop.into());
        }

        match weight {
            Some(weight) => {
                let u = self.interner.intern(u);
                let v = self.interner.intern(v);
                self.graph.update_edge(u, v, Some(weight))?;
            }
            None => {
                let u = self
                    .interner
                    .get(&u)
                    .ok_or(InMemoryGraphError::MissingEdge)?;
                let v = self
                    .interner
                    .get(&v)
                    .ok_or(InMemoryGraphError::MissingEdge)?;
                self.graph.update_edge(u, v, None)?;
            }
        }

        Ok(())
    }

    pub fn apply_pending_node_ops(&mut self) -> Result<(), InMemoryIndexError> {
        let node_ops = self.graph.flush_node_ops();
        if node_ops.is_empty() {
            return Ok(());
        }

        let deleted = node_ops
            .iter()
            .filter_map(|(node, degree)| degree.is_none().then_some(*node))
            .collect::<Vec<_>>();

        self.clustering
            .apply_node_ops(&node_ops)
            .map_err(|err| InMemoryIndexError::Core(err.to_string()))?;

        for node in deleted {
            if !self.graph.contains_node(node) {
                let _ = self.interner.release(node);
            }
        }

        Ok(())
    }

    pub fn query(
        &mut self,
        partition: PartitionType<'_, V>,
        mode: TrialOutputMode,
    ) -> Result<PartitionOutput<V, T>, InMemoryIndexError> {
        self.apply_pending_node_ops()?;
        self.last_oracle_timing = None;

        let internal_output = match partition {
            PartitionType::All => self.query_internal(PartitionType::All, mode)?,
            PartitionType::Subset(nodes) => {
                let internal_nodes = nodes
                    .iter()
                    .map(|node| self.internal_live_node(node))
                    .collect::<Result<Vec<_>, _>>()?;
                self.query_internal(PartitionType::Subset(&internal_nodes), mode)?
            }
        };

        self.map_output(internal_output)
    }

    pub fn contains_node(&self, node: &V) -> bool {
        self.interner
            .get(node)
            .is_some_and(|node| self.graph.contains_node(node))
    }

    pub fn live_node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn live_nodes(&self) -> Vec<V> {
        self.graph
            .nodes()
            .map(|node| {
                self.interner
                    .external(node)
                    .cloned()
                    .expect("live internal node should have an external mapping")
            })
            .collect()
    }

    pub fn last_query_timing(&self) -> Option<&QueryTiming> {
        self.clustering.last_query_timing()
    }

    pub fn last_oracle_timing(&self) -> Option<InMemoryOracleTiming> {
        self.last_oracle_timing
    }

    #[cfg(test)]
    pub(super) fn internal_id_for_test(&self, node: &V) -> Option<NodeIdentity> {
        self.interner.get(node)
    }

    fn internal_live_node(&self, node: &V) -> Result<NodeIdentity, InMemoryIndexError> {
        let node = self
            .interner
            .get(node)
            .ok_or(InMemoryIndexError::UnknownNode)?;
        if self.graph.contains_node(node) {
            Ok(node)
        } else {
            Err(InMemoryIndexError::UnknownNode)
        }
    }

    fn query_internal(
        &mut self,
        partition: PartitionType<'_, NodeIdentity>,
        mode: TrialOutputMode,
    ) -> Result<PartitionOutput<NodeIdentity, T>, InMemoryIndexError> {
        #[cfg(feature = "deep-query-timing")]
        let mut oracles = self
            .graph
            .dense_oracles(self.clustering.num_trials)
            .into_iter()
            .map(MeasuredOracle::new)
            .collect::<Vec<_>>();
        #[cfg(not(feature = "deep-query-timing"))]
        let mut oracles = self.graph.dense_oracles(self.clustering.num_trials);

        let mut oracle_refs = oracles.iter_mut().collect::<Vec<_>>();

        let output = self
            .clustering
            .query(partition, mode, &mut oracle_refs)
            .map_err(|err| InMemoryIndexError::Core(err.to_string()))?;

        #[cfg(feature = "deep-query-timing")]
        {
            let mut timing = InMemoryOracleTiming::default();
            for oracle in &oracles {
                timing.add(oracle.timing);
            }
            self.last_oracle_timing = Some(timing);
        }

        Ok(output)
    }

    fn map_output(
        &self,
        output: PartitionOutput<NodeIdentity, T>,
    ) -> Result<PartitionOutput<V, T>, InMemoryIndexError> {
        match output {
            PartitionOutput::All(nodes, trials) => {
                let nodes = nodes
                    .into_iter()
                    .map(|node| {
                        self.interner
                            .external(node)
                            .cloned()
                            .ok_or(InMemoryIndexError::MissingExternalMapping(node))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(PartitionOutput::All(nodes, trials))
            }
            PartitionOutput::Subset(trials) => Ok(PartitionOutput::Subset(trials)),
        }
    }
}
