use std::hash::Hash;

use raven_core::{
    GraphOracle,
    error::OracleError,
    types::{FloatScalar, Neighbourhoods, NodeIdentity, Strict, StrictCarrierOps},
};
use rustc_hash::{FxHashMap, FxHashSet};

use super::graph::{AdjacencyMap, InMemoryGraphError, InMemoryUndirectedGraph};

/// Query-only oracle handle borrowed from an [`InMemoryUndirectedGraph`].
///
/// Each handle owns its scratch data. Multiple handles may therefore be used
/// by parallel trials while they all observe the same immutably borrowed graph.
/// Returned neighbourhood rows borrow this scratch storage and remain valid
/// until the next oracle call on the same handle.
#[derive(Debug)]
pub struct InMemoryOracle<'a, V, T> {
    graph: &'a AdjacencyMap<V, T>,
    scratch_data: Vec<(V, Strict<T>)>,
    scratch_offsets: Vec<usize>,
}

/// Query-only oracle for dense [`NodeIdentity`] graph storage.
///
/// The dense oracle uses an epoch-marked membership table for target-set
/// filtering, avoiding repeated hash-set construction in intersection-heavy
/// query paths.
#[derive(Debug)]
pub struct DenseInMemoryOracle<'a, T> {
    graph: &'a AdjacencyMap<NodeIdentity, T>,
    scratch_data: Vec<(NodeIdentity, Strict<T>)>,
    scratch_offsets: Vec<usize>,
    marker: DenseMarker,
}

#[derive(Debug, Default)]
pub(super) struct DenseMarker {
    marks: Vec<u32>,
    ordinals: Vec<usize>,
    epoch: u32,
}

impl<'a, V, T> InMemoryOracle<'a, V, T> {
    fn new(graph: &'a AdjacencyMap<V, T>) -> Self {
        Self {
            graph,
            scratch_data: Vec::new(),
            scratch_offsets: Vec::new(),
        }
    }
}

impl<'a, T> DenseInMemoryOracle<'a, T> {
    fn new(graph: &'a AdjacencyMap<NodeIdentity, T>) -> Self {
        Self {
            graph,
            scratch_data: Vec::new(),
            scratch_offsets: Vec::new(),
            marker: DenseMarker::default(),
        }
    }
}

impl DenseMarker {
    pub(super) fn mark_all(&mut self, nodes: &[NodeIdentity]) {
        self.mark_all_with_ordinals(nodes);
    }

    pub(super) fn mark_all_with_ordinals(&mut self, nodes: &[NodeIdentity]) {
        if self.epoch == u32::MAX {
            self.marks.fill(0);
            self.epoch = 1;
        } else {
            self.epoch += 1;
        }

        let required_len = nodes
            .iter()
            .map(|node| node.index())
            .max()
            .map_or(0, |max_index| max_index + 1);
        if self.marks.len() < required_len {
            self.marks.resize(required_len, 0);
        }
        if self.ordinals.len() < required_len {
            self.ordinals.resize(required_len, 0);
        }

        for (ordinal, node) in nodes.iter().enumerate() {
            let index = node.index();
            if self.marks[index] == self.epoch {
                continue;
            }
            self.marks[index] = self.epoch;
            self.ordinals[index] = ordinal;
        }
    }

    pub(super) fn contains(&self, node: NodeIdentity) -> bool {
        self.marks
            .get(node.index())
            .is_some_and(|epoch| *epoch == self.epoch)
    }

    pub(super) fn ordinal(&self, node: NodeIdentity) -> Option<usize> {
        let index = node.index();
        self.marks
            .get(index)
            .is_some_and(|epoch| *epoch == self.epoch)
            .then(|| self.ordinals[index])
    }

    #[cfg(test)]
    pub(super) fn epoch(&self) -> u32 {
        self.epoch
    }

    #[cfg(test)]
    pub(super) fn set_epoch(&mut self, epoch: u32) {
        self.epoch = epoch;
    }
}

impl<V, T> InMemoryUndirectedGraph<V, T> {
    /// Borrow one query oracle with its own scratch storage.
    pub fn oracle(&self) -> InMemoryOracle<'_, V, T> {
        InMemoryOracle::new(&self.graph)
    }

    /// Borrow one independent query oracle per requested trial.
    pub fn oracles(&self, count: usize) -> Vec<InMemoryOracle<'_, V, T>> {
        std::iter::repeat_with(|| self.oracle())
            .take(count)
            .collect()
    }
}

impl<T> InMemoryUndirectedGraph<NodeIdentity, T> {
    /// Borrow one dense query oracle with its own scratch storage.
    pub fn dense_oracle(&self) -> DenseInMemoryOracle<'_, T> {
        DenseInMemoryOracle::new(&self.graph)
    }

    /// Borrow one independent dense query oracle per requested trial.
    pub fn dense_oracles(&self, count: usize) -> Vec<DenseInMemoryOracle<'_, T>> {
        std::iter::repeat_with(|| self.dense_oracle())
            .take(count)
            .collect()
    }
}

impl<V, T> GraphOracle<V, T, InMemoryGraphError> for InMemoryOracle<'_, V, T>
where
    V: Eq + Hash + Copy,
    T: FloatScalar,
    Strict<T>: StrictCarrierOps<Scalar = T> + Copy,
{
    fn graph_neighbourhoods<'a>(
        &'a mut self,
        nodes: &[V],
    ) -> Result<Neighbourhoods<'a, V, T>, OracleError<InMemoryGraphError>> {
        self.scratch_data.clear();
        self.scratch_offsets.clear();
        self.scratch_offsets.reserve(nodes.len() + 1);
        self.scratch_offsets.push(0);

        for node in nodes {
            let row = self
                .graph
                .get(node)
                .ok_or(OracleError::GraphError(InMemoryGraphError::MissingNode))?;
            self.scratch_data.extend(
                row.neighbours
                    .iter()
                    .map(|(&neighbour, &weight)| (neighbour, weight)),
            );
            self.scratch_offsets.push(self.scratch_data.len());
        }

        Ok(Neighbourhoods::new(
            &self.scratch_data,
            &self.scratch_offsets,
        ))
    }

    fn graph_neighbourhoods_intersecting<'a>(
        &'a mut self,
        sources: &[V],
        targets: &[V],
    ) -> Result<Neighbourhoods<'a, V, T>, OracleError<InMemoryGraphError>> {
        let target_set = targets.iter().copied().collect::<FxHashSet<_>>();

        self.scratch_data.clear();
        self.scratch_offsets.clear();
        self.scratch_offsets.reserve(sources.len() + 1);
        self.scratch_offsets.push(0);

        for source in sources {
            let row = self
                .graph
                .get(source)
                .ok_or(OracleError::GraphError(InMemoryGraphError::MissingNode))?;

            if target_set.len() < row.neighbours.len() {
                self.scratch_data
                    .extend(target_set.iter().filter_map(|&target| {
                        row.neighbours.get(&target).map(|&weight| (target, weight))
                    }));
            } else {
                self.scratch_data.extend(
                    row.neighbours
                        .iter()
                        .filter(|(neighbour, _)| target_set.contains(neighbour))
                        .map(|(&neighbour, &weight)| (neighbour, weight)),
                );
            }

            self.scratch_offsets.push(self.scratch_data.len());
        }

        Ok(Neighbourhoods::new(
            &self.scratch_data,
            &self.scratch_offsets,
        ))
    }

    fn visit_graph_neighbourhoods_intersecting<F>(
        &mut self,
        sources: &[V],
        targets: &[V],
        mut visit: F,
    ) -> Result<usize, OracleError<InMemoryGraphError>>
    where
        F: FnMut(usize, V, Strict<T>),
        V: Copy,
        Strict<T>: Copy,
    {
        let target_set = targets.iter().copied().collect::<FxHashSet<_>>();
        let mut edges = 0usize;

        for (row_idx, source) in sources.iter().enumerate() {
            let row = self
                .graph
                .get(source)
                .ok_or(OracleError::GraphError(InMemoryGraphError::MissingNode))?;

            if target_set.len() < row.neighbours.len() {
                for &target in &target_set {
                    if let Some(&weight) = row.neighbours.get(&target) {
                        visit(row_idx, target, weight);
                        edges += 1;
                    }
                }
            } else {
                for (&neighbour, &weight) in row.neighbours.iter() {
                    if target_set.contains(&neighbour) {
                        visit(row_idx, neighbour, weight);
                        edges += 1;
                    }
                }
            }
        }

        Ok(edges)
    }

    fn visit_graph_neighbourhoods_intersecting_with_target_indices<F>(
        &mut self,
        sources: &[V],
        targets: &[V],
        mut visit: F,
    ) -> Result<usize, OracleError<InMemoryGraphError>>
    where
        F: FnMut(usize, usize, V, Strict<T>),
        V: Copy + Eq + Hash,
        Strict<T>: Copy,
    {
        let mut target_indices = FxHashMap::<V, usize>::default();
        for (idx, target) in targets.iter().copied().enumerate() {
            target_indices.entry(target).or_insert(idx);
        }

        let mut edges = 0usize;
        for (row_idx, source) in sources.iter().enumerate() {
            let row = self
                .graph
                .get(source)
                .ok_or(OracleError::GraphError(InMemoryGraphError::MissingNode))?;

            if target_indices.len() < row.neighbours.len() {
                for (&target, &target_idx) in target_indices.iter() {
                    if let Some(&weight) = row.neighbours.get(&target) {
                        visit(row_idx, target_idx, target, weight);
                        edges += 1;
                    }
                }
            } else {
                for (&neighbour, &weight) in row.neighbours.iter() {
                    if let Some(&target_idx) = target_indices.get(&neighbour) {
                        visit(row_idx, target_idx, neighbour, weight);
                        edges += 1;
                    }
                }
            }
        }

        Ok(edges)
    }

    fn coreset_neighbourhoods<'a>(
        &'a mut self,
        nodes: &[V],
    ) -> Result<Neighbourhoods<'a, V, T>, OracleError<InMemoryGraphError>> {
        let coreset_members = nodes.iter().copied().collect::<FxHashSet<_>>();

        self.scratch_data.clear();
        self.scratch_offsets.clear();
        self.scratch_offsets.reserve(nodes.len() + 1);
        self.scratch_offsets.push(0);

        for node in nodes {
            let row = self
                .graph
                .get(node)
                .ok_or(OracleError::CoresetError(InMemoryGraphError::MissingNode))?;
            self.scratch_data.extend(
                row.neighbours
                    .iter()
                    .filter(|(neighbour, _)| coreset_members.contains(neighbour))
                    .map(|(&neighbour, &weight)| (neighbour, weight)),
            );
            self.scratch_offsets.push(self.scratch_data.len());
        }

        Ok(Neighbourhoods::new(
            &self.scratch_data,
            &self.scratch_offsets,
        ))
    }

    fn visit_coreset_neighbourhoods_with_target_indices<F>(
        &mut self,
        nodes: &[V],
        mut visit: F,
    ) -> Result<usize, OracleError<InMemoryGraphError>>
    where
        F: FnMut(usize, usize, V, Strict<T>),
        V: Copy + Eq + Hash,
        Strict<T>: Copy,
    {
        let mut target_indices = FxHashMap::<V, usize>::default();
        for (idx, node) in nodes.iter().copied().enumerate() {
            target_indices.entry(node).or_insert(idx);
        }

        let mut edges = 0usize;
        for (row_idx, node) in nodes.iter().enumerate() {
            let row = self
                .graph
                .get(node)
                .ok_or(OracleError::CoresetError(InMemoryGraphError::MissingNode))?;

            if target_indices.len() < row.neighbours.len() {
                for (&target, &target_idx) in target_indices.iter() {
                    if let Some(&weight) = row.neighbours.get(&target) {
                        visit(row_idx, target_idx, target, weight);
                        edges += 1;
                    }
                }
            } else {
                for (&neighbour, &weight) in row.neighbours.iter() {
                    if let Some(&target_idx) = target_indices.get(&neighbour) {
                        visit(row_idx, target_idx, neighbour, weight);
                        edges += 1;
                    }
                }
            }
        }

        Ok(edges)
    }
}

impl<T> GraphOracle<NodeIdentity, T, InMemoryGraphError> for DenseInMemoryOracle<'_, T>
where
    T: FloatScalar,
    Strict<T>: StrictCarrierOps<Scalar = T> + Copy,
{
    fn graph_neighbourhoods<'a>(
        &'a mut self,
        nodes: &[NodeIdentity],
    ) -> Result<Neighbourhoods<'a, NodeIdentity, T>, OracleError<InMemoryGraphError>> {
        self.scratch_data.clear();
        self.scratch_offsets.clear();
        self.scratch_offsets.reserve(nodes.len() + 1);
        self.scratch_offsets.push(0);

        for node in nodes {
            let row = self
                .graph
                .get(node)
                .ok_or(OracleError::GraphError(InMemoryGraphError::MissingNode))?;
            self.scratch_data.extend(
                row.neighbours
                    .iter()
                    .map(|(&neighbour, &weight)| (neighbour, weight)),
            );
            self.scratch_offsets.push(self.scratch_data.len());
        }

        Ok(Neighbourhoods::new(
            &self.scratch_data,
            &self.scratch_offsets,
        ))
    }

    fn graph_neighbourhoods_intersecting<'a>(
        &'a mut self,
        sources: &[NodeIdentity],
        targets: &[NodeIdentity],
    ) -> Result<Neighbourhoods<'a, NodeIdentity, T>, OracleError<InMemoryGraphError>> {
        self.marker.mark_all(targets);

        self.scratch_data.clear();
        self.scratch_offsets.clear();
        self.scratch_offsets.reserve(sources.len() + 1);
        self.scratch_offsets.push(0);

        for source in sources {
            let row = self
                .graph
                .get(source)
                .ok_or(OracleError::GraphError(InMemoryGraphError::MissingNode))?;
            self.scratch_data.extend(
                row.neighbours
                    .iter()
                    .filter(|(neighbour, _)| self.marker.contains(**neighbour))
                    .map(|(&neighbour, &weight)| (neighbour, weight)),
            );
            self.scratch_offsets.push(self.scratch_data.len());
        }

        Ok(Neighbourhoods::new(
            &self.scratch_data,
            &self.scratch_offsets,
        ))
    }

    fn visit_graph_neighbourhoods_intersecting<F>(
        &mut self,
        sources: &[NodeIdentity],
        targets: &[NodeIdentity],
        mut visit: F,
    ) -> Result<usize, OracleError<InMemoryGraphError>>
    where
        F: FnMut(usize, NodeIdentity, Strict<T>),
        NodeIdentity: Copy,
        Strict<T>: Copy,
    {
        self.marker.mark_all(targets);
        let mut edges = 0usize;

        for (row_idx, source) in sources.iter().enumerate() {
            let row = self
                .graph
                .get(source)
                .ok_or(OracleError::GraphError(InMemoryGraphError::MissingNode))?;
            for (&neighbour, &weight) in row.neighbours.iter() {
                if self.marker.contains(neighbour) {
                    visit(row_idx, neighbour, weight);
                    edges += 1;
                }
            }
        }

        Ok(edges)
    }

    fn visit_graph_neighbourhoods_intersecting_with_target_indices<F>(
        &mut self,
        sources: &[NodeIdentity],
        targets: &[NodeIdentity],
        mut visit: F,
    ) -> Result<usize, OracleError<InMemoryGraphError>>
    where
        F: FnMut(usize, usize, NodeIdentity, Strict<T>),
        NodeIdentity: Copy + Eq + Hash,
        Strict<T>: Copy,
    {
        self.marker.mark_all_with_ordinals(targets);
        let mut edges = 0usize;

        for (row_idx, source) in sources.iter().enumerate() {
            let row = self
                .graph
                .get(source)
                .ok_or(OracleError::GraphError(InMemoryGraphError::MissingNode))?;
            for (&neighbour, &weight) in row.neighbours.iter() {
                if let Some(target_idx) = self.marker.ordinal(neighbour) {
                    visit(row_idx, target_idx, neighbour, weight);
                    edges += 1;
                }
            }
        }

        Ok(edges)
    }

    fn coreset_neighbourhoods<'a>(
        &'a mut self,
        nodes: &[NodeIdentity],
    ) -> Result<Neighbourhoods<'a, NodeIdentity, T>, OracleError<InMemoryGraphError>> {
        self.marker.mark_all(nodes);

        self.scratch_data.clear();
        self.scratch_offsets.clear();
        self.scratch_offsets.reserve(nodes.len() + 1);
        self.scratch_offsets.push(0);

        for node in nodes {
            let row = self
                .graph
                .get(node)
                .ok_or(OracleError::CoresetError(InMemoryGraphError::MissingNode))?;
            self.scratch_data.extend(
                row.neighbours
                    .iter()
                    .filter(|(neighbour, _)| self.marker.contains(**neighbour))
                    .map(|(&neighbour, &weight)| (neighbour, weight)),
            );
            self.scratch_offsets.push(self.scratch_data.len());
        }

        Ok(Neighbourhoods::new(
            &self.scratch_data,
            &self.scratch_offsets,
        ))
    }

    fn visit_coreset_neighbourhoods_with_target_indices<F>(
        &mut self,
        nodes: &[NodeIdentity],
        mut visit: F,
    ) -> Result<usize, OracleError<InMemoryGraphError>>
    where
        F: FnMut(usize, usize, NodeIdentity, Strict<T>),
        NodeIdentity: Copy + Eq + Hash,
        Strict<T>: Copy,
    {
        self.marker.mark_all_with_ordinals(nodes);
        let mut edges = 0usize;

        for (row_idx, node) in nodes.iter().enumerate() {
            let row = self
                .graph
                .get(node)
                .ok_or(OracleError::CoresetError(InMemoryGraphError::MissingNode))?;
            for (&neighbour, &weight) in row.neighbours.iter() {
                if let Some(target_idx) = self.marker.ordinal(neighbour) {
                    visit(row_idx, target_idx, neighbour, weight);
                    edges += 1;
                }
            }
        }

        Ok(edges)
    }
}
