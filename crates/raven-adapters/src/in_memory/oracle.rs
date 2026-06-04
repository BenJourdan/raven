use std::{collections::HashSet, hash::Hash};

use raven_core::{
    GraphOracle,
    error::OracleError,
    types::{FloatScalar, Strict, StrictCarrierOps},
};

use super::graph::{AdjacencyMap, InMemoryGraphError, InMemoryUndirectedGraph};

/// Query-only oracle handle borrowed from an [`InMemoryUndirectedGraph`].
///
/// Each handle owns its scratch rows. Multiple handles may therefore be used
/// by parallel trials while they all observe the same immutably borrowed graph.
/// Returned neighbourhood slices borrow this scratch storage and remain valid
/// until the next oracle call on the same handle.
#[derive(Debug)]
pub struct InMemoryOracle<'a, V, T> {
    graph: &'a AdjacencyMap<V, T>,
    scratch_rows: Vec<Vec<(V, Strict<T>)>>,
}

impl<'a, V, T> InMemoryOracle<'a, V, T> {
    fn new(graph: &'a AdjacencyMap<V, T>) -> Self {
        Self {
            graph,
            scratch_rows: Vec::new(),
        }
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

impl<V, T> GraphOracle<V, T, InMemoryGraphError> for InMemoryOracle<'_, V, T>
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
                .graph
                .get(node)
                .ok_or(OracleError::GraphError(InMemoryGraphError::MissingNode))?
                .iter()
                .map(|(&neighbour, &weight)| (neighbour, weight))
                .collect();
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
            let row = self
                .graph
                .get(node)
                .ok_or(OracleError::CoresetError(InMemoryGraphError::MissingNode))?
                .iter()
                .filter(|(neighbour, _)| coreset_members.contains(neighbour))
                .map(|(&neighbour, &weight)| (neighbour, weight))
                .collect();
            self.scratch_rows.push(row);
        }

        Ok(self.scratch_rows.iter().map(Vec::as_slice).collect())
    }
}
