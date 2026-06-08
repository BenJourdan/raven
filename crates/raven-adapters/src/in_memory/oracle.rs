use std::hash::Hash;

use raven_core::{
    GraphOracle,
    error::OracleError,
    types::{FloatScalar, Neighbourhoods, Strict, StrictCarrierOps},
};
use rustc_hash::FxHashSet;

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

impl<'a, V, T> InMemoryOracle<'a, V, T> {
    fn new(graph: &'a AdjacencyMap<V, T>) -> Self {
        Self {
            graph,
            scratch_data: Vec::new(),
            scratch_offsets: Vec::new(),
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
}
