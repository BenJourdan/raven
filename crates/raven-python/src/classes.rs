use ::raven as raven_api;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::{prelude::*, types::PyAnyMethods};
use raven_api::{EdgeUpdate, RavenConfig, RavenError as ApiRavenError};

use crate::{
    convert::{flat_pair_slice, parse_trial_weighting},
    errors::{RavenError, to_py_err},
};

#[pyclass(name = "QueryResult", frozen, skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyQueryResult {
    #[pyo3(get)]
    nodes: Vec<usize>,
    #[pyo3(get)]
    labels: Vec<usize>,
    #[pyo3(get)]
    scores: Option<Vec<f64>>,
    #[pyo3(get)]
    trial_index: usize,
    #[pyo3(get)]
    num_clusters: usize,
}

impl From<raven_api::QueryResult> for PyQueryResult {
    fn from(value: raven_api::QueryResult) -> Self {
        Self {
            nodes: value.nodes,
            labels: value.labels,
            scores: value.scores,
            trial_index: value.trial_index,
            num_clusters: value.num_clusters,
        }
    }
}

#[pyclass(name = "ConsensusResult", frozen, skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyConsensusResult {
    inner: raven_api::ConsensusResult,
}

impl From<raven_api::ConsensusResult> for PyConsensusResult {
    fn from(value: raven_api::ConsensusResult) -> Self {
        Self { inner: value }
    }
}

#[pymethods]
impl PyConsensusResult {
    #[getter]
    fn nodes(&self) -> Vec<usize> {
        self.inner.nodes.clone()
    }

    #[getter]
    fn labels(&self) -> Vec<Vec<usize>> {
        self.inner.labels.clone()
    }

    #[getter]
    fn trial_weights(&self) -> Vec<f64> {
        self.inner.trial_weights.clone()
    }

    #[getter]
    fn trial_scores(&self) -> Vec<f64> {
        self.inner.trial_scores.clone()
    }

    #[getter]
    fn trial_indices(&self) -> Vec<usize> {
        self.inner.trial_indices.clone()
    }

    #[getter]
    fn num_clusters(&self) -> Vec<usize> {
        self.inner.num_clusters.clone()
    }

    #[getter]
    fn num_trials(&self) -> usize {
        self.inner.num_trials()
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    fn score_pair(&self, u: usize, v: usize) -> PyResult<f64> {
        self.inner.score_pair(u, v).map_err(to_py_err)
    }

    fn score_pairs<'py>(
        &self,
        py: Python<'py>,
        pairs: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let scores = match pairs.extract::<PyReadonlyArray2<'py, usize>>() {
            Ok(array) => self
                .inner
                .score_flat_pairs(flat_pair_slice(&array)?)
                .map_err(to_py_err)?,
            Err(_) => {
                let pairs = pairs.extract::<Vec<(usize, usize)>>().map_err(|_| {
                    RavenError::new_err("pairs must be an Nx2 uint array or a list of (u, v) pairs")
                })?;
                self.inner.score_pairs(&pairs).map_err(to_py_err)?
            }
        };
        Ok(scores.into_pyarray(py))
    }

    #[pyo3(signature = (nodes = None))]
    fn score_matrix(&self, nodes: Option<Vec<usize>>) -> PyResult<Vec<Vec<f64>>> {
        self.inner.score_matrix(nodes.as_deref()).map_err(to_py_err)
    }

    #[pyo3(signature = (pairs, *, threshold = 0.8))]
    fn threshold_pairs(
        &self,
        pairs: Vec<(usize, usize)>,
        threshold: f64,
    ) -> PyResult<Vec<(usize, usize, f64)>> {
        self.inner
            .threshold_pairs(&pairs, threshold)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (pairs, *, threshold = 0.8, include_singletons = true))]
    fn connected_components(
        &self,
        pairs: Vec<(usize, usize)>,
        threshold: f64,
        include_singletons: bool,
    ) -> PyResult<Vec<Vec<usize>>> {
        self.inner
            .connected_components(&pairs, threshold, include_singletons)
            .map_err(to_py_err)
    }
}

#[pyclass(name = "EdgeUpdateStats", frozen, skip_from_py_object)]
#[derive(Clone, Copy)]
pub(crate) struct PyEdgeUpdateStats {
    #[pyo3(get)]
    total: usize,
    #[pyo3(get)]
    set: usize,
    #[pyo3(get)]
    deleted: usize,
    #[pyo3(get)]
    missing_deletes: usize,
}

impl From<raven_api::EdgeUpdateStats> for PyEdgeUpdateStats {
    fn from(value: raven_api::EdgeUpdateStats) -> Self {
        Self {
            total: value.total,
            set: value.set,
            deleted: value.deleted,
            missing_deletes: value.missing_deletes,
        }
    }
}

#[pyclass(name = "Raven")]
pub(crate) struct PyRaven {
    inner: raven_api::Raven,
}

#[pymethods]
impl PyRaven {
    #[new]
    #[pyo3(signature = (
        num_clusters,
        *,
        sigma = 1000.0,
        coreset_size = 8192,
        sampling_seeds = None,
        num_trials = 1,
        rng_seed = None,
        node_capacity = 1024,
        expected_edges_per_node = 16,
        degree_rebuild_threshold = 4096
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_clusters: usize,
        sigma: f64,
        coreset_size: usize,
        sampling_seeds: Option<usize>,
        num_trials: usize,
        rng_seed: Option<u64>,
        node_capacity: usize,
        expected_edges_per_node: usize,
        degree_rebuild_threshold: usize,
    ) -> PyResult<Self> {
        let mut config = RavenConfig::new(num_clusters);
        config.sigma = sigma;
        config.coreset_size = coreset_size;
        config.sampling_seeds = sampling_seeds
            .unwrap_or_else(|| RavenConfig::default_sampling_seeds(num_clusters, coreset_size));
        config.num_trials = num_trials;
        config.rng_seed = rng_seed;
        config.node_capacity = node_capacity;
        config.expected_edges_per_node = expected_edges_per_node;
        config.degree_rebuild_threshold = degree_rebuild_threshold;

        Ok(Self {
            inner: raven_api::Raven::new(config).map_err(to_py_err)?,
        })
    }

    fn update_edge(&mut self, u: usize, v: usize, weight: f64) -> PyResult<()> {
        self.inner.update_edge(u, v, weight).map_err(to_py_err)
    }

    fn delete_edge(&mut self, u: usize, v: usize) -> PyResult<bool> {
        self.inner.delete_edge(u, v).map_err(to_py_err)
    }

    fn update_edges(
        &mut self,
        py: Python<'_>,
        updates: Vec<(usize, usize, Option<f64>)>,
    ) -> PyResult<PyEdgeUpdateStats> {
        let updates = updates
            .into_iter()
            .map(|(u, v, weight)| match weight {
                Some(weight) => EdgeUpdate::set(u, v, weight),
                None => EdgeUpdate::delete(u, v),
            })
            .collect::<Vec<_>>();

        py.detach(|| self.inner.update_edges(updates))
            .map(PyEdgeUpdateStats::from)
            .map_err(to_py_err)
    }

    fn flush(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.flush()).map_err(to_py_err)
    }

    fn query(&mut self, py: Python<'_>, nodes: Vec<usize>) -> PyResult<PyQueryResult> {
        py.detach(|| self.inner.query(&nodes))
            .map(PyQueryResult::from)
            .map_err(to_py_err)
    }

    fn query_all_trials(
        &mut self,
        py: Python<'_>,
        nodes: Vec<usize>,
    ) -> PyResult<Vec<PyQueryResult>> {
        py.detach(|| self.inner.query_all_trials(&nodes))
            .map(|results| results.into_iter().map(PyQueryResult::from).collect())
            .map_err(to_py_err)
    }

    #[pyo3(signature = (
        nodes,
        *,
        trial_weighting = "score_softmax",
        temperature = None
    ))]
    fn query_consensus(
        &mut self,
        py: Python<'_>,
        nodes: Vec<usize>,
        trial_weighting: &str,
        temperature: Option<f64>,
    ) -> PyResult<PyConsensusResult> {
        let trial_weighting = parse_trial_weighting(trial_weighting)?;
        py.detach(|| {
            self.inner
                .query_consensus(&nodes, trial_weighting, temperature)
        })
        .map(PyConsensusResult::from)
        .map_err(to_py_err)
    }

    #[pyo3(signature = (
        u,
        v,
        *,
        trial_weighting = "score_softmax",
        temperature = None
    ))]
    fn score_pair(
        &mut self,
        py: Python<'_>,
        u: usize,
        v: usize,
        trial_weighting: &str,
        temperature: Option<f64>,
    ) -> PyResult<f64> {
        let trial_weighting = parse_trial_weighting(trial_weighting)?;
        py.detach(|| {
            self.inner
                .score_pairs(&[(u, v)], trial_weighting, temperature)
        })
        .and_then(|scores| {
            scores.first().copied().ok_or_else(|| {
                ApiRavenError::UnexpectedOutput("score_pair returned no score".to_string())
            })
        })
        .map_err(to_py_err)
    }

    #[pyo3(signature = (
        pairs,
        *,
        trial_weighting = "score_softmax",
        temperature = None
    ))]
    fn score_pairs<'py>(
        &mut self,
        py: Python<'py>,
        pairs: &Bound<'py, PyAny>,
        trial_weighting: &str,
        temperature: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let trial_weighting = parse_trial_weighting(trial_weighting)?;
        let scores = match pairs.extract::<PyReadonlyArray2<'py, usize>>() {
            Ok(array) => {
                let flat_pairs = flat_pair_slice(&array)?.to_vec();
                py.detach(|| {
                    self.inner
                        .score_flat_pairs(&flat_pairs, trial_weighting, temperature)
                })
                .map_err(to_py_err)?
            }
            Err(_) => {
                let pairs = pairs.extract::<Vec<(usize, usize)>>().map_err(|_| {
                    RavenError::new_err("pairs must be an Nx2 uint array or a list of (u, v) pairs")
                })?;
                py.detach(|| self.inner.score_pairs(&pairs, trial_weighting, temperature))
                    .map_err(to_py_err)?
            }
        };
        Ok(scores.into_pyarray(py))
    }

    fn contains_node(&self, node: usize) -> bool {
        self.inner.contains_node(node)
    }

    fn live_node_count(&self) -> usize {
        self.inner.live_node_count()
    }

    fn live_nodes(&self) -> Vec<usize> {
        self.inner.live_nodes()
    }

    fn clear(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.clear()).map_err(to_py_err)
    }
}
