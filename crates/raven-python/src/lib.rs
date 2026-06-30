use ::raven as raven_api;
use pyo3::{create_exception, exceptions::PyException, prelude::*};
use raven_api::{EdgeUpdate, RavenConfig, RavenError as ApiRavenError};

create_exception!(_raven, RavenError, PyException);

#[pyclass(name = "QueryResult", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyQueryResult {
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

#[pyclass(name = "EdgeUpdateStats", frozen, skip_from_py_object)]
#[derive(Clone, Copy)]
struct PyEdgeUpdateStats {
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
struct PyRaven {
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

#[pymodule]
fn _raven(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("RavenError", py.get_type::<RavenError>())?;
    m.add_class::<PyRaven>()?;
    m.add_class::<PyQueryResult>()?;
    m.add_class::<PyEdgeUpdateStats>()?;
    Ok(())
}

fn to_py_err(err: ApiRavenError) -> PyErr {
    RavenError::new_err(err.to_string())
}
