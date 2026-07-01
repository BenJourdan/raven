mod classes;
mod convert;
mod errors;

use pyo3::prelude::*;

use crate::{
    classes::{PyConsensusResult, PyEdgeUpdateStats, PyQueryResult, PyRaven},
    errors::RavenError,
};

#[pymodule]
fn _raven(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("RavenError", py.get_type::<RavenError>())?;
    m.add_class::<PyRaven>()?;
    m.add_class::<PyQueryResult>()?;
    m.add_class::<PyConsensusResult>()?;
    m.add_class::<PyEdgeUpdateStats>()?;
    Ok(())
}
