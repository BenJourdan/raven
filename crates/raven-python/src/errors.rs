use ::raven as raven_api;
use pyo3::{create_exception, exceptions::PyException, prelude::*};

create_exception!(_raven, RavenError, PyException);

pub(crate) fn to_py_err(err: raven_api::RavenError) -> PyErr {
    RavenError::new_err(err.to_string())
}
