use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use ::raven as raven_api;

use crate::errors::RavenError;

pub(crate) fn parse_trial_weighting(value: &str) -> PyResult<raven_api::TrialWeighting> {
    match value {
        "uniform" => Ok(raven_api::TrialWeighting::Uniform),
        "inverse_score" => Ok(raven_api::TrialWeighting::InverseScore),
        "score_softmax" => Ok(raven_api::TrialWeighting::ScoreSoftmax),
        _ => Err(RavenError::new_err(
            "trial_weighting must be 'uniform', 'inverse_score', or 'score_softmax'",
        )),
    }
}

pub(crate) fn flat_pair_slice<'py>(
    array: &'py PyReadonlyArray2<'py, usize>,
) -> PyResult<&'py [usize]> {
    let shape = array.shape();
    if shape.len() != 2 || shape[1] != 2 {
        return Err(RavenError::new_err(format!(
            "pair array must have shape (n, 2), got {shape:?}"
        )));
    }
    array
        .as_slice()
        .map_err(|_| RavenError::new_err("pair array must be contiguous"))
}
