use raven_core::types::{PartitionOutput, TrialPartition};

use crate::{RavenError, Result};

/// Output from one Raven query trial.
///
/// `nodes` preserves the input order requested by the user. `labels[i]` is the
/// cluster label assigned to `nodes[i]`.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryResult {
    /// Queried node IDs in request order.
    pub nodes: Vec<usize>,
    /// Cluster labels in the same order as `nodes`.
    pub labels: Vec<usize>,
    /// Optional per-node objective scores emitted by the selected trial mode.
    pub scores: Option<Vec<f64>>,
    /// Zero-based trial index.
    pub trial_index: usize,
    /// Number of clusters produced by this trial.
    pub num_clusters: usize,
}

pub(crate) fn query_results_from_output(
    requested_nodes: &[usize],
    output: PartitionOutput<usize, f64>,
) -> Result<Vec<QueryResult>> {
    match output {
        PartitionOutput::Subset(trials) => trials
            .into_iter()
            .map(|trial| query_result_from_trial(requested_nodes, trial))
            .collect(),
        PartitionOutput::All(_, _) => Err(RavenError::UnexpectedOutput(
            "subset query returned all-node output".to_string(),
        )),
    }
}

fn query_result_from_trial(
    requested_nodes: &[usize],
    trial: TrialPartition<f64>,
) -> Result<QueryResult> {
    if trial.labels.len() != requested_nodes.len() {
        return Err(RavenError::UnexpectedOutput(format!(
            "query returned {} labels for {} requested nodes",
            trial.labels.len(),
            requested_nodes.len()
        )));
    }
    if let Some(scores) = trial.scores.as_ref() {
        if scores.len() != requested_nodes.len() {
            return Err(RavenError::UnexpectedOutput(format!(
                "query returned {} scores for {} requested nodes",
                scores.len(),
                requested_nodes.len()
            )));
        }
    }

    Ok(QueryResult {
        nodes: requested_nodes.to_vec(),
        labels: trial.labels,
        scores: trial.scores,
        trial_index: trial.trial_index,
        num_clusters: trial.num_clusters,
    })
}
