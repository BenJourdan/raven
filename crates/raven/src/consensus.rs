use std::collections::HashMap;

use crate::{QueryResult, RavenError, Result};

/// Strategy used to combine independent query trials into pair scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrialWeighting {
    /// Give every trial equal mass.
    Uniform,
    /// Weight trials by inverse total objective score.
    InverseScore,
    /// Weight trials with a softmax over negated total objective scores.
    ScoreSoftmax,
}

/// Reusable consensus view over multiple query trials.
///
/// A `ConsensusResult` keeps each trial partition and exposes lazy pair
/// scoring. This avoids materialising a full pairwise similarity matrix unless
/// the caller explicitly asks for one.
///
/// ```no_run
/// # use raven::{Raven, RavenConfig, TrialWeighting};
/// # let mut config = RavenConfig::new(2);
/// # config.coreset_size = 3;
/// # config.sampling_seeds = 2;
/// # config.num_trials = 3;
/// # let mut index = Raven::new(config)?;
/// # index.update_edge(1, 2, 1.0)?;
/// # index.update_edge(2, 3, 1.0)?;
/// let consensus = index.query_consensus(
///     &[1, 2, 3],
///     TrialWeighting::ScoreSoftmax,
///     None,
/// )?;
/// let same_cluster_probability = consensus.score_pair(1, 2)?;
/// # Ok::<(), raven::RavenError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ConsensusResult {
    /// Queried node IDs in request order.
    pub nodes: Vec<usize>,
    /// Per-trial labels. `labels[t][i]` is the label for `nodes[i]` in trial `t`.
    pub labels: Vec<Vec<usize>>,
    /// Normalized trial weights used by pair scoring.
    pub trial_weights: Vec<f64>,
    /// Total trial objective scores.
    pub trial_scores: Vec<f64>,
    /// Original trial indices.
    pub trial_indices: Vec<usize>,
    /// Per-trial cluster counts.
    pub num_clusters: Vec<usize>,
    node_positions: HashMap<usize, usize>,
    dense_node_positions: Option<Vec<usize>>,
}

impl ConsensusResult {
    /// Builds a reusable consensus object from trial results.
    pub fn from_trials(
        nodes: &[usize],
        trials: Vec<QueryResult>,
        trial_weighting: TrialWeighting,
        temperature: Option<f64>,
    ) -> Result<Self> {
        if trials.is_empty() {
            return Err(RavenError::InvalidInput(
                "consensus requires at least one trial".to_string(),
            ));
        }

        let (node_positions, dense_node_positions) = node_positions(nodes)?;
        let trial_scores = trials.iter().map(trial_score).collect::<Result<Vec<_>>>()?;
        let trial_weights = trial_weights(&trial_scores, trial_weighting, temperature)?;
        let mut labels = Vec::with_capacity(trials.len());
        let mut trial_indices = Vec::with_capacity(trials.len());
        let mut num_clusters = Vec::with_capacity(trials.len());

        for trial in trials {
            if trial.labels.len() != nodes.len() {
                return Err(RavenError::UnexpectedOutput(format!(
                    "trial {} has {} labels for {} consensus nodes",
                    trial.trial_index,
                    trial.labels.len(),
                    nodes.len()
                )));
            }
            labels.push(trial.labels);
            trial_indices.push(trial.trial_index);
            num_clusters.push(trial.num_clusters);
        }

        Ok(Self {
            nodes: nodes.to_vec(),
            labels,
            trial_weights,
            trial_scores,
            trial_indices,
            num_clusters,
            node_positions,
            dense_node_positions,
        })
    }

    /// Number of trials included in this consensus result.
    pub fn num_trials(&self) -> usize {
        self.labels.len()
    }

    /// Number of queried nodes included in this consensus result.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Scores a single pair as weighted same-cluster trial mass.
    pub fn score_pair(&self, u: usize, v: usize) -> Result<f64> {
        let scores = self.score_pairs(&[(u, v)])?;
        scores
            .first()
            .copied()
            .ok_or_else(|| RavenError::UnexpectedOutput("score_pair returned no score".to_string()))
    }

    /// Scores pairs as weighted same-cluster trial mass.
    pub fn score_pairs(&self, pairs: &[(usize, usize)]) -> Result<Vec<f64>> {
        let mut scores = Vec::with_capacity(pairs.len());
        for &(u, v) in pairs {
            scores.push(self.score_pair_by_nodes(u, v)?);
        }
        Ok(scores)
    }

    /// Scores a flat `[u0, v0, u1, v1, ...]` pair buffer.
    pub fn score_flat_pairs(&self, flat_pairs: &[usize]) -> Result<Vec<f64>> {
        if flat_pairs.len() % 2 != 0 {
            return Err(RavenError::InvalidInput(format!(
                "flat pair buffer must contain an even number of node ids, got {}",
                flat_pairs.len()
            )));
        }

        let mut scores = Vec::with_capacity(flat_pairs.len() / 2);
        for pair in flat_pairs.chunks_exact(2) {
            scores.push(self.score_pair_by_nodes(pair[0], pair[1])?);
        }
        Ok(scores)
    }

    /// Materialises a dense score matrix for selected nodes or all consensus nodes.
    pub fn score_matrix(&self, nodes: Option<&[usize]>) -> Result<Vec<Vec<f64>>> {
        let selected = nodes.unwrap_or(&self.nodes);
        let positions = selected
            .iter()
            .map(|node| self.position(*node))
            .collect::<Result<Vec<_>>>()?;

        let mut matrix = vec![vec![0.0; selected.len()]; selected.len()];
        for (i, &left) in positions.iter().enumerate() {
            for (j, &right) in positions.iter().enumerate() {
                let mut score = 0.0;
                for (trial_labels, weight) in self.labels.iter().zip(self.trial_weights.iter()) {
                    if trial_labels[left] == trial_labels[right] {
                        score += weight;
                    }
                }
                matrix[i][j] = score;
            }
        }
        Ok(matrix)
    }

    /// Returns only pairs whose consensus score is at least `threshold`.
    pub fn threshold_pairs(
        &self,
        pairs: &[(usize, usize)],
        threshold: f64,
    ) -> Result<Vec<(usize, usize, f64)>> {
        let scores = self.score_pairs(pairs)?;
        Ok(pairs
            .iter()
            .copied()
            .zip(scores)
            .filter_map(|((u, v), score)| (score >= threshold).then_some((u, v, score)))
            .collect())
    }

    /// Builds connected components from pairs whose score crosses `threshold`.
    pub fn connected_components(
        &self,
        pairs: &[(usize, usize)],
        threshold: f64,
        include_singletons: bool,
    ) -> Result<Vec<Vec<usize>>> {
        let mut parent = (0..self.nodes.len()).collect::<Vec<_>>();
        let mut active = vec![false; self.nodes.len()];

        for &(u, v, _) in &self.threshold_pairs(pairs, threshold)? {
            let left = self.position(u)?;
            let right = self.position(v)?;
            active[left] = true;
            active[right] = true;
            union(&mut parent, left, right);
        }

        let mut components_by_root: HashMap<usize, Vec<usize>> = HashMap::new();
        for idx in 0..self.nodes.len() {
            if !include_singletons && !active[idx] {
                continue;
            }
            let root = find(&mut parent, idx);
            components_by_root
                .entry(root)
                .or_default()
                .push(self.nodes[idx]);
        }

        let mut components = components_by_root
            .into_values()
            .map(|mut component| {
                component.sort_unstable();
                component
            })
            .collect::<Vec<_>>();
        components
            .sort_unstable_by_key(|component| component.first().copied().unwrap_or(usize::MAX));
        Ok(components)
    }

    fn score_pair_by_nodes(&self, u: usize, v: usize) -> Result<f64> {
        let left = self.position(u)?;
        let right = self.position(v)?;
        let mut score = 0.0;
        for (trial_labels, weight) in self.labels.iter().zip(self.trial_weights.iter()) {
            if trial_labels[left] == trial_labels[right] {
                score += weight;
            }
        }
        Ok(score)
    }

    fn position(&self, node: usize) -> Result<usize> {
        if let Some(dense_positions) = self.dense_node_positions.as_ref() {
            if let Some(position) = dense_positions.get(node).copied() {
                if position != usize::MAX {
                    return Ok(position);
                }
            }
        }
        self.node_positions.get(&node).copied().ok_or_else(|| {
            RavenError::InvalidInput(format!("node {node} was not part of this consensus query"))
        })
    }
}

pub(crate) fn validate_unique_nodes(nodes: &[usize]) -> Result<()> {
    let _ = node_positions(nodes)?;
    Ok(())
}

fn node_positions(nodes: &[usize]) -> Result<(HashMap<usize, usize>, Option<Vec<usize>>)> {
    let mut positions = HashMap::with_capacity(nodes.len());
    for (idx, node) in nodes.iter().copied().enumerate() {
        if positions.insert(node, idx).is_some() {
            return Err(RavenError::InvalidInput(format!(
                "consensus nodes must be unique, found duplicate node {node}"
            )));
        }
    }

    let dense_positions = dense_node_positions(nodes);
    Ok((positions, dense_positions))
}

fn dense_node_positions(nodes: &[usize]) -> Option<Vec<usize>> {
    let max_node = nodes.iter().copied().max()?;
    let len = max_node.checked_add(1)?;
    let max_reasonable_len = nodes.len().saturating_mul(64).max(1024);
    if len > max_reasonable_len || len > 10_000_000 {
        return None;
    }

    let mut positions = vec![usize::MAX; len];
    for (idx, node) in nodes.iter().copied().enumerate() {
        positions[node] = idx;
    }
    Some(positions)
}

fn find(parent: &mut [usize], node: usize) -> usize {
    if parent[node] != node {
        parent[node] = find(parent, parent[node]);
    }
    parent[node]
}

fn union(parent: &mut [usize], left: usize, right: usize) {
    let left_root = find(parent, left);
    let right_root = find(parent, right);
    if left_root != right_root {
        parent[right_root] = left_root;
    }
}

fn trial_score(trial: &QueryResult) -> Result<f64> {
    match trial.scores.as_ref() {
        Some(scores) => Ok(scores.iter().sum()),
        None => Err(RavenError::UnexpectedOutput(format!(
            "trial {} did not include scores needed for consensus weighting",
            trial.trial_index
        ))),
    }
}

fn trial_weights(
    scores: &[f64],
    trial_weighting: TrialWeighting,
    temperature: Option<f64>,
) -> Result<Vec<f64>> {
    if scores.is_empty() {
        return Ok(Vec::new());
    }

    match trial_weighting {
        TrialWeighting::Uniform => Ok(vec![1.0 / scores.len() as f64; scores.len()]),
        TrialWeighting::InverseScore => {
            if scores
                .iter()
                .any(|score| !score.is_finite() || *score <= 0.0)
            {
                return Err(RavenError::InvalidInput(
                    "inverse_score requires positive finite trial scores".to_string(),
                ));
            }
            let mut raw = scores
                .iter()
                .map(|score| 1.0 / (*score).max(1e-12))
                .collect::<Vec<_>>();
            normalize_weights(&mut raw)?;
            Ok(raw)
        }
        TrialWeighting::ScoreSoftmax => {
            if scores.iter().any(|score| !score.is_finite()) {
                return Err(RavenError::InvalidInput(
                    "score_softmax requires finite trial scores".to_string(),
                ));
            }
            let temp = match temperature {
                Some(temp) if temp.is_finite() && temp > 0.0 => temp,
                Some(temp) => {
                    return Err(RavenError::InvalidInput(format!(
                        "temperature must be positive and finite, got {temp}"
                    )));
                }
                None => auto_temperature(scores),
            };
            let min_score = scores.iter().copied().fold(f64::INFINITY, f64::min);
            let mut scaled = scores
                .iter()
                .map(|score| -(*score - min_score) / temp)
                .collect::<Vec<_>>();
            let max_scaled = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for value in &mut scaled {
                *value = (*value - max_scaled).exp();
            }
            normalize_weights(&mut scaled)?;
            Ok(scaled)
        }
    }
}

fn normalize_weights(weights: &mut [f64]) -> Result<()> {
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(RavenError::InvalidInput(
            "trial weights have non-positive total mass".to_string(),
        ));
    }
    for weight in weights {
        *weight /= total;
    }
    Ok(())
}

fn auto_temperature(scores: &[f64]) -> f64 {
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let variance = scores
        .iter()
        .map(|score| {
            let diff = *score - mean;
            diff * diff
        })
        .sum::<f64>()
        / scores.len() as f64;
    variance.sqrt().max(1e-12)
}
