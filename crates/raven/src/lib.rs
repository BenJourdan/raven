use std::{error::Error, fmt, num::NonZeroUsize};

use raven_adapters::in_memory::{InMemoryGraphError, InMemoryIndex, InMemoryIndexError};
use raven_core::{
    alg::DynamicClustering,
    clustering::{LeidenConfig, leiden_community_detection_alg},
    types::{
        NodeIdentity, PartitionOutput, PartitionType, Strict, StrictCarrierOps, TrialObjective,
        TrialOutputMode, TrialPartition,
    },
};

const ARITY: usize = 8;

pub type Result<T> = std::result::Result<T, RavenError>;

#[derive(Debug, Clone, PartialEq)]
pub struct RavenConfig {
    pub num_clusters: usize,
    pub sigma: f64,
    pub coreset_size: usize,
    pub sampling_seeds: usize,
    pub num_trials: usize,
    pub rng_seed: Option<u64>,
    pub node_capacity: usize,
    pub expected_edges_per_node: usize,
    pub degree_rebuild_threshold: usize,
}

impl RavenConfig {
    pub fn new(num_clusters: usize) -> Self {
        let coreset_size = 8192;
        Self {
            num_clusters,
            sigma: 1000.0,
            coreset_size,
            sampling_seeds: Self::default_sampling_seeds(num_clusters, coreset_size),
            num_trials: 1,
            rng_seed: None,
            node_capacity: 1024,
            expected_edges_per_node: 16,
            degree_rebuild_threshold: 4096,
        }
    }

    pub fn default_sampling_seeds(num_clusters: usize, coreset_size: usize) -> usize {
        (num_clusters.saturating_mul(4))
            .min(coreset_size.saturating_sub(1))
            .max(2)
    }

    pub fn validate(&self) -> Result<()> {
        if self.num_clusters == 0 {
            return Err(RavenError::InvalidConfig(
                "num_clusters must be non-zero".to_string(),
            ));
        }
        if !self.sigma.is_finite() || self.sigma <= 0.0 {
            return Err(RavenError::InvalidConfig(format!(
                "sigma must be positive and finite, got {}",
                self.sigma
            )));
        }
        if self.coreset_size == 0 {
            return Err(RavenError::InvalidConfig(
                "coreset_size must be non-zero".to_string(),
            ));
        }
        if self.sampling_seeds == 0 {
            return Err(RavenError::InvalidConfig(
                "sampling_seeds must be non-zero".to_string(),
            ));
        }
        if self.sampling_seeds >= self.coreset_size {
            return Err(RavenError::InvalidConfig(format!(
                "sampling_seeds must be smaller than coreset_size, got {} >= {}",
                self.sampling_seeds, self.coreset_size
            )));
        }
        if self.num_trials == 0 {
            return Err(RavenError::InvalidConfig(
                "num_trials must be non-zero".to_string(),
            ));
        }
        if self.node_capacity == 0 {
            return Err(RavenError::InvalidConfig(
                "node_capacity must be non-zero".to_string(),
            ));
        }
        if self.expected_edges_per_node == 0 {
            return Err(RavenError::InvalidConfig(
                "expected_edges_per_node must be non-zero".to_string(),
            ));
        }
        if self.degree_rebuild_threshold == 0 {
            return Err(RavenError::InvalidConfig(
                "degree_rebuild_threshold must be non-zero".to_string(),
            ));
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeUpdate {
    Set { u: usize, v: usize, weight: f64 },
    Delete { u: usize, v: usize },
}

impl EdgeUpdate {
    pub fn set(u: usize, v: usize, weight: f64) -> Self {
        Self::Set { u, v, weight }
    }

    pub fn delete(u: usize, v: usize) -> Self {
        Self::Delete { u, v }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EdgeUpdateStats {
    pub total: usize,
    pub set: usize,
    pub deleted: usize,
    pub missing_deletes: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QueryResult {
    pub nodes: Vec<usize>,
    pub labels: Vec<usize>,
    pub scores: Option<Vec<f64>>,
    pub trial_index: usize,
    pub num_clusters: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RavenError {
    InvalidConfig(String),
    InvalidWeight(String),
    Index(String),
    UnexpectedOutput(String),
}

impl fmt::Display for RavenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid Raven config: {message}"),
            Self::InvalidWeight(message) => write!(f, "invalid edge weight: {message}"),
            Self::Index(message) => write!(f, "Raven index error: {message}"),
            Self::UnexpectedOutput(message) => write!(f, "unexpected Raven output: {message}"),
        }
    }
}

impl Error for RavenError {}

impl From<InMemoryIndexError> for RavenError {
    fn from(value: InMemoryIndexError) -> Self {
        Self::Index(value.to_string())
    }
}

pub struct Raven {
    config: RavenConfig,
    inner: InMemoryIndex<ARITY, usize, f64>,
}

impl Raven {
    pub fn new(config: RavenConfig) -> Result<Self> {
        let inner = build_index(&config)?;
        Ok(Self { config, inner })
    }

    pub fn update_edge(&mut self, u: usize, v: usize, weight: f64) -> Result<()> {
        let weight = strict_weight(weight)?;
        self.inner.update_edge(u, v, Some(weight))?;
        Ok(())
    }

    pub fn delete_edge(&mut self, u: usize, v: usize) -> Result<bool> {
        match self.inner.update_edge(u, v, None) {
            Ok(()) => Ok(true),
            Err(InMemoryIndexError::Graph(InMemoryGraphError::MissingEdge)) => Ok(false),
            Err(err) => Err(err.into()),
        }
    }

    pub fn update_edges<I>(&mut self, updates: I) -> Result<EdgeUpdateStats>
    where
        I: IntoIterator<Item = EdgeUpdate>,
    {
        let mut stats = EdgeUpdateStats::default();
        for update in updates {
            stats.total += 1;
            match update {
                EdgeUpdate::Set { u, v, weight } => {
                    self.update_edge(u, v, weight)?;
                    stats.set += 1;
                }
                EdgeUpdate::Delete { u, v } => {
                    if self.delete_edge(u, v)? {
                        stats.deleted += 1;
                    } else {
                        stats.missing_deletes += 1;
                    }
                }
            }
        }
        Ok(stats)
    }

    pub fn flush(&mut self) -> Result<()> {
        self.inner.apply_pending_node_ops()?;
        Ok(())
    }

    pub fn query(&mut self, nodes: &[usize]) -> Result<QueryResult> {
        let output = self.inner.query(
            PartitionType::Subset(nodes),
            TrialOutputMode::Winner(TrialObjective::KernelDistance),
        )?;
        let mut results = query_results_from_output(nodes, output)?;
        results
            .pop()
            .ok_or_else(|| RavenError::UnexpectedOutput("query returned no trials".to_string()))
    }

    pub fn query_all_trials(&mut self, nodes: &[usize]) -> Result<Vec<QueryResult>> {
        let output = self
            .inner
            .query(PartitionType::Subset(nodes), TrialOutputMode::AllTrials)?;
        query_results_from_output(nodes, output)
    }

    pub fn contains_node(&self, node: usize) -> bool {
        self.inner.contains_node(&node)
    }

    pub fn live_node_count(&self) -> usize {
        self.inner.live_node_count()
    }

    pub fn live_nodes(&self) -> Vec<usize> {
        let mut nodes = self.inner.live_nodes();
        nodes.sort_unstable();
        nodes
    }

    pub fn clear(&mut self) -> Result<()> {
        self.inner = build_index(&self.config)?;
        Ok(())
    }

    pub fn config(&self) -> &RavenConfig {
        &self.config
    }
}

fn build_index(config: &RavenConfig) -> Result<InMemoryIndex<ARITY, usize, f64>> {
    config.validate()?;

    let mut leiden_config = LeidenConfig {
        seed: config.rng_seed,
        ..LeidenConfig::default()
    };
    leiden_config.seed = config.rng_seed;

    let mut clustering = DynamicClustering::<ARITY, NodeIdentity, f64>::new(
        leiden_community_detection_alg(leiden_config),
    )
    .with_sigma(strict_weight(config.sigma)?)
    .with_num_trials(config.num_trials)
    .with_coreset_size(config.coreset_size)
    .with_sampling_seeds(config.sampling_seeds)
    .with_num_clusters(config.num_clusters)
    .with_prop_name("w");

    if let Some(seed) = config.rng_seed {
        clustering = clustering.with_rng_seed(seed);
    } else {
        clustering = clustering.with_random_rng();
    }

    Ok(InMemoryIndex::with_capacity(
        clustering,
        non_zero(config.node_capacity, "node_capacity")?,
        non_zero(config.expected_edges_per_node, "expected_edges_per_node")?,
        non_zero(config.degree_rebuild_threshold, "degree_rebuild_threshold")?,
    ))
}

fn query_results_from_output(
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

fn strict_weight(weight: f64) -> Result<Strict<f64>> {
    Strict::<f64>::from_positive_scalar(weight).map_err(|err| {
        RavenError::InvalidWeight(format!(
            "expected a positive finite weight, got {weight}: {err}"
        ))
    })
}

fn non_zero(value: usize, field: &'static str) -> Result<NonZeroUsize> {
    NonZeroUsize::new(value)
        .ok_or_else(|| RavenError::InvalidConfig(format!("{field} must be non-zero")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> RavenConfig {
        RavenConfig {
            coreset_size: 3,
            sampling_seeds: 2,
            node_capacity: 16,
            expected_edges_per_node: 4,
            rng_seed: Some(42),
            ..RavenConfig::new(2)
        }
    }

    #[test]
    fn config_defaults_and_validation() {
        let config = RavenConfig::new(4);
        assert_eq!(config.num_clusters, 4);
        assert_eq!(config.sigma, 1000.0);
        assert_eq!(config.sampling_seeds, 16);
        assert_eq!(config.coreset_size, 8192);
        assert!(config.validate().is_ok());

        let mut invalid = config.clone();
        invalid.sigma = 0.0;
        assert!(matches!(
            invalid.validate(),
            Err(RavenError::InvalidConfig(_))
        ));

        let mut invalid = config;
        invalid.sampling_seeds = invalid.coreset_size;
        assert!(matches!(
            invalid.validate(),
            Err(RavenError::InvalidConfig(_))
        ));
    }

    #[test]
    fn update_delete_live_nodes_and_clear() {
        let mut raven = Raven::new(small_config()).unwrap();

        assert!(!raven.contains_node(1));
        raven.update_edge(1, 2, 1.0).unwrap();
        raven.update_edge(2, 3, 2.0).unwrap();
        assert!(raven.contains_node(1));
        assert_eq!(raven.live_node_count(), 3);
        assert_eq!(raven.live_nodes(), vec![1, 2, 3]);

        assert!(raven.delete_edge(1, 2).unwrap());
        assert!(!raven.delete_edge(1, 2).unwrap());
        raven.flush().unwrap();
        assert_eq!(raven.live_nodes(), vec![2, 3]);

        raven.clear().unwrap();
        assert_eq!(raven.live_node_count(), 0);
        assert!(raven.live_nodes().is_empty());
        raven.update_edge(10, 11, 1.0).unwrap();
        assert_eq!(raven.live_nodes(), vec![10, 11]);
    }

    #[test]
    fn update_edges_reports_stats() {
        let mut raven = Raven::new(small_config()).unwrap();
        let stats = raven
            .update_edges([
                EdgeUpdate::set(1, 2, 1.0),
                EdgeUpdate::set(2, 3, 1.0),
                EdgeUpdate::delete(1, 2),
                EdgeUpdate::delete(1, 2),
            ])
            .unwrap();

        assert_eq!(
            stats,
            EdgeUpdateStats {
                total: 4,
                set: 2,
                deleted: 1,
                missing_deletes: 1,
            }
        );
    }

    #[test]
    fn query_auto_flushes_and_preserves_order() {
        let mut raven = Raven::new(small_config()).unwrap();
        raven.update_edge(1, 2, 1.0).unwrap();
        raven.update_edge(2, 3, 1.0).unwrap();
        raven.update_edge(3, 4, 1.0).unwrap();

        let result = raven.query(&[3, 1, 2]).unwrap();
        assert_eq!(result.nodes, vec![3, 1, 2]);
        assert_eq!(result.labels.len(), 3);
        assert_eq!(result.trial_index, 0);
        assert!(result.num_clusters > 0);

        let all_trials = raven.query_all_trials(&[4, 2]).unwrap();
        assert_eq!(all_trials.len(), 1);
        assert_eq!(all_trials[0].nodes, vec![4, 2]);
        assert_eq!(all_trials[0].labels.len(), 2);
    }

    #[test]
    fn invalid_weight_is_rejected() {
        let mut raven = Raven::new(small_config()).unwrap();
        assert!(matches!(
            raven.update_edge(1, 2, f64::NAN),
            Err(RavenError::InvalidWeight(_))
        ));
    }
}
