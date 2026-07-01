use std::{collections::HashMap, num::NonZeroUsize};

use raven_adapters::in_memory::{InMemoryGraphError, InMemoryIndex, InMemoryIndexError};
use raven_core::{
    alg::DynamicClustering,
    clustering::{LeidenConfig, leiden_community_detection_alg},
    types::{
        NodeIdentity, PartitionType, Strict, StrictCarrierOps, TrialObjective, TrialOutputMode,
    },
};

use crate::{
    ARITY, ConsensusResult, QueryResult, RavenConfig, RavenError, Result, TrialWeighting,
    consensus::validate_unique_nodes, query::query_results_from_output,
};

/// Edge update accepted by [`Raven::update_edges`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeUpdate {
    /// Insert or update an undirected weighted edge.
    Set { u: usize, v: usize, weight: f64 },
    /// Delete an undirected edge if present.
    Delete { u: usize, v: usize },
}

impl EdgeUpdate {
    /// Creates an insert/update operation.
    pub fn set(u: usize, v: usize, weight: f64) -> Self {
        Self::Set { u, v, weight }
    }

    /// Creates a delete operation.
    pub fn delete(u: usize, v: usize) -> Self {
        Self::Delete { u, v }
    }
}

/// Summary returned by [`Raven::update_edges`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EdgeUpdateStats {
    /// Number of operations supplied.
    pub total: usize,
    /// Number of insert/update operations applied.
    pub set: usize,
    /// Number of delete operations that removed an existing edge.
    pub deleted: usize,
    /// Number of delete operations whose edge was absent.
    pub missing_deletes: usize,
}

/// Public in-memory Raven index.
///
/// `Raven` is intentionally stateful: edge updates are ingested into an
/// in-memory graph, pending node-degree updates are flushed into the dynamic
/// clustering data structure, and queries run against the current state.
///
/// ```no_run
/// # use raven::{Raven, RavenConfig};
/// # let mut config = RavenConfig::new(2);
/// # config.coreset_size = 3;
/// # config.sampling_seeds = 2;
/// let mut index = Raven::new(config)?;
/// index.update_edge(10, 11, 2.0)?;
/// index.update_edge(11, 12, 1.0)?;
/// let result = index.query(&[10, 11, 12])?;
/// assert_eq!(result.nodes, vec![10, 11, 12]);
/// # Ok::<(), raven::RavenError>(())
/// ```
pub struct Raven {
    config: RavenConfig,
    inner: InMemoryIndex<ARITY, usize, f64>,
}

impl Raven {
    /// Creates an empty Raven index.
    pub fn new(config: RavenConfig) -> Result<Self> {
        let inner = build_index(&config)?;
        Ok(Self { config, inner })
    }

    /// Inserts or updates an undirected weighted edge.
    pub fn update_edge(&mut self, u: usize, v: usize, weight: f64) -> Result<()> {
        let weight = strict_weight(weight)?;
        self.inner.update_edge(u, v, Some(weight))?;
        Ok(())
    }

    /// Deletes an undirected edge.
    ///
    /// Returns `false` if the edge was absent.
    pub fn delete_edge(&mut self, u: usize, v: usize) -> Result<bool> {
        match self.inner.update_edge(u, v, None) {
            Ok(()) => Ok(true),
            Err(InMemoryIndexError::Graph(InMemoryGraphError::MissingEdge)) => Ok(false),
            Err(err) => Err(err.into()),
        }
    }

    /// Applies a batch of edge updates.
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

    /// Flushes pending graph-degree changes into the clustering data structure.
    ///
    /// Queries call this automatically. Manual flushing is useful when users want
    /// explicit timing or back-pressure around update batches.
    pub fn flush(&mut self) -> Result<()> {
        self.inner.apply_pending_node_ops()?;
        Ok(())
    }

    /// Queries a node subset and returns the winning trial.
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

    /// Queries a node subset and returns every trial.
    pub fn query_all_trials(&mut self, nodes: &[usize]) -> Result<Vec<QueryResult>> {
        let output = self
            .inner
            .query(PartitionType::Subset(nodes), TrialOutputMode::AllTrials)?;
        query_results_from_output(nodes, output)
    }

    /// Queries a node subset and builds a reusable consensus object.
    pub fn query_consensus(
        &mut self,
        nodes: &[usize],
        trial_weighting: TrialWeighting,
        temperature: Option<f64>,
    ) -> Result<ConsensusResult> {
        validate_unique_nodes(nodes)?;
        let trials = self.query_all_trials(nodes)?;
        ConsensusResult::from_trials(nodes, trials, trial_weighting, temperature)
    }

    /// Scores one pair by querying only the required nodes.
    pub fn score_pair(&mut self, u: usize, v: usize) -> Result<f64> {
        let scores = self.score_pairs(&[(u, v)], TrialWeighting::ScoreSoftmax, None)?;
        scores
            .first()
            .copied()
            .ok_or_else(|| RavenError::UnexpectedOutput("score_pair returned no score".to_string()))
    }

    /// Scores pairs by querying the unique nodes present in the pair list.
    pub fn score_pairs(
        &mut self,
        pairs: &[(usize, usize)],
        trial_weighting: TrialWeighting,
        temperature: Option<f64>,
    ) -> Result<Vec<f64>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        let mut nodes = Vec::new();
        let mut positions = HashMap::new();
        for &(u, v) in pairs {
            for node in [u, v] {
                if let std::collections::hash_map::Entry::Vacant(entry) = positions.entry(node) {
                    entry.insert(nodes.len());
                    nodes.push(node);
                }
            }
        }

        let consensus = self.query_consensus(&nodes, trial_weighting, temperature)?;
        consensus.score_pairs(pairs)
    }

    /// Scores a flat `[u0, v0, u1, v1, ...]` pair buffer.
    pub fn score_flat_pairs(
        &mut self,
        flat_pairs: &[usize],
        trial_weighting: TrialWeighting,
        temperature: Option<f64>,
    ) -> Result<Vec<f64>> {
        if flat_pairs.len() % 2 != 0 {
            return Err(RavenError::InvalidInput(format!(
                "flat pair buffer must contain an even number of node ids, got {}",
                flat_pairs.len()
            )));
        }
        if flat_pairs.is_empty() {
            return Ok(Vec::new());
        }

        let mut nodes = Vec::new();
        let mut positions = HashMap::new();
        for node in flat_pairs.iter().copied() {
            if let std::collections::hash_map::Entry::Vacant(entry) = positions.entry(node) {
                entry.insert(nodes.len());
                nodes.push(node);
            }
        }

        let consensus = self.query_consensus(&nodes, trial_weighting, temperature)?;
        consensus.score_flat_pairs(flat_pairs)
    }

    /// Returns whether a node is currently present in the live graph.
    pub fn contains_node(&self, node: usize) -> bool {
        self.inner.contains_node(&node)
    }

    /// Number of currently live graph nodes.
    pub fn live_node_count(&self) -> usize {
        self.inner.live_node_count()
    }

    /// Sorted currently live graph nodes.
    pub fn live_nodes(&self) -> Vec<usize> {
        let mut nodes = self.inner.live_nodes();
        nodes.sort_unstable();
        nodes
    }

    /// Clears graph and clustering state while preserving the config.
    pub fn clear(&mut self) -> Result<()> {
        self.inner = build_index(&self.config)?;
        Ok(())
    }

    /// Returns the active config.
    pub fn config(&self) -> &RavenConfig {
        &self.config
    }
}

fn build_index(config: &RavenConfig) -> Result<InMemoryIndex<ARITY, usize, f64>> {
    config.validate()?;

    let leiden_config = LeidenConfig {
        seed: config.rng_seed,
        ..LeidenConfig::default()
    };

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
    fn score_pairs_deduplicates_nodes_and_scores_pairs() {
        let mut config = small_config();
        config.num_trials = 3;
        let mut raven = Raven::new(config).unwrap();
        raven.update_edge(1, 2, 1.0).unwrap();
        raven.update_edge(2, 3, 1.0).unwrap();
        raven.update_edge(3, 4, 1.0).unwrap();

        assert!(
            raven
                .score_pairs(&[], TrialWeighting::ScoreSoftmax, None)
                .unwrap()
                .is_empty()
        );

        let scores = raven
            .score_pairs(&[(1, 2), (1, 3), (2, 2)], TrialWeighting::Uniform, None)
            .unwrap();

        assert_eq!(scores.len(), 3);
        assert!(scores.iter().all(|score| (0.0..=1.0).contains(score)));
        assert_eq!(scores[2], 1.0);
    }

    #[test]
    fn query_consensus_returns_reusable_consensus_object() {
        let mut config = small_config();
        config.num_trials = 3;
        let mut raven = Raven::new(config).unwrap();
        raven.update_edge(1, 2, 1.0).unwrap();
        raven.update_edge(2, 3, 1.0).unwrap();
        raven.update_edge(3, 4, 1.0).unwrap();

        let consensus = raven
            .query_consensus(&[1, 2, 3], TrialWeighting::Uniform, None)
            .unwrap();

        assert_eq!(consensus.nodes, vec![1, 2, 3]);
        assert_eq!(consensus.num_trials(), 3);
        assert_eq!(consensus.num_nodes(), 3);
        assert_eq!(consensus.labels.len(), 3);
        assert_eq!(consensus.trial_weights, vec![1.0 / 3.0; 3]);

        let scores = consensus.score_pairs(&[(1, 2), (1, 3), (2, 2)]).unwrap();
        assert_eq!(scores.len(), 3);
        assert_eq!(scores[2], 1.0);

        let matrix = consensus.score_matrix(Some(&[1, 2])).unwrap();
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);
        assert_eq!(matrix[0][0], 1.0);

        assert_eq!(
            consensus
                .threshold_pairs(&[(1, 2), (2, 3)], 0.0)
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            consensus
                .connected_components(&[(1, 2), (2, 3)], 0.0, false)
                .unwrap(),
            vec![vec![1, 2, 3]]
        );

        assert!(matches!(
            consensus.score_pair(1, 99),
            Err(RavenError::InvalidInput(_))
        ));
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
