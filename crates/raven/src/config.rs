use crate::{RavenError, Result};

/// Configuration for an in-memory [`Raven`](crate::Raven) index.
///
/// `RavenConfig::new(num_clusters)` provides conservative defaults for an
/// in-memory workload. Override capacity hints when the graph size is known
/// ahead of time to avoid avoidable allocation during ingestion.
#[derive(Debug, Clone, PartialEq)]
pub struct RavenConfig {
    /// Target number of clusters requested from the coreset clustering step.
    pub num_clusters: usize,
    /// Smoothing scale used by the Raven coreset sampler.
    pub sigma: f64,
    /// Number of points sampled into each coreset trial.
    pub coreset_size: usize,
    /// Number of repair seeds used while extracting each coreset.
    pub sampling_seeds: usize,
    /// Number of independent query trials to run.
    pub num_trials: usize,
    /// Optional deterministic RNG seed.
    pub rng_seed: Option<u64>,
    /// Initial node capacity hint for the in-memory graph.
    pub node_capacity: usize,
    /// Initial expected edges-per-node capacity hint.
    pub expected_edges_per_node: usize,
    /// Pending degree-change threshold before dense degree rebuilds are used.
    pub degree_rebuild_threshold: usize,
}

impl RavenConfig {
    /// Builds a config with defaults suitable for a first in-memory index.
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

    /// Default seed count for a requested number of clusters and coreset size.
    pub fn default_sampling_seeds(num_clusters: usize, coreset_size: usize) -> usize {
        (num_clusters.saturating_mul(4))
            .min(coreset_size.saturating_sub(1))
            .max(2)
    }

    /// Validates the config before index construction.
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
