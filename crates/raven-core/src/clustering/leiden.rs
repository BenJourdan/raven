//! Native Leiden clustering over Raven coreset sparse matrices.
//!
//! The implementation in this module is adapted from `leiden-rs` 0.8.1, which
//! is licensed `MIT OR Apache-2.0`. Raven only ports the weighted undirected
//! modularity path used by the in-memory coreset query pipeline.

use std::{collections::VecDeque, fmt, sync::Arc};

use faer::sparse::SparseRowMat;
use rand::{SeedableRng, seq::SliceRandom};
use rustc_hash::FxHashMap;

use crate::types::{AlgType, FloatScalar};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum QualityType {
    #[default]
    Modularity,
    CPM,
    RBConfiguration,
    RBER,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LeidenConfig {
    pub max_iterations: usize,
    pub resolution: f64,
    pub seed: Option<u64>,
    pub quality: QualityType,
    pub epsilon: f64,
    pub max_comm_size: usize,
    pub parallel_local_moving_threshold: Option<usize>,
    pub parallel_aggregation_threshold: Option<usize>,
    pub skip_refinement: bool,
    pub min_iterations: usize,
    pub track_quality_history: bool,
}

impl Default for LeidenConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            resolution: 1.0,
            seed: None,
            quality: QualityType::Modularity,
            epsilon: 1e-10,
            max_comm_size: 0,
            parallel_local_moving_threshold: None,
            parallel_aggregation_threshold: None,
            skip_refinement: false,
            min_iterations: 1,
            track_quality_history: false,
        }
    }
}

impl LeidenConfig {
    fn validate(&self) -> Result<(), LeidenError> {
        if self.max_iterations == 0 {
            return Err(LeidenError::InvalidConfig(
                "max_iterations must be non-zero".to_string(),
            ));
        }
        if !self.resolution.is_finite() || self.resolution < 0.0 {
            return Err(LeidenError::InvalidConfig(format!(
                "resolution must be finite and non-negative, got {}",
                self.resolution
            )));
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(LeidenError::InvalidConfig(format!(
                "epsilon must be finite and positive, got {}",
                self.epsilon
            )));
        }
        if self.quality != QualityType::Modularity {
            return Err(LeidenError::UnsupportedQuality(self.quality));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
enum LeidenError {
    InvalidConfig(String),
    UnsupportedQuality(QualityType),
    NonSquareGraph { rows: usize, cols: usize },
    InvalidEdgeWeight { row: usize, col: usize, weight: f64 },
}

impl fmt::Display for LeidenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid Leiden config: {message}"),
            Self::UnsupportedQuality(quality) => {
                write!(f, "unsupported native Leiden quality: {quality:?}")
            }
            Self::NonSquareGraph { rows, cols } => {
                write!(
                    f,
                    "Leiden clustering requires a square graph, got {rows}x{cols}"
                )
            }
            Self::InvalidEdgeWeight { row, col, weight } => write!(
                f,
                "Leiden clustering requires finite non-negative edge weights; ({row}, {col}) has {weight}"
            ),
        }
    }
}

impl std::error::Error for LeidenError {}

/// Cluster a symmetric sparse graph using Raven's native Leiden implementation.
///
/// Raven's coreset graph is treated as undirected. Diagonal entries are ignored
/// in the input graph. v1 supports only weighted undirected modularity.
pub fn leiden_community_detection<T>(
    graph: &mut SparseRowMat<usize, T>,
    config: &LeidenConfig,
) -> (Vec<usize>, usize)
where
    T: FloatScalar,
{
    leiden_community_detection_result(graph, config)
        .unwrap_or_else(|err| panic!("Leiden clustering failed: {err}"))
}

fn leiden_community_detection_result<T>(
    graph: &SparseRowMat<usize, T>,
    config: &LeidenConfig,
) -> Result<(Vec<usize>, usize), LeidenError>
where
    T: FloatScalar,
{
    config.validate()?;
    let input = SparseInputGraph::new(graph)?;
    let output = run_leiden(input, config)?;
    Ok((output.partition.membership, output.num_communities))
}

/// Wrap [`leiden_community_detection`] as a Raven clustering callback.
///
/// The requested cluster count is ignored. Leiden chooses its own number of
/// communities from the configured modularity objective and resolution.
pub fn leiden_community_detection_alg<T>(config: LeidenConfig) -> AlgType<T>
where
    T: FloatScalar + Send + Sync + 'static,
{
    Arc::new(move |graph, _requested_k| leiden_community_detection(graph, &config))
}

/// Reference adapter that preserves the old `leiden-rs` conversion path.
///
/// This is intentionally kept for benchmarks and tests while Raven's native
/// implementation settles.
pub fn leiden_rs_community_detection<T>(
    graph: &mut SparseRowMat<usize, T>,
    config: &LeidenConfig,
) -> (Vec<usize>, usize)
where
    T: FloatScalar,
{
    let reference_config = leiden_rs::LeidenConfig {
        max_iterations: config.max_iterations,
        resolution: config.resolution,
        seed: config.seed,
        quality: match config.quality {
            QualityType::Modularity => leiden_rs::QualityType::Modularity,
            QualityType::CPM => leiden_rs::QualityType::CPM,
            QualityType::RBConfiguration => leiden_rs::QualityType::RBConfiguration,
            QualityType::RBER => leiden_rs::QualityType::RBER,
        },
        epsilon: config.epsilon,
        max_comm_size: config.max_comm_size,
        parallel_local_moving_threshold: config.parallel_local_moving_threshold,
        parallel_aggregation_threshold: config.parallel_aggregation_threshold,
        skip_refinement: config.skip_refinement,
        min_iterations: config.min_iterations,
        track_quality_history: config.track_quality_history,
    };
    reference_leiden_community_detection(graph, &reference_config)
}

pub fn leiden_rs_community_detection_alg<T>(config: LeidenConfig) -> AlgType<T>
where
    T: FloatScalar + Send + Sync + 'static,
{
    Arc::new(move |graph, _requested_k| leiden_rs_community_detection(graph, &config))
}

fn reference_leiden_community_detection<T>(
    graph: &mut SparseRowMat<usize, T>,
    config: &leiden_rs::LeidenConfig,
) -> (Vec<usize>, usize)
where
    T: FloatScalar,
{
    use leiden_rs::{GraphDataBuilder, Leiden};

    let (symbolic, vals) = graph.parts();
    let (nrows, ncols, row_ptr, _row_nnz, col_idx) = symbolic.parts();
    assert_eq!(nrows, ncols, "Leiden clustering requires a square graph");

    if nrows == 0 {
        return (Vec::new(), 0);
    }

    let mut builder = GraphDataBuilder::new(nrows);
    for i in 0..nrows {
        for idx in row_ptr[i]..row_ptr[i + 1] {
            let j = col_idx[idx];
            if i >= j {
                continue;
            }

            let weight = vals[idx];
            assert!(
                weight.is_finite() && weight >= T::ZERO,
                "Leiden clustering requires finite non-negative edge weights"
            );
            if weight > T::ZERO {
                builder
                    .add_edge(
                        i,
                        j,
                        weight
                            .to_f64()
                            .expect("finite Raven float should convert to f64"),
                    )
                    .expect("validated Leiden graph edge should be accepted");
            }
        }
    }

    let leiden_graph = builder
        .build()
        .expect("validated Leiden graph should build successfully");
    let output = Leiden::new(config.clone())
        .run(&leiden_graph)
        .expect("Leiden clustering failed");
    let labels = output.partition.as_slice().to_vec();
    let num_communities = output.partition.num_communities();

    (labels, num_communities)
}

struct LeidenOutput {
    partition: Partition,
    num_communities: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Partition {
    membership: Vec<usize>,
    num_communities: usize,
}

impl Partition {
    fn singleton(n: usize) -> Self {
        Self {
            membership: (0..n).collect(),
            num_communities: n,
        }
    }

    fn from_membership(membership: Vec<usize>) -> Self {
        let num_communities = membership.iter().copied().max().map_or(0, |max| max + 1);
        Self {
            membership,
            num_communities,
        }
    }

    #[inline]
    fn community_of(&self, node: usize) -> usize {
        self.membership[node]
    }

    fn move_node(&mut self, node: usize, community: usize) {
        self.membership[node] = community;
        if community >= self.num_communities {
            self.num_communities = community + 1;
        }
    }

    fn renumber(&mut self) {
        if self.membership.is_empty() {
            self.num_communities = 0;
            return;
        }

        let max_comm = self.membership.iter().copied().max().unwrap_or(0);
        let mut mapping = vec![usize::MAX; max_comm + 1];
        let mut next = 0usize;
        for comm in &mut self.membership {
            if mapping[*comm] == usize::MAX {
                mapping[*comm] = next;
                next += 1;
            }
            *comm = mapping[*comm];
        }
        self.num_communities = next;
    }
}

trait UndirectedGraph {
    fn node_count(&self) -> usize;
    fn total_weight(&self) -> f64;
    fn degree(&self, node: usize) -> f64;
    fn node_weight(&self, node: usize) -> f64;
    fn for_each_neighbor(&self, node: usize, visitor: impl FnMut(usize, f64));
}

struct SparseInputGraph<'a, T> {
    n: usize,
    row_ptr: &'a [usize],
    col_idx: &'a [usize],
    vals: &'a [T],
    degree: Vec<f64>,
    total_weight: f64,
}

impl<'a, T> SparseInputGraph<'a, T>
where
    T: FloatScalar,
{
    fn new(graph: &'a SparseRowMat<usize, T>) -> Result<Self, LeidenError> {
        let (symbolic, vals) = graph.parts();
        let (nrows, ncols, row_ptr, _row_nnz, col_idx) = symbolic.parts();
        if nrows != ncols {
            return Err(LeidenError::NonSquareGraph {
                rows: nrows,
                cols: ncols,
            });
        }

        let mut degree = vec![0.0; nrows];
        for row in 0..nrows {
            for entry in row_ptr[row]..row_ptr[row + 1] {
                let col = col_idx[entry];
                if row == col {
                    continue;
                }
                let weight = vals[entry].to_f64().ok_or(LeidenError::InvalidEdgeWeight {
                    row,
                    col,
                    weight: f64::NAN,
                })?;
                if !weight.is_finite() || weight < 0.0 {
                    return Err(LeidenError::InvalidEdgeWeight { row, col, weight });
                }
                if weight > 0.0 {
                    degree[row] += weight;
                }
            }
        }
        let total_weight = degree.iter().sum::<f64>() / 2.0;

        Ok(Self {
            n: nrows,
            row_ptr,
            col_idx,
            vals,
            degree,
            total_weight,
        })
    }
}

impl<T> UndirectedGraph for SparseInputGraph<'_, T>
where
    T: FloatScalar,
{
    #[inline]
    fn node_count(&self) -> usize {
        self.n
    }

    #[inline]
    fn total_weight(&self) -> f64 {
        self.total_weight
    }

    #[inline]
    fn degree(&self, node: usize) -> f64 {
        self.degree.get(node).copied().unwrap_or(0.0)
    }

    #[inline]
    fn node_weight(&self, node: usize) -> f64 {
        if node < self.n { 1.0 } else { 0.0 }
    }

    #[inline]
    fn for_each_neighbor(&self, node: usize, mut visitor: impl FnMut(usize, f64)) {
        if node >= self.n {
            return;
        }
        for entry in self.row_ptr[node]..self.row_ptr[node + 1] {
            let neighbour = self.col_idx[entry];
            if neighbour == node {
                continue;
            }
            let weight = self.vals[entry]
                .to_f64()
                .expect("validated sparse input edge should convert to f64");
            if weight > 0.0 {
                visitor(neighbour, weight);
            }
        }
    }
}

#[derive(Debug, Clone)]
struct OwnedGraph {
    n: usize,
    offsets: Vec<usize>,
    targets: Vec<usize>,
    weights: Vec<f64>,
    degree: Vec<f64>,
    node_weight: Vec<f64>,
    total_weight: f64,
}

impl OwnedGraph {
    fn from_edges(n: usize, edges: FxHashMap<(usize, usize), f64>, node_weight: Vec<f64>) -> Self {
        let mut rows = vec![Vec::<(usize, f64)>::new(); n];
        let mut degree = vec![0.0; n];
        let mut total_weight = 0.0;

        let mut sorted_edges = edges.into_iter().collect::<Vec<_>>();
        sorted_edges.sort_unstable_by_key(|&((u, v), _)| (u, v));

        for ((u, v), weight) in sorted_edges {
            if weight <= 0.0 {
                continue;
            }
            if u == v {
                rows[u].push((u, weight));
                degree[u] += 2.0 * weight;
                total_weight += weight;
            } else {
                rows[u].push((v, weight));
                rows[v].push((u, weight));
                degree[u] += weight;
                degree[v] += weight;
                total_weight += weight;
            }
        }

        let mut offsets = Vec::with_capacity(n + 1);
        let mut targets = Vec::new();
        let mut weights = Vec::new();
        offsets.push(0);
        for row in &mut rows {
            row.sort_unstable_by_key(|&(target, _)| target);
            targets.extend(row.iter().map(|&(target, _)| target));
            weights.extend(row.iter().map(|&(_, weight)| weight));
            offsets.push(targets.len());
        }
        Self {
            n,
            offsets,
            targets,
            weights,
            degree,
            node_weight,
            total_weight,
        }
    }
}

impl UndirectedGraph for OwnedGraph {
    #[inline]
    fn node_count(&self) -> usize {
        self.n
    }

    #[inline]
    fn total_weight(&self) -> f64 {
        self.total_weight
    }

    #[inline]
    fn degree(&self, node: usize) -> f64 {
        self.degree.get(node).copied().unwrap_or(0.0)
    }

    #[inline]
    fn node_weight(&self, node: usize) -> f64 {
        self.node_weight.get(node).copied().unwrap_or(0.0)
    }

    #[inline]
    fn for_each_neighbor(&self, node: usize, mut visitor: impl FnMut(usize, f64)) {
        if node >= self.n {
            return;
        }
        for entry in self.offsets[node]..self.offsets[node + 1] {
            visitor(self.targets[entry], self.weights[entry]);
        }
    }
}

enum CurrentGraph<'a, T> {
    Sparse(SparseInputGraph<'a, T>),
    Owned(OwnedGraph),
}

impl<T> UndirectedGraph for CurrentGraph<'_, T>
where
    T: FloatScalar,
{
    fn node_count(&self) -> usize {
        match self {
            Self::Sparse(graph) => graph.node_count(),
            Self::Owned(graph) => graph.node_count(),
        }
    }

    fn total_weight(&self) -> f64 {
        match self {
            Self::Sparse(graph) => graph.total_weight(),
            Self::Owned(graph) => graph.total_weight(),
        }
    }

    fn degree(&self, node: usize) -> f64 {
        match self {
            Self::Sparse(graph) => graph.degree(node),
            Self::Owned(graph) => graph.degree(node),
        }
    }

    fn node_weight(&self, node: usize) -> f64 {
        match self {
            Self::Sparse(graph) => graph.node_weight(node),
            Self::Owned(graph) => graph.node_weight(node),
        }
    }

    fn for_each_neighbor(&self, node: usize, visitor: impl FnMut(usize, f64)) {
        match self {
            Self::Sparse(graph) => graph.for_each_neighbor(node, visitor),
            Self::Owned(graph) => graph.for_each_neighbor(node, visitor),
        }
    }
}

fn run_leiden<T>(
    input: SparseInputGraph<'_, T>,
    config: &LeidenConfig,
) -> Result<LeidenOutput, LeidenError>
where
    T: FloatScalar,
{
    let original_n = input.node_count();
    if original_n == 0 {
        return Ok(LeidenOutput {
            partition: Partition::singleton(0),
            num_communities: 0,
        });
    }

    let mut graph = CurrentGraph::Sparse(input);
    let mut partition = Partition::singleton(graph.node_count());
    let mut flat_mapping = (0..graph.node_count()).collect::<Vec<_>>();
    let mut rng = match config.seed {
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
        None => rand::make_rng(),
    };

    for iteration in 0..config.max_iterations {
        let q_before = modularity_quality(&graph, &partition, config.resolution);
        let changed = local_moving(&graph, &mut partition, config, &mut rng);
        if !changed {
            break;
        }

        partition.renumber();
        let q_after = modularity_quality(&graph, &partition, config.resolution);
        if iteration >= config.min_iterations && (q_after - q_before).abs() < config.epsilon {
            break;
        }

        let refined = if config.skip_refinement {
            partition.clone()
        } else {
            refinement(&graph, &partition, config, &mut rng)
        };
        let (aggregate, orig_to_agg, aggregate_initial) = aggregate(&graph, &refined, &partition);

        for original_node in 0..original_n {
            flat_mapping[original_node] = orig_to_agg[flat_mapping[original_node]];
        }

        graph = CurrentGraph::Owned(aggregate);
        partition = aggregate_initial;
        if graph.node_count() <= 1 {
            break;
        }
    }

    let mut result = Partition::from_membership(vec![0; original_n]);
    for (original_node, &aggregate_node) in flat_mapping.iter().enumerate() {
        result.move_node(original_node, partition.community_of(aggregate_node));
    }
    result.renumber();
    let num_communities = result.num_communities;

    Ok(LeidenOutput {
        partition: result,
        num_communities,
    })
}

fn modularity_delta(
    resolution: f64,
    two_m: f64,
    k_v: f64,
    k_v_to_target: f64,
    k_v_to_current: f64,
    sigma_target: f64,
    sigma_current: f64,
) -> f64 {
    if two_m == 0.0 {
        return 0.0;
    }
    (k_v_to_target - k_v_to_current) * 2.0 / two_m
        - resolution * k_v * (sigma_target - sigma_current + k_v) * 2.0 / (two_m * two_m)
}

fn modularity_quality<G: UndirectedGraph>(
    graph: &G,
    partition: &Partition,
    resolution: f64,
) -> f64 {
    let n = graph.node_count();
    let m = graph.total_weight();
    if m == 0.0 {
        return 0.0;
    }

    let mut sigma_tot = vec![0.0; partition.num_communities];
    let mut internal = vec![0.0; partition.num_communities];

    for node in 0..n {
        let community = partition.community_of(node);
        sigma_tot[community] += graph.degree(node);
        graph.for_each_neighbor(node, |neighbour, weight| {
            if neighbour >= node && partition.community_of(neighbour) == community {
                internal[community] += weight;
            }
        });
    }

    let two_m = 2.0 * m;
    (0..partition.num_communities)
        .map(|community| {
            internal[community] / m - resolution * (sigma_tot[community] / two_m).powi(2)
        })
        .sum()
}

fn local_moving<G: UndirectedGraph>(
    graph: &G,
    partition: &mut Partition,
    config: &LeidenConfig,
    rng: &mut rand::rngs::StdRng,
) -> bool {
    let n = graph.node_count();
    if n == 0 || graph.total_weight() <= 0.0 {
        return false;
    }

    let mut community_degree = vec![0.0; n];
    let mut community_size = vec![0.0; n];
    for node in 0..n {
        let community = partition.community_of(node);
        community_degree[community] += graph.degree(node);
        community_size[community] += graph.node_weight(node);
    }

    let mut order = (0..n)
        .filter(|&node| graph.degree(node) > 0.0)
        .collect::<Vec<_>>();
    order.shuffle(rng);
    let mut queue = order.into_iter().collect::<VecDeque<_>>();
    let mut in_queue = vec![false; n];
    for &node in &queue {
        in_queue[node] = true;
    }

    let mut edge_weight_to_community = vec![0.0; n];
    let mut touched = Vec::with_capacity(64);
    let mut touched_mark = vec![false; n];
    let two_m = 2.0 * graph.total_weight();
    let mut changed = false;

    while let Some(node) = queue.pop_front() {
        in_queue[node] = false;
        let current_community = partition.community_of(node);
        let mut current_touched = false;

        graph.for_each_neighbor(node, |neighbour, weight| {
            if neighbour == node {
                return;
            }
            let community = partition.community_of(neighbour);
            if edge_weight_to_community[community] == 0.0 {
                if community == current_community {
                    current_touched = true;
                } else if !touched_mark[community] {
                    touched_mark[community] = true;
                    touched.push(community);
                }
            }
            edge_weight_to_community[community] += weight;
        });

        let node_degree = graph.degree(node);
        let node_weight = graph.node_weight(node);
        let current_edge_weight = edge_weight_to_community[current_community];
        let mut best_community = current_community;
        let mut best_delta = config.epsilon;
        for &target_community in &touched {
            if config.max_comm_size > 0
                && community_size[target_community] + node_weight > config.max_comm_size as f64
            {
                continue;
            }
            let delta = modularity_delta(
                config.resolution,
                two_m,
                node_degree,
                edge_weight_to_community[target_community],
                current_edge_weight,
                community_degree[target_community],
                community_degree[current_community],
            );
            if delta > best_delta {
                best_delta = delta;
                best_community = target_community;
            }
        }

        if current_touched {
            edge_weight_to_community[current_community] = 0.0;
        }
        for &community in &touched {
            edge_weight_to_community[community] = 0.0;
            touched_mark[community] = false;
        }
        touched.clear();

        if best_community != current_community {
            partition.move_node(node, best_community);
            community_degree[current_community] -= node_degree;
            community_degree[best_community] += node_degree;
            community_size[current_community] -= node_weight;
            community_size[best_community] += node_weight;
            changed = true;

            graph.for_each_neighbor(node, |neighbour, _| {
                if !in_queue[neighbour] {
                    queue.push_back(neighbour);
                    in_queue[neighbour] = true;
                }
            });
        }
    }

    changed
}

fn refinement<G: UndirectedGraph>(
    graph: &G,
    partition: &Partition,
    config: &LeidenConfig,
    rng: &mut rand::rngs::StdRng,
) -> Partition {
    if graph.total_weight() <= 0.0 {
        return Partition::singleton(graph.node_count());
    }

    let n = graph.node_count();
    let mut community_nodes = vec![Vec::new(); partition.num_communities];
    for node in 0..n {
        community_nodes[partition.community_of(node)].push(node);
    }
    for nodes in &mut community_nodes {
        nodes.shuffle(rng);
    }

    let mut refined = Partition::singleton(n);
    for (community, nodes) in community_nodes.iter().enumerate() {
        let moves = refine_community(graph, partition, community, nodes, config);
        for (node, new_community) in moves {
            refined.move_node(node, new_community);
        }
    }
    refined.renumber();
    refined
}

fn refine_community<G: UndirectedGraph>(
    graph: &G,
    coarse_partition: &Partition,
    community: usize,
    nodes: &[usize],
    config: &LeidenConfig,
) -> Vec<(usize, usize)> {
    if nodes.len() <= 1 {
        return Vec::new();
    }

    let n = graph.node_count();
    let mut refined_map = (0..n).collect::<Vec<_>>();
    let mut community_degree = vec![0.0; n];
    let mut community_size = vec![0.0; n];
    for &node in nodes {
        community_degree[node] += graph.degree(node);
        community_size[node] += graph.node_weight(node);
    }

    let mut edge_weight_to_community = vec![0.0; n];
    let mut touched = Vec::with_capacity(64);
    let mut touched_mark = vec![false; n];
    let two_m = 2.0 * graph.total_weight();
    let mut moves = Vec::new();

    for &node in nodes {
        let current_refined = refined_map[node];

        graph.for_each_neighbor(node, |neighbour, weight| {
            if neighbour == node || coarse_partition.community_of(neighbour) != community {
                return;
            }
            let refined_community = refined_map[neighbour];
            if edge_weight_to_community[refined_community] == 0.0
                && refined_community != current_refined
                && !touched_mark[refined_community]
            {
                touched_mark[refined_community] = true;
                touched.push(refined_community);
            }
            edge_weight_to_community[refined_community] += weight;
        });

        let node_degree = graph.degree(node);
        let node_weight = graph.node_weight(node);
        let current_edge_weight = edge_weight_to_community[current_refined];
        let mut best_refined = current_refined;
        let mut best_delta = config.epsilon;
        for &target_refined in &touched {
            let delta = modularity_delta(
                config.resolution,
                two_m,
                node_degree,
                edge_weight_to_community[target_refined],
                current_edge_weight,
                community_degree[target_refined],
                community_degree[current_refined],
            );
            if delta > best_delta {
                best_delta = delta;
                best_refined = target_refined;
            }
        }

        edge_weight_to_community[current_refined] = 0.0;
        for &refined_community in &touched {
            edge_weight_to_community[refined_community] = 0.0;
            touched_mark[refined_community] = false;
        }
        touched.clear();

        if best_refined != current_refined {
            refined_map[node] = best_refined;
            community_degree[current_refined] -= node_degree;
            community_degree[best_refined] += node_degree;
            community_size[current_refined] -= node_weight;
            community_size[best_refined] += node_weight;
            moves.push((node, best_refined));
        }
    }

    moves
}

fn aggregate<G: UndirectedGraph>(
    graph: &G,
    refined_partition: &Partition,
    coarse_partition: &Partition,
) -> (OwnedGraph, Vec<usize>, Partition) {
    let n = graph.node_count();
    let (orig_to_agg, agg_n) = build_orig_to_agg_mapping(n, refined_partition);

    let mut edge_map = FxHashMap::<(usize, usize), f64>::default();
    for u in 0..n {
        let ru = orig_to_agg[u];
        graph.for_each_neighbor(u, |v, weight| {
            if u == v {
                *edge_map.entry((ru, ru)).or_default() += weight;
            } else if v > u {
                let rv = orig_to_agg[v];
                let key = if ru <= rv { (ru, rv) } else { (rv, ru) };
                *edge_map.entry(key).or_default() += weight;
            }
        });
    }

    let mut node_weight = vec![0.0; agg_n];
    for (original, &aggregate_node) in orig_to_agg.iter().enumerate() {
        node_weight[aggregate_node] += graph.node_weight(original);
    }
    let aggregate_graph = OwnedGraph::from_edges(agg_n, edge_map, node_weight);

    let mut aggregate_initial = Partition::singleton(agg_n);
    for (original, &aggregate_node) in orig_to_agg.iter().enumerate() {
        aggregate_initial.move_node(aggregate_node, coarse_partition.community_of(original));
    }
    aggregate_initial.renumber();

    (aggregate_graph, orig_to_agg, aggregate_initial)
}

fn build_orig_to_agg_mapping(n: usize, refined_partition: &Partition) -> (Vec<usize>, usize) {
    let mut orig_to_agg = vec![0; n];
    let mut community_to_agg = vec![usize::MAX; refined_partition.num_communities];
    let mut next = 0usize;

    for (node, aggregate) in orig_to_agg.iter_mut().enumerate() {
        let community = refined_partition.community_of(node);
        if community_to_agg[community] == usize::MAX {
            community_to_agg[community] = next;
            next += 1;
        }
        *aggregate = community_to_agg[community];
    }

    (orig_to_agg, next)
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::SymbolicSparseRowMat;
    use rand::{RngExt, SeedableRng};

    fn graph_from_rows<T: FloatScalar>(mut rows: Vec<Vec<(usize, T)>>) -> SparseRowMat<usize, T> {
        for row in &mut rows {
            row.sort_unstable_by_key(|(col, _)| *col);
        }

        let n = rows.len();
        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        let mut nnz_per_row = Vec::with_capacity(n);
        indptr.push(0);

        for row in rows {
            nnz_per_row.push(row.len());
            for (col, weight) in row {
                indices.push(col);
                data.push(weight);
            }
            indptr.push(indices.len());
        }

        SparseRowMat::new(
            SymbolicSparseRowMat::new_checked(n, n, indptr, Some(nnz_per_row), indices),
            data,
        )
    }

    fn two_block_graph<T: FloatScalar>(with_diagonal: bool) -> SparseRowMat<usize, T> {
        let n = 6;
        let mut rows = vec![Vec::<(usize, T)>::new(); n];
        let strong = T::ONE;
        let weak = T::from(0.01).expect("test scalar should convert");
        let diagonal = T::from(1000.0).expect("test scalar should convert");

        for &(i, j, weight) in &[
            (0, 1, strong),
            (0, 2, strong),
            (1, 2, strong),
            (3, 4, strong),
            (3, 5, strong),
            (4, 5, strong),
            (2, 3, weak),
        ] {
            rows[i].push((j, weight));
            rows[j].push((i, weight));
        }

        if with_diagonal {
            for (i, row) in rows.iter_mut().enumerate() {
                row.push((i, diagonal));
            }
        }

        graph_from_rows(rows)
    }

    fn same_partition(lhs: &[usize], rhs: &[usize]) -> bool {
        if lhs.len() != rhs.len() {
            return false;
        }
        for i in 0..lhs.len() {
            for j in 0..lhs.len() {
                if (lhs[i] == lhs[j]) != (rhs[i] == rhs[j]) {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn empty_graph_returns_empty_labels() {
        let mut graph = SparseRowMat::new(
            SymbolicSparseRowMat::new_checked(0, 0, vec![0], Some(Vec::new()), Vec::new()),
            Vec::<f64>::new(),
        );

        assert_eq!(
            leiden_community_detection(&mut graph, &LeidenConfig::default()),
            (Vec::new(), 0)
        );
    }

    #[test]
    fn single_node_graph_returns_singleton_partition() {
        let mut graph = graph_from_rows::<f64>(vec![Vec::new()]);
        assert_eq!(
            leiden_community_detection(&mut graph, &LeidenConfig::default()),
            (vec![0], 1)
        );
    }

    #[test]
    fn diagonal_entries_are_ignored() {
        let config = LeidenConfig {
            seed: Some(42),
            ..LeidenConfig::default()
        };
        let mut without_diagonal = two_block_graph::<f64>(false);
        let mut with_diagonal = two_block_graph::<f64>(true);

        let baseline = leiden_community_detection(&mut without_diagonal, &config);
        let diagonal = leiden_community_detection(&mut with_diagonal, &config);

        assert_eq!(baseline, diagonal);
        assert_eq!(baseline.0.len(), 6);
    }

    #[test]
    fn zero_weight_edges_are_ignored() {
        let mut rows = vec![Vec::<(usize, f64)>::new(); 3];
        rows[0].push((1, 1.0));
        rows[1].push((0, 1.0));
        rows[1].push((2, 0.0));
        rows[2].push((1, 0.0));
        let mut with_zero = graph_from_rows(rows);

        let mut rows = vec![Vec::<(usize, f64)>::new(); 3];
        rows[0].push((1, 1.0));
        rows[1].push((0, 1.0));
        let mut without_zero = graph_from_rows(rows);

        let config = LeidenConfig {
            seed: Some(7),
            ..LeidenConfig::default()
        };
        assert_eq!(
            leiden_community_detection(&mut with_zero, &config),
            leiden_community_detection(&mut without_zero, &config)
        );
    }

    #[test]
    #[should_panic(expected = "finite non-negative edge weights")]
    fn invalid_weights_are_rejected() {
        let mut graph = graph_from_rows(vec![vec![(1, -1.0)], vec![(0, -1.0)]]);
        let _ = leiden_community_detection(&mut graph, &LeidenConfig::default());
    }

    #[test]
    #[should_panic(expected = "unsupported")]
    fn unsupported_quality_is_rejected() {
        let mut graph = two_block_graph::<f64>(false);
        let config = LeidenConfig {
            quality: QualityType::CPM,
            ..LeidenConfig::default()
        };
        let _ = leiden_community_detection(&mut graph, &config);
    }

    #[test]
    fn native_matches_reference_on_two_block_graph() {
        let config = LeidenConfig {
            seed: Some(42),
            ..LeidenConfig::default()
        };
        let mut native_graph = two_block_graph::<f64>(false);
        let mut reference_graph = two_block_graph::<f64>(false);

        let native = leiden_community_detection(&mut native_graph, &config);
        let reference = leiden_rs_community_detection(&mut reference_graph, &config);

        assert_eq!(native.1, reference.1);
        assert!(same_partition(&native.0, &reference.0));
    }

    #[test]
    fn native_is_deterministic_with_fixed_seed() {
        let config = LeidenConfig {
            seed: Some(99),
            ..LeidenConfig::default()
        };
        let mut first = two_block_graph::<f64>(false);
        let mut second = two_block_graph::<f64>(false);

        assert_eq!(
            leiden_community_detection(&mut first, &config),
            leiden_community_detection(&mut second, &config)
        );
    }

    #[test]
    fn random_symmetric_graphs_match_reference_partition_relation() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        for case in 0..12 {
            let n = 8 + case;
            let mut rows = vec![Vec::<(usize, f64)>::new(); n];
            for i in 0..n {
                for j in (i + 1)..n {
                    if rng.random_range(0.0..1.0) < 0.35 {
                        let weight = rng.random_range(0.05..3.0);
                        rows[i].push((j, weight));
                        rows[j].push((i, weight));
                    }
                }
            }

            let config = LeidenConfig {
                seed: Some(11 + case as u64),
                ..LeidenConfig::default()
            };
            let mut native_graph = graph_from_rows(rows.clone());
            let mut reference_graph = graph_from_rows(rows);
            let native = leiden_community_detection(&mut native_graph, &config);
            let reference = leiden_rs_community_detection(&mut reference_graph, &config);

            assert_eq!(native.1, reference.1, "case {case}");
            assert!(
                same_partition(&native.0, &reference.0),
                "case {case}: native={:?}, reference={:?}",
                native.0,
                reference.0
            );
        }
    }

    #[test]
    fn local_moving_does_not_reduce_modularity() {
        let sparse = two_block_graph::<f64>(false);
        let graph = SparseInputGraph::new(&sparse).unwrap();
        let mut partition = Partition::singleton(graph.node_count());
        let before = modularity_quality(&graph, &partition, 1.0);
        let config = LeidenConfig {
            seed: Some(5),
            ..LeidenConfig::default()
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        local_moving(&graph, &mut partition, &config, &mut rng);
        partition.renumber();
        let after = modularity_quality(&graph, &partition, 1.0);
        assert!(after + 1e-12 >= before);
    }

    #[test]
    fn aggregation_preserves_node_weights_and_internal_edges() {
        let sparse = two_block_graph::<f64>(false);
        let graph = SparseInputGraph::new(&sparse).unwrap();
        let refined = Partition::from_membership(vec![0, 0, 1, 1, 1, 1]);
        let coarse = refined.clone();

        let (aggregate, _mapping, initial) = aggregate(&graph, &refined, &coarse);

        assert_eq!(aggregate.node_count(), 2);
        assert_eq!(initial.num_communities, 2);
        assert!((aggregate.node_weight(0) - 2.0).abs() < 1e-10);
        assert!((aggregate.node_weight(1) - 4.0).abs() < 1e-10);
        assert!(aggregate.total_weight() > 0.0);
    }
}
