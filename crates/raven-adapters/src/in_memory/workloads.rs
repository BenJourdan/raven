use std::{collections::VecDeque, fmt};

use rand::{RngExt, SeedableRng};
use raven_core::types::{FloatScalar, Strict, StrictCarrierOps};
use rustc_hash::{FxBuildHasher, FxHashMap};

use super::{InMemoryGraphError, InMemoryUndirectedGraph};

#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedSbmCommands {
    pub nodes: Vec<usize>,
    pub operations: Vec<SbmInstruction>,
    pub expected_edges: usize,
    pub cluster_labels: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SbmInstruction {
    Insert {
        time: i64,
        u: usize,
        v: usize,
        weight_delta: f64,
    },
    Delete {
        time: i64,
        u: usize,
        v: usize,
    },
    SetWeight {
        time: i64,
        u: usize,
        v: usize,
        weight: f64,
    },
}

impl SbmInstruction {
    pub fn time(self) -> i64 {
        match self {
            Self::Insert { time, .. }
            | Self::Delete { time, .. }
            | Self::SetWeight { time, .. } => time,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SbmDiffWorkload<T> {
    pub nodes: Vec<usize>,
    pub cluster_labels: Vec<usize>,
    pub expected_edges: usize,
    pub snapshot_step: i64,
    pub batches: Vec<SbmUpdateBatch<T>>,
}

#[derive(Debug, Clone)]
pub struct SbmUpdateBatch<T> {
    pub time: i64,
    pub edge_ops: Vec<SbmEdgeOp<T>>,
    pub node_ops: Vec<(usize, Option<Strict<T>>)>,
}

impl<T> SbmUpdateBatch<T>
where
    T: FloatScalar,
    Strict<T>: StrictCarrierOps<Scalar = T> + Copy,
{
    pub fn apply_to_graph(
        &self,
        graph: &mut InMemoryUndirectedGraph<usize, T>,
    ) -> Result<(), InMemoryGraphError> {
        for op in &self.edge_ops {
            op.apply_to_graph(graph)?;
        }
        Ok(())
    }

    pub fn apply_to_graph_and_flush_node_ops(
        &self,
        graph: &mut InMemoryUndirectedGraph<usize, T>,
    ) -> Result<Vec<(usize, Option<Strict<T>>)>, InMemoryGraphError> {
        self.apply_to_graph(graph)?;
        Ok(graph.flush_node_ops())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SbmEdgeOp<T> {
    Set {
        u: usize,
        v: usize,
        weight: Strict<T>,
    },
    Delete {
        u: usize,
        v: usize,
    },
}

impl<T> SbmEdgeOp<T>
where
    T: FloatScalar,
    Strict<T>: StrictCarrierOps<Scalar = T> + Copy,
{
    pub fn apply_to_graph(
        self,
        graph: &mut InMemoryUndirectedGraph<usize, T>,
    ) -> Result<(), InMemoryGraphError> {
        match self {
            Self::Set { u, v, weight } => graph.update_edge(u, v, Some(weight)),
            Self::Delete { u, v } => match graph.update_edge(u, v, None) {
                Ok(()) | Err(InMemoryGraphError::MissingEdge) => Ok(()),
                Err(err) => Err(err),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SbmWorkloadError {
    InvalidParameter(&'static str),
    InvalidWeight,
    Graph(InMemoryGraphError),
}

impl fmt::Display for SbmWorkloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(param) => write!(f, "invalid SBM workload parameter: {param}"),
            Self::InvalidWeight => write!(f, "generated edge weight was not positive and finite"),
            Self::Graph(err) => write!(f, "failed to replay generated graph operation: {err}"),
        }
    }
}

impl std::error::Error for SbmWorkloadError {}

impl From<InMemoryGraphError> for SbmWorkloadError {
    fn from(value: InMemoryGraphError) -> Self {
        Self::Graph(value)
    }
}

/// Generate an SBM-style stream of edge updates with finite lifetimes.
///
/// This is a direct Raven-friendly port of the generator used by dyn-cc's
/// `prepare_diff_workload_sbm`: it samples internal vs external endpoints from
/// the expected SBM edge mass, increments edge weights by one, and expires each
/// increment after a fixed lifetime.
pub fn generate_sbm_commands(
    seed: u64,
    n_per_cluster: usize,
    k_clusters: usize,
    p_internal: f64,
    q_external: f64,
    n_multiplier: usize,
    lifetime_multiplier: f64,
) -> Result<GeneratedSbmCommands, SbmWorkloadError> {
    validate_sbm_params(
        n_per_cluster,
        k_clusters,
        p_internal,
        q_external,
        n_multiplier,
        lifetime_multiplier,
        1.0,
    )?;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let total_nodes = n_per_cluster * k_clusters;
    let nodes = (0..total_nodes).collect::<Vec<_>>();

    let cluster_nodes = (0..k_clusters)
        .map(|c| {
            let start = c * n_per_cluster;
            (start..start + n_per_cluster).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let expected_internal = (k_clusters as f64)
        * (n_per_cluster as f64)
        * ((n_per_cluster - 1) as f64)
        * 0.5
        * p_internal;
    let expected_external = (k_clusters * (k_clusters - 1)) as f64
        * 0.5
        * (n_per_cluster * n_per_cluster) as f64
        * q_external;
    let expected_edges = expected_internal + expected_external;
    if !(expected_edges.is_finite() && expected_edges > 0.0) {
        return Err(SbmWorkloadError::InvalidParameter(
            "expected edge count must be positive",
        ));
    }

    let num_updates = ((n_multiplier as f64) * expected_edges).ceil() as usize;
    let lifetime_steps = ((lifetime_multiplier * expected_edges).ceil() as i64).max(1);
    let internal_prob = expected_internal / expected_edges;

    let mut edge_weights = FxHashMap::<(usize, usize), f64>::with_capacity_and_hasher(
        expected_edges.ceil() as usize,
        FxBuildHasher,
    );
    let mut expirations = VecDeque::<(i64, (usize, usize))>::new();
    let mut operations = Vec::with_capacity(num_updates * 2);
    let mut t = 0i64;

    for _ in 0..num_updates {
        while let Some(&(added_at, (u, v))) = expirations.front() {
            if t - added_at < lifetime_steps {
                break;
            }

            expirations.pop_front();
            let key = ordered_edge(u, v);
            if let Some(weight) = edge_weights.get_mut(&key) {
                let new_weight = (*weight - 1.0).max(0.0);
                *weight = new_weight;
                if new_weight == 0.0 {
                    edge_weights.remove(&key);
                    operations.push(SbmInstruction::Delete { time: t, u, v });
                } else {
                    operations.push(SbmInstruction::SetWeight {
                        time: t,
                        u,
                        v,
                        weight: new_weight,
                    });
                }
            }
        }

        let (u, v) = if rng.random_range(0.0..1.0) < internal_prob {
            pick_internal(&mut rng, &cluster_nodes)
        } else {
            pick_cross(&mut rng, &cluster_nodes)
        };
        if u == v {
            t += 1;
            continue;
        }

        operations.push(SbmInstruction::Insert {
            time: t,
            u,
            v,
            weight_delta: 1.0,
        });
        *edge_weights.entry(ordered_edge(u, v)).or_insert(0.0) += 1.0;
        expirations.push_back((t, (u, v)));

        t += 1;
    }

    let mut cluster_labels = Vec::with_capacity(total_nodes);
    for c in 0..k_clusters {
        cluster_labels.extend(std::iter::repeat_n(c, n_per_cluster));
    }

    Ok(GeneratedSbmCommands {
        nodes,
        operations,
        expected_edges: expected_edges as usize,
        cluster_labels,
    })
}

/// Prepare a deterministic SBM update workload for replay into
/// [`InMemoryUndirectedGraph`].
///
/// `step_size` is a fraction of the generated time span, matching dyn-cc's
/// benchmark convention. For example, `0.1` produces roughly ten replay
/// batches.
pub fn prepare_diff_workload_sbm<T>(
    seed: u64,
    n_per_cluster: usize,
    k_clusters: usize,
    p_internal: f64,
    q_external: f64,
    n_multiplier: usize,
    lifetime_multiplier: f64,
    step_size: f64,
) -> Result<SbmDiffWorkload<T>, SbmWorkloadError>
where
    T: FloatScalar,
    Strict<T>: StrictCarrierOps<Scalar = T> + Copy,
{
    validate_sbm_params(
        n_per_cluster,
        k_clusters,
        p_internal,
        q_external,
        n_multiplier,
        lifetime_multiplier,
        step_size,
    )?;

    let commands = generate_sbm_commands(
        seed,
        n_per_cluster,
        k_clusters,
        p_internal,
        q_external,
        n_multiplier,
        lifetime_multiplier,
    )?;
    let Some(first_time) = commands.operations.first().map(|op| op.time()) else {
        return Err(SbmWorkloadError::InvalidParameter(
            "generated command stream was empty",
        ));
    };
    let Some(last_time) = commands.operations.last().map(|op| op.time()) else {
        return Err(SbmWorkloadError::InvalidParameter(
            "generated command stream was empty",
        ));
    };
    if first_time == last_time {
        return Err(SbmWorkloadError::InvalidParameter(
            "generated command stream has zero duration",
        ));
    }

    let time_span = last_time - first_time;
    let snapshot_step = ((step_size * time_span as f64) as i64).max(1);
    let mut graph = InMemoryUndirectedGraph::<usize, T>::new();
    let mut batches = Vec::new();
    let mut cursor = 0usize;

    let mut snapshot_time = first_time + snapshot_step;
    while snapshot_time < last_time {
        let batch =
            drain_snapshot_batch(&commands.operations, &mut cursor, snapshot_time, &mut graph)?;
        if !batch.edge_ops.is_empty() || !batch.node_ops.is_empty() {
            batches.push(batch);
        }
        snapshot_time += snapshot_step;
    }

    let final_batch =
        drain_snapshot_batch(&commands.operations, &mut cursor, last_time, &mut graph)?;
    if !final_batch.edge_ops.is_empty() || !final_batch.node_ops.is_empty() {
        batches.push(final_batch);
    }

    Ok(SbmDiffWorkload {
        nodes: commands.nodes,
        cluster_labels: commands.cluster_labels,
        expected_edges: commands.expected_edges,
        snapshot_step,
        batches,
    })
}

fn drain_snapshot_batch<T>(
    operations: &[SbmInstruction],
    cursor: &mut usize,
    snapshot_time: i64,
    graph: &mut InMemoryUndirectedGraph<usize, T>,
) -> Result<SbmUpdateBatch<T>, SbmWorkloadError>
where
    T: FloatScalar,
    Strict<T>: StrictCarrierOps<Scalar = T> + Copy,
{
    let mut edge_ops = Vec::new();

    while *cursor < operations.len() && operations[*cursor].time() <= snapshot_time {
        let edge_op = instruction_to_edge_op(operations[*cursor], graph)?;
        edge_op.apply_to_graph(graph)?;
        edge_ops.push(edge_op);
        *cursor += 1;
    }

    let node_ops = graph.flush_node_ops();
    Ok(SbmUpdateBatch {
        time: snapshot_time,
        edge_ops,
        node_ops,
    })
}

fn instruction_to_edge_op<T>(
    instruction: SbmInstruction,
    graph: &InMemoryUndirectedGraph<usize, T>,
) -> Result<SbmEdgeOp<T>, SbmWorkloadError>
where
    T: FloatScalar,
    Strict<T>: StrictCarrierOps<Scalar = T> + Copy,
{
    match instruction {
        SbmInstruction::Insert {
            u, v, weight_delta, ..
        } => {
            let old_weight = graph
                .edge_weight(u, v)
                .map(|weight| weight.into_scalar())
                .unwrap_or(T::ZERO);
            let delta = T::from(weight_delta).ok_or(SbmWorkloadError::InvalidWeight)?;
            strict_edge_set(u, v, old_weight + delta)
        }
        SbmInstruction::Delete { u, v, .. } => Ok(SbmEdgeOp::Delete { u, v }),
        SbmInstruction::SetWeight { u, v, weight, .. } => {
            let weight = T::from(weight).ok_or(SbmWorkloadError::InvalidWeight)?;
            strict_edge_set(u, v, weight)
        }
    }
}

fn strict_edge_set<T>(u: usize, v: usize, weight: T) -> Result<SbmEdgeOp<T>, SbmWorkloadError>
where
    T: FloatScalar,
    Strict<T>: StrictCarrierOps<Scalar = T> + Copy,
{
    Ok(SbmEdgeOp::Set {
        u,
        v,
        weight: Strict::<T>::from_positive_scalar(weight)
            .map_err(|_| SbmWorkloadError::InvalidWeight)?,
    })
}

fn validate_sbm_params(
    n_per_cluster: usize,
    k_clusters: usize,
    p_internal: f64,
    q_external: f64,
    n_multiplier: usize,
    lifetime_multiplier: f64,
    step_size: f64,
) -> Result<(), SbmWorkloadError> {
    if n_per_cluster == 0 {
        return Err(SbmWorkloadError::InvalidParameter("n_per_cluster"));
    }
    if k_clusters == 0 {
        return Err(SbmWorkloadError::InvalidParameter("k_clusters"));
    }
    if n_multiplier == 0 {
        return Err(SbmWorkloadError::InvalidParameter("n_multiplier"));
    }
    if !(p_internal.is_finite() && p_internal >= 0.0) {
        return Err(SbmWorkloadError::InvalidParameter("p_internal"));
    }
    if !(q_external.is_finite() && q_external >= 0.0) {
        return Err(SbmWorkloadError::InvalidParameter("q_external"));
    }
    if !(lifetime_multiplier.is_finite() && lifetime_multiplier > 0.0) {
        return Err(SbmWorkloadError::InvalidParameter("lifetime_multiplier"));
    }
    if !(step_size.is_finite() && step_size > 0.0) {
        return Err(SbmWorkloadError::InvalidParameter("step_size"));
    }

    Ok(())
}

fn pick_internal(rng: &mut rand::rngs::StdRng, cluster_nodes: &[Vec<usize>]) -> (usize, usize) {
    let c = rng.random_range(0..cluster_nodes.len());
    let cluster = &cluster_nodes[c];
    if cluster.len() == 1 {
        return (cluster[0], cluster[0]);
    }
    let a = rng.random_range(0..cluster.len());
    let mut b = rng.random_range(0..cluster.len());
    while b == a {
        b = rng.random_range(0..cluster.len());
    }
    (cluster[a], cluster[b])
}

fn pick_cross(rng: &mut rand::rngs::StdRng, cluster_nodes: &[Vec<usize>]) -> (usize, usize) {
    if cluster_nodes.len() < 2 {
        return pick_internal(rng, cluster_nodes);
    }

    let c1 = rng.random_range(0..cluster_nodes.len());
    let mut c2 = rng.random_range(0..cluster_nodes.len());
    while c2 == c1 {
        c2 = rng.random_range(0..cluster_nodes.len());
    }
    let u_cluster = &cluster_nodes[c1];
    let v_cluster = &cluster_nodes[c2];
    (
        u_cluster[rng.random_range(0..u_cluster.len())],
        v_cluster[rng.random_range(0..v_cluster.len())],
    )
}

fn ordered_edge(u: usize, v: usize) -> (usize, usize) {
    if u <= v { (u, v) } else { (v, u) }
}
