use std::{fmt, future::Future, hash::Hash, num::TryFromIntError, time::Duration};

use moka::future::Cache;
use neo4rs::{BoltMap, BoltString, BoltType, Graph, query};
use raven_core::types::Strict;
use rustc_hash::FxHashSet;

use crate::remote::{
    OwnedNeighbourhoods, RemoteGraphBackend, RemoteGraphError, SnapshotId,
    memgraph::MemgraphQueries,
};

type CacheKey<V> = (SnapshotId, V);
type CachedRow<V> = Vec<(V, Strict<f64>)>;
type MemgraphRowCache<V> = Cache<CacheKey<V>, CachedRow<V>>;

#[derive(Clone)]
pub struct MemgraphBackend<V> {
    graph: Graph,
    queries: MemgraphQueries,
    cache: MemgraphRowCache<V>,
    filter_strategy: MemgraphFilterStrategy,
}

impl<V> MemgraphBackend<V>
where
    V: MemgraphNodeId,
{
    pub fn new(graph: Graph) -> Self {
        Self::with_config(
            graph,
            MemgraphQueries::default(),
            MemgraphCacheConfig::default(),
        )
    }

    pub fn with_queries(graph: Graph, queries: MemgraphQueries) -> Self {
        Self::with_config(graph, queries, MemgraphCacheConfig::default())
    }

    pub fn with_config(
        graph: Graph,
        queries: MemgraphQueries,
        cache_config: MemgraphCacheConfig,
    ) -> Self {
        Self {
            graph,
            queries,
            cache: cache_config.build_cache(),
            filter_strategy: MemgraphFilterStrategy::default(),
        }
    }

    pub fn with_filter_strategy(mut self, filter_strategy: MemgraphFilterStrategy) -> Self {
        self.filter_strategy = filter_strategy;
        self
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn queries(&self) -> &MemgraphQueries {
        &self.queries
    }

    async fn complete_rows(
        &self,
        snapshot: SnapshotId,
        nodes: Vec<V>,
    ) -> Result<Vec<CachedRow<V>>, MemgraphBackendError> {
        let mut rows = vec![None; nodes.len()];
        let mut missing_nodes = Vec::new();
        let mut missing_indices = Vec::new();

        for (index, node) in nodes.iter().enumerate() {
            let key = (snapshot, node.clone());
            if let Some(row) = self.cache.get(&key).await {
                rows[index] = Some(row);
            } else {
                missing_indices.push(index);
                missing_nodes.push(node.clone());
            }
        }

        if !missing_nodes.is_empty() {
            let fetched = self.fetch_complete_rows(&missing_nodes).await?;

            for (index, row) in missing_indices.into_iter().zip(fetched) {
                let key = (snapshot, nodes[index].clone());
                self.cache.insert(key, row.clone()).await;
                rows[index] = Some(row);
            }
        }

        rows.into_iter()
            .map(|row| row.ok_or(MemgraphBackendError::Graph(RemoteGraphError::MissingNode)))
            .collect()
    }

    async fn fetch_complete_rows(
        &self,
        nodes: &[V],
    ) -> Result<Vec<CachedRow<V>>, MemgraphBackendError> {
        let rows = self
            .execute_nodes_neighbourhood_query(&self.queries.graph_neighbourhoods, nodes)?
            .await?;
        let packed = pack_neighbourhoods(nodes.len(), rows).map_err(map_complete_row_pack_error)?;

        Ok(packed
            .offsets
            .windows(2)
            .map(|window| packed.data[window[0]..window[1]].to_vec())
            .collect())
    }

    async fn fetch_coreset_neighbourhoods(
        &self,
        nodes: &[V],
    ) -> Result<OwnedNeighbourhoods<V, f64>, MemgraphBackendError> {
        let rows = self
            .execute_coreset_neighbourhood_query(&self.queries.coreset_neighbourhoods, nodes)?
            .await?;
        pack_neighbourhoods(nodes.len(), rows).map_err(map_complete_row_pack_error)
    }

    async fn fetch_intersecting_neighbourhoods(
        &self,
        sources: &[V],
        targets: &[V],
    ) -> Result<OwnedNeighbourhoods<V, f64>, MemgraphBackendError> {
        let rows = self
            .execute_intersecting_neighbourhood_query(
                &self.queries.graph_neighbourhoods_intersecting,
                sources,
                targets,
            )?
            .await?;
        pack_neighbourhoods(sources.len(), rows).map_err(map_complete_row_pack_error)
    }

    fn execute_nodes_neighbourhood_query<'a>(
        &'a self,
        cypher: &'a str,
        nodes: &'a [V],
    ) -> Result<
        impl Future<Output = Result<Vec<MemgraphNeighbourhoodRow<V, f64>>, MemgraphBackendError>>
        + Send
        + 'a,
        MemgraphBackendError,
    > {
        let node_values = nodes
            .iter()
            .map(MemgraphNodeId::to_bolt_type)
            .collect::<Result<Vec<_>, _>>()?;
        let query = query(cypher).param("nodes", BoltType::from(node_values));

        Ok(async move {
            let mut stream = self
                .graph
                .execute_read(query)
                .await
                .map_err(MemgraphBackendError::Client)?;
            let mut rows = Vec::new();

            while let Some(row) = stream.next().await.map_err(MemgraphBackendError::Client)? {
                rows.push(decode_neighbourhood_row(row)?);
            }

            Ok(rows)
        })
    }

    fn execute_intersecting_neighbourhood_query<'a>(
        &'a self,
        cypher: &'a str,
        sources: &'a [V],
        targets: &'a [V],
    ) -> Result<
        impl Future<Output = Result<Vec<MemgraphNeighbourhoodRow<V, f64>>, MemgraphBackendError>>
        + Send
        + 'a,
        MemgraphBackendError,
    > {
        let source_values = sources
            .iter()
            .map(MemgraphNodeId::to_bolt_type)
            .collect::<Result<Vec<_>, _>>()?;
        let query = query(cypher)
            .param("sources", BoltType::from(source_values))
            .param("target_lookup", node_lookup_map(targets)?);

        Ok(async move {
            let mut stream = self
                .graph
                .execute_read(query)
                .await
                .map_err(MemgraphBackendError::Client)?;
            let mut rows = Vec::new();

            while let Some(row) = stream.next().await.map_err(MemgraphBackendError::Client)? {
                rows.push(decode_neighbourhood_row(row)?);
            }

            Ok(rows)
        })
    }

    fn execute_coreset_neighbourhood_query<'a>(
        &'a self,
        cypher: &'a str,
        nodes: &'a [V],
    ) -> Result<
        impl Future<Output = Result<Vec<MemgraphNeighbourhoodRow<V, f64>>, MemgraphBackendError>>
        + Send
        + 'a,
        MemgraphBackendError,
    > {
        let node_values = nodes
            .iter()
            .map(MemgraphNodeId::to_bolt_type)
            .collect::<Result<Vec<_>, _>>()?;
        let query = query(cypher)
            .param("nodes", BoltType::from(node_values))
            .param("node_lookup", node_lookup_map(nodes)?);

        Ok(async move {
            let mut stream = self
                .graph
                .execute_read(query)
                .await
                .map_err(MemgraphBackendError::Client)?;
            let mut rows = Vec::new();

            while let Some(row) = stream.next().await.map_err(MemgraphBackendError::Client)? {
                rows.push(decode_neighbourhood_row(row)?);
            }

            Ok(rows)
        })
    }
}

impl<V> RemoteGraphBackend<V, f64> for MemgraphBackend<V>
where
    V: MemgraphNodeId,
{
    type Error = MemgraphBackendError;

    fn graph_neighbourhoods(
        &self,
        snapshot: SnapshotId,
        nodes: Vec<V>,
    ) -> impl Future<Output = Result<OwnedNeighbourhoods<V, f64>, Self::Error>> + Send {
        async move {
            if nodes.is_empty() {
                return Ok(OwnedNeighbourhoods::empty());
            }

            let rows = self.complete_rows(snapshot, nodes).await?;
            Ok(owned_from_rows(rows))
        }
    }

    fn graph_neighbourhoods_intersecting(
        &self,
        snapshot: SnapshotId,
        sources: Vec<V>,
        targets: Vec<V>,
    ) -> impl Future<Output = Result<OwnedNeighbourhoods<V, f64>, Self::Error>> + Send {
        async move {
            if sources.is_empty() {
                return Ok(OwnedNeighbourhoods::empty());
            }

            match self.filter_strategy {
                MemgraphFilterStrategy::FullRowsThenFilterLocally => {
                    let target_set = targets.into_iter().collect::<FxHashSet<_>>();
                    let rows = self.complete_rows(snapshot, sources).await?;
                    Ok(owned_from_filtered_rows(rows, &target_set))
                }
                MemgraphFilterStrategy::PushDownToCypher => {
                    let _ = snapshot;
                    self.fetch_intersecting_neighbourhoods(&sources, &targets)
                        .await
                }
            }
        }
    }

    fn coreset_neighbourhoods(
        &self,
        snapshot: SnapshotId,
        nodes: Vec<V>,
    ) -> impl Future<Output = Result<OwnedNeighbourhoods<V, f64>, Self::Error>> + Send {
        async move {
            if nodes.is_empty() {
                return Ok(OwnedNeighbourhoods::empty());
            }

            match self.filter_strategy {
                MemgraphFilterStrategy::FullRowsThenFilterLocally => {
                    let coreset_set = nodes.iter().cloned().collect::<FxHashSet<_>>();
                    let rows = self.complete_rows(snapshot, nodes).await?;
                    Ok(owned_from_filtered_rows(rows, &coreset_set))
                }
                MemgraphFilterStrategy::PushDownToCypher => {
                    let _ = snapshot;
                    self.fetch_coreset_neighbourhoods(&nodes).await
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemgraphFilterStrategy {
    FullRowsThenFilterLocally,
    PushDownToCypher,
}

impl Default for MemgraphFilterStrategy {
    fn default() -> Self {
        Self::FullRowsThenFilterLocally
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemgraphCacheConfig {
    pub max_capacity: u64,
    pub time_to_live: Option<Duration>,
}

impl Default for MemgraphCacheConfig {
    fn default() -> Self {
        Self {
            max_capacity: 100_000,
            time_to_live: None,
        }
    }
}

impl MemgraphCacheConfig {
    fn build_cache<V>(self) -> MemgraphRowCache<V>
    where
        V: Clone + Eq + Hash + Send + Sync + 'static,
    {
        let builder = Cache::builder().max_capacity(self.max_capacity);
        match self.time_to_live {
            Some(ttl) => builder.time_to_live(ttl).build(),
            None => builder.build(),
        }
    }
}

pub trait MemgraphNodeId: Clone + Eq + Hash + Send + Sync + 'static {
    fn to_bolt_type(&self) -> Result<BoltType, MemgraphDecodeError>;
    fn to_bolt_lookup_key(&self) -> Result<String, MemgraphDecodeError>;
    fn from_bolt_type(value: &BoltType) -> Result<Self, MemgraphDecodeError>;
}

impl MemgraphNodeId for usize {
    fn to_bolt_type(&self) -> Result<BoltType, MemgraphDecodeError> {
        i64::try_from(*self)
            .map(BoltType::from)
            .map_err(MemgraphDecodeError::IntegerOutOfRange)
    }

    fn to_bolt_lookup_key(&self) -> Result<String, MemgraphDecodeError> {
        Ok(self.to_string())
    }

    fn from_bolt_type(value: &BoltType) -> Result<Self, MemgraphDecodeError> {
        let value = decode_i64(value, "node")?;
        usize::try_from(value).map_err(MemgraphDecodeError::IntegerOutOfRange)
    }
}

impl MemgraphNodeId for u64 {
    fn to_bolt_type(&self) -> Result<BoltType, MemgraphDecodeError> {
        i64::try_from(*self)
            .map(BoltType::from)
            .map_err(MemgraphDecodeError::IntegerOutOfRange)
    }

    fn to_bolt_lookup_key(&self) -> Result<String, MemgraphDecodeError> {
        Ok(self.to_string())
    }

    fn from_bolt_type(value: &BoltType) -> Result<Self, MemgraphDecodeError> {
        let value = decode_i64(value, "node")?;
        u64::try_from(value).map_err(MemgraphDecodeError::IntegerOutOfRange)
    }
}

impl MemgraphNodeId for i64 {
    fn to_bolt_type(&self) -> Result<BoltType, MemgraphDecodeError> {
        Ok(BoltType::from(*self))
    }

    fn to_bolt_lookup_key(&self) -> Result<String, MemgraphDecodeError> {
        Ok(self.to_string())
    }

    fn from_bolt_type(value: &BoltType) -> Result<Self, MemgraphDecodeError> {
        decode_i64(value, "node")
    }
}

impl MemgraphNodeId for String {
    fn to_bolt_type(&self) -> Result<BoltType, MemgraphDecodeError> {
        Ok(BoltType::from(self.clone()))
    }

    fn to_bolt_lookup_key(&self) -> Result<String, MemgraphDecodeError> {
        Ok(self.clone())
    }

    fn from_bolt_type(value: &BoltType) -> Result<Self, MemgraphDecodeError> {
        match value {
            BoltType::String(value) => Ok(value.value.clone()),
            _ => Err(MemgraphDecodeError::UnexpectedType { field: "node" }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemgraphNeighbourhoodRow<V, T> {
    pub row_index: usize,
    pub neighbours: Vec<(V, Strict<T>)>,
}

impl<V, T> MemgraphNeighbourhoodRow<V, T> {
    pub fn new(row_index: usize, neighbours: Vec<(V, Strict<T>)>) -> Self {
        Self {
            row_index,
            neighbours,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemgraphRowError {
    DuplicateRow,
    MissingRow,
    RowIndexOutOfBounds,
}

impl fmt::Display for MemgraphRowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateRow => write!(f, "Memgraph returned the same row index more than once"),
            Self::MissingRow => write!(f, "Memgraph did not return one row per requested source"),
            Self::RowIndexOutOfBounds => write!(f, "Memgraph returned an out-of-bounds row index"),
        }
    }
}

impl std::error::Error for MemgraphRowError {}

#[derive(Debug)]
pub enum MemgraphDecodeError {
    MissingColumn(&'static str),
    MissingNeighbourField(&'static str),
    UnexpectedType { field: &'static str },
    IntegerOutOfRange(TryFromIntError),
    InvalidWeight(f64),
}

impl fmt::Display for MemgraphDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingColumn(field) => write!(f, "Memgraph row was missing `{field}`"),
            Self::MissingNeighbourField(field) => {
                write!(f, "Memgraph neighbour entry was missing `{field}`")
            }
            Self::UnexpectedType { field } => {
                write!(f, "Memgraph returned an unexpected type for `{field}`")
            }
            Self::IntegerOutOfRange(_) => {
                write!(f, "Memgraph integer did not fit the node id type")
            }
            Self::InvalidWeight(value) => {
                write!(f, "Memgraph returned an invalid edge weight `{value}`")
            }
        }
    }
}

impl std::error::Error for MemgraphDecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IntegerOutOfRange(err) => Some(err),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum MemgraphBackendError {
    Client(neo4rs::Error),
    Graph(RemoteGraphError),
    Rows(MemgraphRowError),
    Decode(MemgraphDecodeError),
}

impl fmt::Display for MemgraphBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Client(err) => write!(f, "Memgraph client error: {err}"),
            Self::Graph(err) => write!(f, "Memgraph graph error: {err}"),
            Self::Rows(err) => write!(f, "Memgraph row decoding error: {err}"),
            Self::Decode(err) => write!(f, "Memgraph value decoding error: {err}"),
        }
    }
}

impl std::error::Error for MemgraphBackendError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Client(err) => Some(err),
            Self::Graph(err) => Some(err),
            Self::Rows(err) => Some(err),
            Self::Decode(err) => Some(err),
        }
    }
}

impl From<RemoteGraphError> for MemgraphBackendError {
    fn from(value: RemoteGraphError) -> Self {
        Self::Graph(value)
    }
}

impl From<MemgraphRowError> for MemgraphBackendError {
    fn from(value: MemgraphRowError) -> Self {
        Self::Rows(value)
    }
}

impl From<MemgraphDecodeError> for MemgraphBackendError {
    fn from(value: MemgraphDecodeError) -> Self {
        Self::Decode(value)
    }
}

fn owned_from_rows<V>(rows: Vec<CachedRow<V>>) -> OwnedNeighbourhoods<V, f64> {
    let row_count = rows.len();
    let data_len = rows.iter().map(Vec::len).sum();
    let mut data = Vec::with_capacity(data_len);
    let mut offsets = Vec::with_capacity(row_count + 1);
    offsets.push(0);

    for row in rows {
        data.extend(row);
        offsets.push(data.len());
    }

    OwnedNeighbourhoods { data, offsets }
}

fn owned_from_filtered_rows<V>(
    rows: Vec<CachedRow<V>>,
    allowed: &FxHashSet<V>,
) -> OwnedNeighbourhoods<V, f64>
where
    V: Clone + Eq + Hash,
{
    let row_count = rows.len();
    let mut data = Vec::new();
    let mut offsets = Vec::with_capacity(row_count + 1);
    offsets.push(0);

    for row in rows {
        data.extend(
            row.into_iter()
                .filter(|(neighbour, _)| allowed.contains(neighbour)),
        );
        offsets.push(data.len());
    }

    OwnedNeighbourhoods { data, offsets }
}

fn node_lookup_map<V>(nodes: &[V]) -> Result<BoltType, MemgraphDecodeError>
where
    V: MemgraphNodeId,
{
    let mut value = std::collections::HashMap::with_capacity(nodes.len());
    for node in nodes {
        value.insert(
            BoltString::from(node.to_bolt_lookup_key()?),
            BoltType::from(true),
        );
    }
    Ok(BoltType::Map(BoltMap { value }))
}

fn pack_neighbourhoods<V, T>(
    row_count: usize,
    rows: Vec<MemgraphNeighbourhoodRow<V, T>>,
) -> Result<OwnedNeighbourhoods<V, T>, MemgraphBackendError> {
    let mut by_index = (0..row_count).map(|_| None).collect::<Vec<_>>();

    for row in rows {
        let slot = by_index
            .get_mut(row.row_index)
            .ok_or(MemgraphRowError::RowIndexOutOfBounds)?;
        if slot.replace(row.neighbours).is_some() {
            return Err(MemgraphRowError::DuplicateRow.into());
        }
    }

    let mut data = Vec::new();
    let mut offsets = Vec::with_capacity(row_count + 1);
    offsets.push(0);

    for neighbours in by_index {
        let neighbours = neighbours.ok_or(MemgraphRowError::MissingRow)?;
        data.extend(neighbours);
        offsets.push(data.len());
    }

    Ok(OwnedNeighbourhoods { data, offsets })
}

fn map_complete_row_pack_error(err: MemgraphBackendError) -> MemgraphBackendError {
    match err {
        MemgraphBackendError::Rows(MemgraphRowError::MissingRow) => {
            MemgraphBackendError::Graph(RemoteGraphError::MissingNode)
        }
        err => err,
    }
}

fn decode_neighbourhood_row<V>(
    row: neo4rs::Row,
) -> Result<MemgraphNeighbourhoodRow<V, f64>, MemgraphBackendError>
where
    V: MemgraphNodeId,
{
    let row_index = row
        .get::<BoltType>("row_index")
        .map_err(|_| MemgraphDecodeError::MissingColumn("row_index"))
        .and_then(|value| decode_usize(&value, "row_index"))?;
    let neighbours = row
        .get::<BoltType>("neighbours")
        .map_err(|_| MemgraphDecodeError::MissingColumn("neighbours"))
        .and_then(|value| decode_neighbours(&value))?;

    Ok(MemgraphNeighbourhoodRow::new(row_index, neighbours))
}

fn decode_neighbours<V>(value: &BoltType) -> Result<Vec<(V, Strict<f64>)>, MemgraphDecodeError>
where
    V: MemgraphNodeId,
{
    let BoltType::List(neighbours) = value else {
        return Err(MemgraphDecodeError::UnexpectedType {
            field: "neighbours",
        });
    };

    neighbours
        .iter()
        .map(|entry| decode_neighbour_entry(entry))
        .collect()
}

fn decode_neighbour_entry<V>(entry: &BoltType) -> Result<(V, Strict<f64>), MemgraphDecodeError>
where
    V: MemgraphNodeId,
{
    let BoltType::Map(entry) = entry else {
        return Err(MemgraphDecodeError::UnexpectedType { field: "neighbour" });
    };
    let node = map_get(entry, "node")?;
    let weight = map_get(entry, "weight")?;

    Ok((V::from_bolt_type(node)?, decode_weight(weight)?))
}

fn map_get<'a>(map: &'a BoltMap, field: &'static str) -> Result<&'a BoltType, MemgraphDecodeError> {
    map.value
        .get(field)
        .ok_or(MemgraphDecodeError::MissingNeighbourField(field))
}

fn decode_weight(value: &BoltType) -> Result<Strict<f64>, MemgraphDecodeError> {
    let weight = match value {
        BoltType::Float(value) => value.value,
        BoltType::Integer(value) => value.value as f64,
        _ => return Err(MemgraphDecodeError::UnexpectedType { field: "weight" }),
    };

    Strict::<f64>::new(weight).map_err(|_| MemgraphDecodeError::InvalidWeight(weight))
}

fn decode_usize(value: &BoltType, field: &'static str) -> Result<usize, MemgraphDecodeError> {
    let value = decode_i64(value, field)?;
    usize::try_from(value).map_err(MemgraphDecodeError::IntegerOutOfRange)
}

fn decode_i64(value: &BoltType, field: &'static str) -> Result<i64, MemgraphDecodeError> {
    match value {
        BoltType::Integer(value) => Ok(value.value),
        _ => Err(MemgraphDecodeError::UnexpectedType { field }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neo4rs::{BoltFloat, BoltInteger, BoltList, BoltString};

    fn strict(value: f64) -> Strict<f64> {
        Strict::<f64>::new(value).unwrap()
    }

    fn neighbour(node: BoltType, weight: BoltType) -> BoltType {
        BoltType::Map(BoltMap {
            value: [
                (BoltString::from("node"), node),
                (BoltString::from("weight"), weight),
            ]
            .into_iter()
            .collect(),
        })
    }

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
    }

    #[test]
    fn pack_neighbourhoods_preserves_requested_row_order() {
        let rows = vec![
            MemgraphNeighbourhoodRow::new(1, vec![(3, strict(3.0))]),
            MemgraphNeighbourhoodRow::new(0, vec![(2, strict(2.0))]),
        ];

        let packed = pack_neighbourhoods(2, rows).unwrap();

        assert_eq!(packed.offsets, vec![0, 1, 2]);
        assert_eq!(packed.data, vec![(2, strict(2.0)), (3, strict(3.0))]);
    }

    #[test]
    fn pack_neighbourhoods_rejects_duplicate_rows() {
        let rows = vec![
            MemgraphNeighbourhoodRow::<usize, f64>::new(0, vec![]),
            MemgraphNeighbourhoodRow::new(0, vec![]),
        ];

        let err = pack_neighbourhoods(1, rows).unwrap_err();

        assert!(matches!(
            err,
            MemgraphBackendError::Rows(MemgraphRowError::DuplicateRow)
        ));
    }

    #[test]
    fn pack_neighbourhoods_rejects_missing_rows() {
        let rows = vec![MemgraphNeighbourhoodRow::<usize, f64>::new(1, vec![])];

        let err = pack_neighbourhoods(2, rows).unwrap_err();

        assert!(matches!(
            err,
            MemgraphBackendError::Rows(MemgraphRowError::MissingRow)
        ));
    }

    #[test]
    fn filtered_rows_only_include_target_neighbours_and_preserve_source_rows() {
        let rows = vec![
            vec![(2, strict(1.0)), (3, strict(2.0))],
            vec![(1, strict(3.0)), (4, strict(4.0))],
        ];
        let targets = [3, 4].into_iter().collect::<FxHashSet<_>>();

        let packed = owned_from_filtered_rows(rows, &targets);

        assert_eq!(packed.offsets, vec![0, 1, 2]);
        assert_eq!(packed.data, vec![(3, strict(2.0)), (4, strict(4.0))]);
    }

    #[test]
    fn cache_rows_are_scoped_by_snapshot() {
        runtime().block_on(async {
            let cache = MemgraphCacheConfig::default().build_cache::<usize>();
            cache
                .insert((SnapshotId(1), 10), vec![(20, strict(1.0))])
                .await;

            assert_eq!(
                cache.get(&(SnapshotId(1), 10)).await,
                Some(vec![(20, strict(1.0))])
            );
            assert_eq!(cache.get(&(SnapshotId(2), 10)).await, None);
        });
    }

    #[test]
    fn node_id_conversion_round_trips_supported_ids() {
        let usize_value = 42usize;
        let u64_value = 43u64;
        let i64_value = -44i64;
        let string_value = String::from("node-45");

        assert_eq!(
            usize::from_bolt_type(&usize_value.to_bolt_type().unwrap()).unwrap(),
            usize_value
        );
        assert_eq!(
            u64::from_bolt_type(&u64_value.to_bolt_type().unwrap()).unwrap(),
            u64_value
        );
        assert_eq!(
            i64::from_bolt_type(&i64_value.to_bolt_type().unwrap()).unwrap(),
            i64_value
        );
        assert_eq!(
            String::from_bolt_type(&string_value.to_bolt_type().unwrap()).unwrap(),
            string_value
        );
    }

    #[test]
    fn decode_neighbours_rejects_invalid_weights() {
        let neighbours = BoltType::List(BoltList::from(vec![neighbour(
            BoltType::from(1),
            BoltType::from(0.0),
        )]));

        let err = decode_neighbours::<usize>(&neighbours).unwrap_err();

        assert!(matches!(err, MemgraphDecodeError::InvalidWeight(0.0)));
    }

    #[test]
    fn decode_neighbours_rejects_non_finite_weights() {
        let neighbours = BoltType::List(BoltList::from(vec![neighbour(
            BoltType::from(1),
            BoltType::Float(BoltFloat::new(f64::NAN)),
        )]));

        let err = decode_neighbours::<usize>(&neighbours).unwrap_err();

        assert!(matches!(err, MemgraphDecodeError::InvalidWeight(value) if value.is_nan()));
    }

    #[test]
    fn missing_returned_source_rows_map_to_missing_node_at_backend_boundary() {
        let rows = vec![MemgraphNeighbourhoodRow::<usize, f64>::new(1, vec![])];
        let err = pack_neighbourhoods(2, rows).unwrap_err();
        let err = map_complete_row_pack_error(err);

        assert!(matches!(
            err,
            MemgraphBackendError::Graph(RemoteGraphError::MissingNode)
        ));
    }

    #[test]
    #[ignore = "requires MEMGRAPH_URI, MEMGRAPH_USER, and MEMGRAPH_PASSWORD"]
    fn memgraph_backend_smoke_test() {
        let uri = match std::env::var("MEMGRAPH_URI") {
            Ok(uri) => uri,
            Err(_) => return,
        };
        let user = std::env::var("MEMGRAPH_USER").unwrap_or_default();
        let password = std::env::var("MEMGRAPH_PASSWORD").unwrap_or_default();
        let graph = Graph::new(uri, user, password).unwrap();
        let backend = MemgraphBackend::<i64>::new(graph);

        runtime().block_on(async {
            let rows = backend
                .graph_neighbourhoods(SnapshotId(0), Vec::new())
                .await
                .unwrap();
            assert_eq!(rows.offsets, vec![0]);
            assert!(rows.data.is_empty());
        });
    }

    #[test]
    fn bolt_type_helpers_accept_integer_weights() {
        let neighbours = BoltType::List(BoltList::from(vec![neighbour(
            BoltType::from(1),
            BoltType::Integer(BoltInteger::new(2)),
        )]));

        assert_eq!(
            decode_neighbours::<usize>(&neighbours).unwrap(),
            vec![(1, strict(2.0))]
        );
    }
}
