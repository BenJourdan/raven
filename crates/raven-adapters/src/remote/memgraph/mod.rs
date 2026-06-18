//! Memgraph backend using Bolt/Cypher via `neo4rs`.
//!
//! The backend caches complete source-node neighbourhood rows per Raven
//! snapshot. Filtered oracle calls can either reuse those full rows and apply
//! the target filter locally, or push the filter into Cypher.

mod backend;
mod queries;

pub use backend::{
    MemgraphBackend, MemgraphBackendError, MemgraphCacheConfig, MemgraphDecodeError,
    MemgraphFilterStrategy, MemgraphNeighbourhoodRow, MemgraphNodeId, MemgraphRowError,
};
pub use queries::{
    CORESET_NEIGHBOURHOODS_QUERY, GRAPH_NEIGHBOURHOODS_INTERSECTING_QUERY,
    GRAPH_NEIGHBOURHOODS_QUERY, MemgraphQueries,
};
