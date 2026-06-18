#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemgraphQueries {
    pub graph_neighbourhoods: String,
    pub graph_neighbourhoods_intersecting: String,
    pub coreset_neighbourhoods: String,
}

impl Default for MemgraphQueries {
    fn default() -> Self {
        Self {
            graph_neighbourhoods: GRAPH_NEIGHBOURHOODS_QUERY.to_string(),
            graph_neighbourhoods_intersecting: GRAPH_NEIGHBOURHOODS_INTERSECTING_QUERY.to_string(),
            coreset_neighbourhoods: CORESET_NEIGHBOURHOODS_QUERY.to_string(),
        }
    }
}

/// Sketch query for complete graph rows.
///
/// Expected parameters:
/// - `nodes`: source node ids in row order
///
/// Expected returned columns:
/// - `row_index`: source row index
/// - `neighbours`: list of `{node, weight}` maps
pub const GRAPH_NEIGHBOURHOODS_QUERY: &str = r#"
WITH $nodes AS nodes
UNWIND range(0, size(nodes) - 1) AS row_index
WITH row_index, nodes[row_index] AS source_id
MATCH (source:RavenNode {id: source_id})
OPTIONAL MATCH (source)-[edge:RAVEN_EDGE]-(target:RavenNode)
WITH row_index,
     collect(
         CASE
             WHEN target IS NULL THEN null
             ELSE {node: target.id, weight: edge.weight}
         END
     ) AS neighbours
RETURN row_index, [entry IN neighbours WHERE entry IS NOT NULL] AS neighbours
ORDER BY row_index
"#;

/// Sketch query for graph rows filtered to a target set.
///
/// Expected parameters:
/// - `sources`: source node ids in row order
/// - `target_lookup`: map from `toString(node_id)` to `true` for allowed neighbours
pub const GRAPH_NEIGHBOURHOODS_INTERSECTING_QUERY: &str = r#"
WITH $sources AS sources, $target_lookup AS target_lookup
UNWIND range(0, size(sources) - 1) AS row_index
WITH row_index, sources[row_index] AS source_id, target_lookup
MATCH (source:RavenNode {id: source_id})
OPTIONAL MATCH (source)-[edge:RAVEN_EDGE]-(target:RavenNode)
WITH row_index,
     collect(
         CASE
             WHEN target IS NOT NULL AND target_lookup[toString(target.id)] IS NOT NULL
             THEN {node: target.id, weight: edge.weight}
             ELSE null
         END
     ) AS neighbours
RETURN row_index, [entry IN neighbours WHERE entry IS NOT NULL] AS neighbours
ORDER BY row_index
"#;

/// Sketch query for coreset-induced rows.
///
/// Expected parameters:
/// - `nodes`: complete coreset node ids in row order
/// - `node_lookup`: map from `toString(node_id)` to `true` for coreset nodes
pub const CORESET_NEIGHBOURHOODS_QUERY: &str = r#"
WITH $nodes AS nodes, $node_lookup AS node_lookup
UNWIND range(0, size(nodes) - 1) AS row_index
WITH row_index, nodes[row_index] AS source_id, node_lookup
MATCH (source:RavenNode {id: source_id})
OPTIONAL MATCH (source)-[edge:RAVEN_EDGE]-(target:RavenNode)
WITH row_index,
     collect(
         CASE
             WHEN target IS NOT NULL AND node_lookup[toString(target.id)] IS NOT NULL
             THEN {node: target.id, weight: edge.weight}
             ELSE null
         END
     ) AS neighbours
RETURN row_index, [entry IN neighbours WHERE entry IS NOT NULL] AS neighbours
ORDER BY row_index
"#;
