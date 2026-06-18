#![cfg(feature = "remote-playground")]

use std::{thread, time::Duration};

use neo4rs::{Graph, query};
use raven_adapters::remote::{
    MemgraphBackend, MemgraphBackendError, RemoteGraphBackend, RemoteGraphError, SnapshotId,
};
use raven_core::types::Strict;
use testcontainers::{
    ContainerAsync, GenericImage,
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
};

const SNAPSHOT: SnapshotId = SnapshotId(42);

#[test]
fn memgraph_backend_fetches_and_filters_real_rows() {
    runtime().block_on(async {
        let fixture = start_seeded_memgraph().await;
        let backend = MemgraphBackend::<i64>::new(fixture.graph.clone());

        let rows = backend
            .graph_neighbourhoods(SNAPSHOT, vec![1, 2, 3])
            .await
            .unwrap();

        assert_eq!(
            sorted_rows(&rows),
            vec![
                vec![(2, strict(3.0)), (3, strict(5.0))],
                vec![(1, strict(3.0)), (3, strict(7.0))],
                vec![(1, strict(5.0)), (2, strict(7.0))],
            ]
        );

        let intersecting = backend
            .graph_neighbourhoods_intersecting(SNAPSHOT, vec![1, 2, 3], vec![3])
            .await
            .unwrap();

        assert_eq!(
            sorted_rows(&intersecting),
            vec![vec![(3, strict(5.0))], vec![(3, strict(7.0))], vec![]]
        );

        let coreset = backend
            .coreset_neighbourhoods(SNAPSHOT, vec![1, 3])
            .await
            .unwrap();

        assert_eq!(
            sorted_rows(&coreset),
            vec![vec![(3, strict(5.0))], vec![(1, strict(5.0))]]
        );
    });
}

#[test]
fn memgraph_backend_reports_missing_source_nodes() {
    runtime().block_on(async {
        let fixture = start_seeded_memgraph().await;
        let backend = MemgraphBackend::<i64>::new(fixture.graph.clone());

        let err = backend
            .graph_neighbourhoods(SNAPSHOT, vec![1, 99])
            .await
            .unwrap_err();

        assert!(matches!(
            err,
            MemgraphBackendError::Graph(RemoteGraphError::MissingNode)
        ));
    });
}

struct MemgraphFixture {
    graph: Graph,
    _container: ContainerAsync<GenericImage>,
}

async fn start_seeded_memgraph() -> MemgraphFixture {
    let container = GenericImage::new("memgraph/memgraph", "latest")
        .with_exposed_port(7687.tcp())
        .with_wait_for(WaitFor::seconds(2))
        .start()
        .await
        .unwrap();
    let port = container.get_host_port_ipv4(7687.tcp()).await.unwrap();
    let graph = Graph::new(format!("bolt://127.0.0.1:{port}"), "", "").unwrap();

    wait_for_bolt(&graph).await;
    seed_graph(&graph).await;

    MemgraphFixture {
        graph,
        _container: container,
    }
}

async fn wait_for_bolt(graph: &Graph) {
    let mut last_error = None;

    for _ in 0..30 {
        match graph.run(query("RETURN 1")).await {
            Ok(()) => return,
            Err(err) => {
                last_error = Some(err);
                thread::sleep(Duration::from_millis(250));
            }
        }
    }

    panic!("Memgraph Bolt endpoint did not become ready: {last_error:?}");
}

async fn seed_graph(graph: &Graph) {
    graph
        .run(query(
            r#"
            CREATE (:RavenNode {id: 1})
            CREATE (:RavenNode {id: 2})
            CREATE (:RavenNode {id: 3})
            "#,
        ))
        .await
        .unwrap();

    graph
        .run(query(
            r#"
            MATCH (one:RavenNode {id: 1})
            MATCH (two:RavenNode {id: 2})
            MATCH (three:RavenNode {id: 3})
            CREATE (one)-[:RAVEN_EDGE {weight: 3.0}]->(two)
            CREATE (one)-[:RAVEN_EDGE {weight: 5.0}]->(three)
            CREATE (two)-[:RAVEN_EDGE {weight: 7.0}]->(three)
            "#,
        ))
        .await
        .unwrap();
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .enable_time()
        .build()
        .unwrap()
}

fn strict(value: f64) -> Strict<f64> {
    Strict::<f64>::new(value).unwrap()
}

fn sorted_rows(
    rows: &raven_adapters::remote::OwnedNeighbourhoods<i64, f64>,
) -> Vec<Vec<(i64, Strict<f64>)>> {
    rows.offsets
        .windows(2)
        .map(|window| {
            let mut row = rows.data[window[0]..window[1]].to_vec();
            row.sort_by_key(|(node, _)| *node);
            row
        })
        .collect()
}
