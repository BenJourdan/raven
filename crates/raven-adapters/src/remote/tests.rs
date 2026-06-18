use std::{
    future::{Future, ready},
    sync::{Arc, Mutex},
};

use raven_core::{
    DynamicClusteringAlg, GraphOracle,
    alg::DynamicClustering,
    error::OracleError,
    types::{AlgType, Strict},
};

use super::{
    OwnedNeighbourhoods, RemoteGraphBackend, RemoteGraphClient, RemoteOracle,
    RemoteSnapshotNodeOps, SnapshotId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TestRemoteError {
    GraphFailed,
    CoresetFailed,
}

#[derive(Debug, Clone, PartialEq)]
enum Request {
    Graph {
        snapshot: SnapshotId,
        nodes: Vec<usize>,
    },
    Intersecting {
        snapshot: SnapshotId,
        sources: Vec<usize>,
        targets: Vec<usize>,
    },
    Coreset {
        snapshot: SnapshotId,
        nodes: Vec<usize>,
    },
}

#[derive(Default)]
struct FakeRemoteBackend {
    requests: Mutex<Vec<Request>>,
    fail_graph: bool,
    fail_coreset: bool,
}

impl FakeRemoteBackend {
    fn requests(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }
}

impl RemoteGraphBackend<usize, f64> for FakeRemoteBackend {
    type Error = TestRemoteError;

    fn graph_neighbourhoods(
        &self,
        snapshot: SnapshotId,
        nodes: Vec<usize>,
    ) -> impl Future<Output = Result<OwnedNeighbourhoods<usize, f64>, Self::Error>> + Send {
        self.requests
            .lock()
            .unwrap()
            .push(Request::Graph { snapshot, nodes });

        if self.fail_graph {
            return ready(Err(TestRemoteError::GraphFailed));
        }

        ready(Ok(OwnedNeighbourhoods {
            data: vec![(2, strict(1.5)), (3, strict(2.5)), (1, strict(1.5))],
            offsets: vec![0, 2, 3],
        }))
    }

    fn graph_neighbourhoods_intersecting(
        &self,
        snapshot: SnapshotId,
        sources: Vec<usize>,
        targets: Vec<usize>,
    ) -> impl Future<Output = Result<OwnedNeighbourhoods<usize, f64>, Self::Error>> + Send {
        self.requests.lock().unwrap().push(Request::Intersecting {
            snapshot,
            sources,
            targets,
        });

        if self.fail_graph {
            return ready(Err(TestRemoteError::GraphFailed));
        }

        ready(Ok(OwnedNeighbourhoods {
            data: vec![(3, strict(2.5))],
            offsets: vec![0, 1, 1],
        }))
    }

    fn coreset_neighbourhoods(
        &self,
        snapshot: SnapshotId,
        nodes: Vec<usize>,
    ) -> impl Future<Output = Result<OwnedNeighbourhoods<usize, f64>, Self::Error>> + Send {
        self.requests
            .lock()
            .unwrap()
            .push(Request::Coreset { snapshot, nodes });

        if self.fail_coreset {
            return ready(Err(TestRemoteError::CoresetFailed));
        }

        ready(Ok(OwnedNeighbourhoods {
            data: vec![(3, strict(4.0))],
            offsets: vec![0, 1],
        }))
    }
}

fn strict(value: f64) -> Strict<f64> {
    Strict::<f64>::new(value).unwrap()
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
}

fn dummy_cluster_alg() -> AlgType<f64> {
    Arc::new(|_, _| (Vec::new(), 0))
}

#[test]
fn remote_snapshot_node_ops_exposes_snapshot_and_core_node_ops() {
    let snapshot_node_ops =
        RemoteSnapshotNodeOps::new(SnapshotId(5), vec![(1, Some(strict(3.0))), (2, None)]);

    assert_eq!(snapshot_node_ops.snapshot, SnapshotId(5));
    assert_eq!(
        snapshot_node_ops.node_ops(),
        &[(1, Some(strict(3.0))), (2, None)]
    );
    assert_eq!(
        snapshot_node_ops.into_node_ops(),
        vec![(1, Some(strict(3.0))), (2, None)]
    );
}

#[test]
fn remote_snapshot_node_ops_can_update_the_core_tree() {
    let snapshot_node_ops = RemoteSnapshotNodeOps::new(
        SnapshotId(13),
        vec![(10, Some(strict(4.0))), (20, Some(strict(4.0)))],
    );
    let mut clustering = DynamicClustering::<2, usize, f64>::new(dummy_cluster_alg());

    clustering
        .apply_node_ops(snapshot_node_ops.node_ops())
        .unwrap();

    assert_eq!(snapshot_node_ops.snapshot, SnapshotId(13));
    assert_eq!(clustering.num_leaves(), 2);
}

#[test]
fn remote_snapshot_node_ops_snapshot_can_be_used_to_create_remote_oracles() {
    let backend = Arc::new(FakeRemoteBackend::default());
    let runtime = runtime();
    let client = RemoteGraphClient::new(backend.clone(), runtime.handle().clone());
    let snapshot_node_ops = RemoteSnapshotNodeOps::new(
        SnapshotId(21),
        vec![(1, Some(strict(1.0))), (2, Some(strict(1.0)))],
    );
    let snapshot = client.snapshot(snapshot_node_ops.snapshot);
    let mut oracle = snapshot.oracle::<usize, f64>();

    oracle.graph_neighbourhoods(&[1, 2]).unwrap();

    assert_eq!(
        backend.requests(),
        vec![Request::Graph {
            snapshot: SnapshotId(21),
            nodes: vec![1, 2],
        }]
    );
}

#[test]
fn remote_graph_client_creates_oracles_for_snapshots() {
    let backend = Arc::new(FakeRemoteBackend::default());
    let runtime = runtime();
    let client = RemoteGraphClient::new(backend.clone(), runtime.handle().clone());
    let mut oracle = client.oracle::<usize, f64>(SnapshotId(17));

    assert_eq!(oracle.snapshot(), SnapshotId(17));
    oracle.graph_neighbourhoods(&[1]).unwrap();

    assert_eq!(
        backend.requests(),
        vec![Request::Graph {
            snapshot: SnapshotId(17),
            nodes: vec![1],
        }]
    );
}

#[test]
fn remote_graph_snapshot_creates_one_oracle_per_trial() {
    let backend = Arc::new(FakeRemoteBackend::default());
    let runtime = runtime();
    let client = RemoteGraphClient::new(backend.clone(), runtime.handle().clone());
    let snapshot = client.snapshot(SnapshotId(23));
    let mut oracles = snapshot.oracles::<usize, f64>(2);

    assert_eq!(snapshot.id(), SnapshotId(23));
    assert_eq!(oracles.len(), 2);
    assert!(
        oracles
            .iter()
            .all(|oracle| oracle.snapshot() == SnapshotId(23))
    );

    oracles[0].graph_neighbourhoods(&[1]).unwrap();
    oracles[1].coreset_neighbourhoods(&[2]).unwrap();

    assert_eq!(
        backend.requests(),
        vec![
            Request::Graph {
                snapshot: SnapshotId(23),
                nodes: vec![1],
            },
            Request::Coreset {
                snapshot: SnapshotId(23),
                nodes: vec![2],
            },
        ]
    );
}

#[test]
fn remote_graph_snapshot_with_oracles_hides_mut_ref_plumbing() {
    let backend = Arc::new(FakeRemoteBackend::default());
    let runtime = runtime();
    let client = RemoteGraphClient::new(backend.clone(), runtime.handle().clone());
    let snapshot = client.snapshot(SnapshotId(29));

    let all_snapshots_match = snapshot.with_oracles::<usize, f64, _>(2, |oracles| {
        assert_eq!(oracles.len(), 2);
        assert!(
            oracles
                .iter()
                .all(|oracle| oracle.snapshot() == SnapshotId(29))
        );

        oracles[0].graph_neighbourhoods(&[1]).unwrap();
        oracles[1].coreset_neighbourhoods(&[2]).unwrap();

        true
    });

    assert!(all_snapshots_match);
    assert_eq!(
        backend.requests(),
        vec![
            Request::Graph {
                snapshot: SnapshotId(29),
                nodes: vec![1],
            },
            Request::Coreset {
                snapshot: SnapshotId(29),
                nodes: vec![2],
            },
        ]
    );
}

#[test]
fn remote_graph_client_with_oracles_uses_the_requested_snapshot() {
    let backend = Arc::new(FakeRemoteBackend::default());
    let runtime = runtime();
    let client = RemoteGraphClient::new(backend.clone(), runtime.handle().clone());

    let queried_snapshot = client.with_oracles::<usize, f64, _>(SnapshotId(31), 1, |oracles| {
        assert_eq!(oracles.len(), 1);
        oracles[0].graph_neighbourhoods(&[1]).unwrap();
        oracles[0].snapshot()
    });

    assert_eq!(queried_snapshot, SnapshotId(31));
    assert_eq!(
        backend.requests(),
        vec![Request::Graph {
            snapshot: SnapshotId(31),
            nodes: vec![1],
        }]
    );
}

#[test]
fn remote_oracle_returns_graph_rows_from_async_backend() {
    let backend = Arc::new(FakeRemoteBackend::default());
    let runtime = runtime();
    let mut oracle = RemoteOracle::new(backend.clone(), runtime.handle().clone(), SnapshotId(7));

    let rows = oracle.graph_neighbourhoods(&[1, 2]).unwrap();

    assert_eq!(rows.len(), 2);
    assert_eq!(rows.row(0).unwrap(), &[(2, strict(1.5)), (3, strict(2.5))]);
    assert_eq!(rows.row(1).unwrap(), &[(1, strict(1.5))]);
    assert_eq!(
        backend.requests(),
        vec![Request::Graph {
            snapshot: SnapshotId(7),
            nodes: vec![1, 2],
        }]
    );
}

#[test]
fn remote_oracle_returns_intersecting_rows_from_async_backend() {
    let backend = Arc::new(FakeRemoteBackend::default());
    let runtime = runtime();
    let mut oracle = RemoteOracle::new(backend.clone(), runtime.handle().clone(), SnapshotId(9));

    let rows = oracle
        .graph_neighbourhoods_intersecting(&[1, 2], &[3, 4])
        .unwrap();

    assert_eq!(rows.len(), 2);
    assert_eq!(rows.row(0).unwrap(), &[(3, strict(2.5))]);
    assert_eq!(rows.row(1).unwrap(), &[]);
    assert_eq!(
        backend.requests(),
        vec![Request::Intersecting {
            snapshot: SnapshotId(9),
            sources: vec![1, 2],
            targets: vec![3, 4],
        }]
    );
}

#[test]
fn remote_oracle_maps_graph_backend_errors_to_graph_errors() {
    let backend = Arc::new(FakeRemoteBackend {
        fail_graph: true,
        ..FakeRemoteBackend::default()
    });
    let runtime = runtime();
    let mut oracle = RemoteOracle::new(backend, runtime.handle().clone(), SnapshotId(1));

    let err = oracle.graph_neighbourhoods(&[1]).unwrap_err();

    assert!(matches!(
        err,
        OracleError::GraphError(TestRemoteError::GraphFailed)
    ));
}

#[test]
fn remote_oracle_maps_coreset_backend_errors_to_coreset_errors() {
    let backend = Arc::new(FakeRemoteBackend {
        fail_coreset: true,
        ..FakeRemoteBackend::default()
    });
    let runtime = runtime();
    let mut oracle = RemoteOracle::new(backend, runtime.handle().clone(), SnapshotId(1));

    let err = oracle.coreset_neighbourhoods(&[1]).unwrap_err();

    assert!(matches!(
        err,
        OracleError::CoresetError(TestRemoteError::CoresetFailed)
    ));
}

#[test]
fn remote_oracle_returns_coreset_rows_from_async_backend() {
    let backend = Arc::new(FakeRemoteBackend::default());
    let runtime = runtime();
    let mut oracle = RemoteOracle::new(backend.clone(), runtime.handle().clone(), SnapshotId(11));

    let rows = oracle.coreset_neighbourhoods(&[1]).unwrap();

    assert_eq!(rows.len(), 1);
    assert_eq!(rows.row(0).unwrap(), &[(3, strict(4.0))]);
    assert_eq!(
        backend.requests(),
        vec![Request::Coreset {
            snapshot: SnapshotId(11),
            nodes: vec![1],
        }]
    );
}
