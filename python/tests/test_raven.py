import pytest

import raven


def make_raven():
    return raven.Raven(
        2,
        coreset_size=3,
        sampling_seeds=2,
        rng_seed=42,
        node_capacity=16,
        expected_edges_per_node=4,
    )


def populate(index):
    index.update_edge(1, 2, 1.0)
    index.update_edge(2, 3, 1.0)
    index.update_edge(3, 4, 1.0)


def test_update_delete_live_nodes_and_clear():
    index = make_raven()

    assert index.live_node_count() == 0
    assert not index.contains_node(1)

    index.update_edge(1, 2, 1.0)
    index.update_edge(2, 3, 2.0)
    assert index.contains_node(1)
    assert index.live_node_count() == 3
    assert index.live_nodes() == [1, 2, 3]

    assert index.delete_edge(1, 2) is True
    assert index.delete_edge(1, 2) is False
    index.flush()
    assert index.live_nodes() == [2, 3]

    index.clear()
    assert index.live_node_count() == 0
    assert index.live_nodes() == []
    index.update_edge(10, 11, 1.0)
    assert index.live_nodes() == [10, 11]


def test_batch_update_edges_reports_stats():
    index = make_raven()

    stats = index.update_edges(
        [
            (1, 2, 1.0),
            (2, 3, 1.0),
            (1, 2, None),
            (1, 2, None),
        ]
    )

    assert stats.total == 4
    assert stats.set == 2
    assert stats.deleted == 1
    assert stats.missing_deletes == 1


def test_query_and_query_all_trials():
    index = make_raven()
    populate(index)

    result = index.query([3, 1, 2])
    assert result.nodes == [3, 1, 2]
    assert len(result.labels) == 3
    assert result.trial_index == 0
    assert result.num_clusters > 0
    assert result.scores is None or len(result.scores) == 3

    all_trials = index.query_all_trials([4, 2])
    assert len(all_trials) == 1
    assert all_trials[0].nodes == [4, 2]
    assert len(all_trials[0].labels) == 2


def test_query_consensus_lazy_pair_probabilities():
    index = raven.Raven(
        2,
        coreset_size=3,
        sampling_seeds=2,
        num_trials=3,
        rng_seed=42,
        node_capacity=16,
        expected_edges_per_node=4,
    )
    populate(index)

    consensus = index.query_consensus([1, 2, 3], trial_weighting="uniform")

    assert consensus.nodes == [1, 2, 3]
    assert consensus.num_trials == 3
    assert consensus.num_nodes == 3
    assert consensus.labels.shape == (3, 3)
    assert consensus.trial_weights.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    probability = consensus.score_pair(1, 2)
    assert 0.0 <= probability <= 1.0

    probabilities = consensus.score_pairs([(1, 2), (1, 3)])
    assert probabilities.shape == (2,)
    assert all(0.0 <= value <= 1.0 for value in probabilities)

    matrix = consensus.score_matrix([1, 2])
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[1, 1] == pytest.approx(1.0)

    assert len(consensus.threshold_pairs([(1, 2), (2, 3)], threshold=0.0)) == 2
    assert consensus.connected_components(
        [(1, 2), (2, 3)],
        threshold=0.0,
        include_singletons=False,
    ) == [[1, 2, 3]]

    with pytest.raises(KeyError):
        consensus.score_pair(1, 99)


def test_errors_are_raven_errors():
    index = make_raven()

    with pytest.raises(raven.RavenError):
        index.update_edge(1, 2, float("nan"))

    with pytest.raises(raven.RavenError):
        index.query([1, 2])
