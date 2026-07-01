from __future__ import annotations

"""Python API for Raven's in-memory dynamic graph clustering index.

The main entry point is :class:`Raven`. It owns graph storage and the dynamic
clustering index, so callers can update edges and query clusters through one
object.

Example:
    >>> import numpy as np
    >>> import raven
    >>> index = raven.Raven(2, coreset_size=3, sampling_seeds=2, rng_seed=7)
    >>> index.update_edges([(1, 2, 1.0), (2, 3, 1.0), (10, 11, 1.0)])
    EdgeUpdateStats(...)
    >>> result = index.query([1, 2, 3])
    >>> len(result.labels)
    3
    >>> consensus = index.query_consensus([1, 2, 3])
    >>> consensus.score_pairs(np.array([[1, 2], [1, 3]], dtype=np.uintp))
    array([...])
"""

from typing import Iterable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from ._raven import (
    ConsensusResult,
    EdgeUpdateStats,
    QueryResult,
    Raven as _NativeRaven,
    RavenError,
)

TrialWeighting = Literal["uniform", "inverse_score", "score_softmax"]
PairArray = NDArray[np.uintp]


class Raven:
    """In-memory Raven index.

    Parameters mirror the Rust ``RavenConfig`` defaults. Node IDs must be
    non-negative Python integers representable as Rust ``usize``. Edge weights
    must be positive finite floats.
    """

    def __init__(
        self,
        num_clusters: int,
        *,
        sigma: float = 1000.0,
        coreset_size: int = 8192,
        sampling_seeds: int | None = None,
        num_trials: int = 1,
        rng_seed: int | None = None,
        node_capacity: int = 1024,
        expected_edges_per_node: int = 16,
        degree_rebuild_threshold: int = 4096,
    ) -> None:
        self._inner = _NativeRaven(
            num_clusters,
            sigma=sigma,
            coreset_size=coreset_size,
            sampling_seeds=sampling_seeds,
            num_trials=num_trials,
            rng_seed=rng_seed,
            node_capacity=node_capacity,
            expected_edges_per_node=expected_edges_per_node,
            degree_rebuild_threshold=degree_rebuild_threshold,
        )

    def update_edge(self, u: int, v: int, weight: float) -> None:
        """Insert or update one undirected weighted edge."""
        self._inner.update_edge(u, v, weight)

    def delete_edge(self, u: int, v: int) -> bool:
        """Delete one undirected edge, returning ``False`` if it was absent."""
        return self._inner.delete_edge(u, v)

    def update_edges(
        self, updates: Iterable[tuple[int, int, float | None]]
    ) -> EdgeUpdateStats:
        """Apply a batch of updates.

        Each update is ``(u, v, weight)``. Use ``weight=None`` to delete an edge.
        """
        return self._inner.update_edges(list(updates))

    def flush(self) -> None:
        """Flush pending node-degree changes into Raven's dynamic state.

        Queries call this automatically. Manual flushing is useful when callers
        want to time ingestion and index maintenance separately.
        """
        self._inner.flush()

    def query(self, nodes: Sequence[int]) -> QueryResult:
        """Cluster ``nodes`` and return the winner trial."""
        return self._inner.query(list(nodes))

    def query_all_trials(self, nodes: Sequence[int]) -> list[QueryResult]:
        """Cluster ``nodes`` and return every trial partition."""
        return self._inner.query_all_trials(list(nodes))

    def query_consensus(
        self,
        nodes: Sequence[int],
        *,
        trial_weighting: TrialWeighting = "score_softmax",
        temperature: float | Literal["auto"] = "auto",
    ) -> ConsensusResult:
        """Cluster ``nodes`` and return a reusable lazy consensus object.

        The returned object stores the trial partitions and can score node pairs
        without materialising a full similarity matrix.
        """
        return self._inner.query_consensus(
            list(nodes),
            trial_weighting=trial_weighting,
            temperature=_temperature_arg(temperature),
        )

    def score_pair(
        self,
        u: int,
        v: int,
        *,
        trial_weighting: TrialWeighting = "score_softmax",
        temperature: float | Literal["auto"] | None = "auto",
    ) -> float:
        """Score one pair by querying the pair's nodes directly."""
        return self._inner.score_pair(
            u,
            v,
            trial_weighting=trial_weighting,
            temperature=_temperature_arg(temperature),
        )

    def score_pairs(
        self,
        pairs: Iterable[tuple[int, int]],
        *,
        trial_weighting: TrialWeighting = "score_softmax",
        temperature: float | Literal["auto"] | None = "auto",
    ) -> PairArray:
        """Score pairs by querying the unique nodes found in ``pairs``.

        For high-throughput scoring, pass a contiguous ``Nx2`` NumPy array with
        ``dtype=np.uintp``. Lists of ``(u, v)`` tuples also work but require more
        Python-side conversion.
        """
        return self._inner.score_pairs(
            _pairs_arg(pairs),
            trial_weighting=trial_weighting,
            temperature=_temperature_arg(temperature),
        )

    def contains_node(self, node: int) -> bool:
        """Return whether ``node`` is currently live in the graph."""
        return self._inner.contains_node(node)

    def live_node_count(self) -> int:
        """Return the number of currently live graph nodes."""
        return self._inner.live_node_count()

    def live_nodes(self) -> list[int]:
        """Return live graph nodes sorted ascending."""
        return self._inner.live_nodes()

    def clear(self) -> None:
        """Reset the graph and index state while preserving configuration."""
        self._inner.clear()


def _temperature_arg(temperature: float | Literal["auto"] | None) -> float | None:
    if temperature is None or temperature == "auto":
        return None
    return float(temperature)


def _pairs_arg(pairs: Iterable[tuple[int, int]] | np.ndarray) -> PairArray:
    if isinstance(pairs, np.ndarray):
        pair_array = np.asarray(pairs, dtype=np.uintp)
        if pair_array.size == 0:
            return np.empty((0, 2), dtype=np.uintp)
        return np.ascontiguousarray(pair_array.reshape(-1, 2))

    pair_list = list(pairs)
    if not pair_list:
        return np.empty((0, 2), dtype=np.uintp)
    return np.ascontiguousarray(np.asarray(pair_list, dtype=np.uintp).reshape(-1, 2))


__all__ = [
    "ConsensusResult",
    "EdgeUpdateStats",
    "PairArray",
    "QueryResult",
    "Raven",
    "RavenError",
    "TrialWeighting",
]
