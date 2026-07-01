from __future__ import annotations

from typing import Iterable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray

TrialWeighting = Literal["uniform", "inverse_score", "score_softmax"]
PairArray = NDArray[np.uintp]


class RavenError(Exception): ...


class QueryResult:
    @property
    def nodes(self) -> list[int]: ...
    @property
    def labels(self) -> list[int]: ...
    @property
    def scores(self) -> list[float] | None: ...
    @property
    def trial_index(self) -> int: ...
    @property
    def num_clusters(self) -> int: ...


class EdgeUpdateStats:
    @property
    def total(self) -> int: ...
    @property
    def set(self) -> int: ...
    @property
    def deleted(self) -> int: ...
    @property
    def missing_deletes(self) -> int: ...


class ConsensusResult:
    @property
    def nodes(self) -> list[int]: ...
    @property
    def labels(self) -> list[list[int]]: ...
    @property
    def trial_weights(self) -> list[float]: ...
    @property
    def trial_scores(self) -> list[float]: ...
    @property
    def trial_indices(self) -> list[int]: ...
    @property
    def num_clusters(self) -> list[int]: ...
    @property
    def num_trials(self) -> int: ...
    @property
    def num_nodes(self) -> int: ...
    def score_pair(self, u: int, v: int) -> float: ...
    def score_pairs(
        self,
        pairs: Sequence[tuple[int, int]] | PairArray,
    ) -> PairArray: ...
    def score_matrix(self, nodes: Sequence[int] | None = None) -> list[list[float]]: ...
    def threshold_pairs(
        self,
        pairs: Sequence[tuple[int, int]],
        *,
        threshold: float = 0.8,
    ) -> list[tuple[int, int, float]]: ...
    def connected_components(
        self,
        pairs: Sequence[tuple[int, int]],
        *,
        threshold: float = 0.8,
        include_singletons: bool = True,
    ) -> list[list[int]]: ...


class Raven:
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
    ) -> None: ...
    def update_edge(self, u: int, v: int, weight: float) -> None: ...
    def delete_edge(self, u: int, v: int) -> bool: ...
    def update_edges(
        self,
        updates: Iterable[tuple[int, int, float | None]],
    ) -> EdgeUpdateStats: ...
    def flush(self) -> None: ...
    def query(self, nodes: Sequence[int]) -> QueryResult: ...
    def query_all_trials(self, nodes: Sequence[int]) -> list[QueryResult]: ...
    def query_consensus(
        self,
        nodes: Sequence[int],
        *,
        trial_weighting: TrialWeighting = "score_softmax",
        temperature: float | Literal["auto"] = "auto",
    ) -> ConsensusResult: ...
    def score_pair(
        self,
        u: int,
        v: int,
        *,
        trial_weighting: TrialWeighting = "score_softmax",
        temperature: float | Literal["auto"] | None = "auto",
    ) -> float: ...
    def score_pairs(
        self,
        pairs: Iterable[tuple[int, int]] | PairArray,
        *,
        trial_weighting: TrialWeighting = "score_softmax",
        temperature: float | Literal["auto"] | None = "auto",
    ) -> PairArray: ...
    def contains_node(self, node: int) -> bool: ...
    def live_node_count(self) -> int: ...
    def live_nodes(self) -> list[int]: ...
    def clear(self) -> None: ...
