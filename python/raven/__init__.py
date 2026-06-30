from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np

from ._raven import EdgeUpdateStats, QueryResult, Raven as _NativeRaven, RavenError

TrialWeighting = Literal["uniform", "inverse_score", "score_softmax"]


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
        self._inner.update_edge(u, v, weight)

    def delete_edge(self, u: int, v: int) -> bool:
        return self._inner.delete_edge(u, v)

    def update_edges(
        self, updates: Iterable[tuple[int, int, float | None]]
    ) -> EdgeUpdateStats:
        return self._inner.update_edges(list(updates))

    def flush(self) -> None:
        self._inner.flush()

    def query(self, nodes: Sequence[int]) -> QueryResult:
        return self._inner.query(list(nodes))

    def query_all_trials(self, nodes: Sequence[int]) -> list[QueryResult]:
        return self._inner.query_all_trials(list(nodes))

    def query_consensus(
        self,
        nodes: Sequence[int],
        *,
        trial_weighting: TrialWeighting = "score_softmax",
        temperature: float | Literal["auto"] = "auto",
    ) -> ConsensusResult:
        trials = self.query_all_trials(nodes)
        return ConsensusResult.from_trials(
            nodes,
            trials,
            trial_weighting=trial_weighting,
            temperature=temperature,
        )

    def contains_node(self, node: int) -> bool:
        return self._inner.contains_node(node)

    def live_node_count(self) -> int:
        return self._inner.live_node_count()

    def live_nodes(self) -> list[int]:
        return self._inner.live_nodes()

    def clear(self) -> None:
        self._inner.clear()


@dataclass(frozen=True)
class ConsensusResult:
    nodes: list[int]
    labels: np.ndarray
    trial_weights: np.ndarray
    trial_scores: np.ndarray
    trial_indices: list[int]
    num_clusters: list[int]
    _node_positions: dict[int, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_node_positions", {node: i for i, node in enumerate(self.nodes)}
        )

    @classmethod
    def from_trials(
        cls,
        nodes: Sequence[int],
        trials: Sequence[QueryResult],
        *,
        trial_weighting: TrialWeighting = "score_softmax",
        temperature: float | Literal["auto"] = "auto",
    ) -> ConsensusResult:
        nodes = list(nodes)
        if not trials:
            raise ValueError("consensus requires at least one trial")

        labels = np.asarray([trial.labels for trial in trials], dtype=np.int64)
        if labels.ndim != 2 or labels.shape[1] != len(nodes):
            raise ValueError("trial labels must have shape (num_trials, num_nodes)")

        scores = np.asarray([trial_score(trial) for trial in trials], dtype=np.float64)
        weights = trial_weights(
            scores,
            trial_weighting=trial_weighting,
            temperature=temperature,
        )
        labels.setflags(write=False)
        scores.setflags(write=False)
        weights.setflags(write=False)

        return cls(
            nodes=nodes,
            labels=labels,
            trial_weights=weights,
            trial_scores=scores,
            trial_indices=[trial.trial_index for trial in trials],
            num_clusters=[trial.num_clusters for trial in trials],
        )

    @property
    def num_trials(self) -> int:
        return int(self.labels.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.labels.shape[1])

    def score_pair(self, u: int, v: int) -> float:
        left, right = self._positions_for_pairs([(u, v)])
        return float(self.trial_weights @ (self.labels[:, left[0]] == self.labels[:, right[0]]))

    def score_pairs(
        self, pairs: Iterable[tuple[int, int]]
    ) -> np.ndarray:
        pairs = list(pairs)
        if not pairs:
            return np.asarray([], dtype=np.float64)

        left, right = self._positions_for_pairs(pairs)
        same = self.labels[:, left] == self.labels[:, right]
        return self.trial_weights @ same

    def threshold_pairs(
        self,
        pairs: Iterable[tuple[int, int]],
        *,
        threshold: float = 0.8,
    ) -> list[tuple[int, int, float]]:
        pairs = list(pairs)
        probs = self.score_pairs(pairs)
        return [
            (u, v, float(prob))
            for (u, v), prob in zip(pairs, probs, strict=True)
            if prob >= threshold
        ]

    def connected_components(
        self,
        pairs: Iterable[tuple[int, int]],
        *,
        threshold: float = 0.8,
        include_singletons: bool = True,
    ) -> list[list[int]]:
        active_nodes: set[int] = set()
        parent = {node: node for node in self.nodes}

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(u: int, v: int) -> None:
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                parent[root_v] = root_u

        for u, v, _ in self.threshold_pairs(pairs, threshold=threshold):
            active_nodes.update((u, v))
            union(u, v)

        components: dict[int, list[int]] = {}
        component_nodes = self.nodes if include_singletons else sorted(active_nodes)
        for node in component_nodes:
            root = find(node)
            components.setdefault(root, []).append(node)

        return [sorted(component) for component in components.values()]

    def score_matrix(self, nodes: Sequence[int] | None = None) -> np.ndarray:
        selected = self.nodes if nodes is None else list(nodes)
        positions = np.asarray([self._position(node) for node in selected], dtype=np.int64)
        labels = self.labels[:, positions]
        same = labels[:, :, None] == labels[:, None, :]
        return np.tensordot(self.trial_weights, same, axes=(0, 0))

    def _positions_for_pairs(
        self, pairs: Sequence[tuple[int, int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        left = np.empty(len(pairs), dtype=np.int64)
        right = np.empty(len(pairs), dtype=np.int64)
        for i, (u, v) in enumerate(pairs):
            left[i] = self._position(u)
            right[i] = self._position(v)
        return left, right

    def _position(self, node: int) -> int:
        try:
            return self._node_positions[node]
        except KeyError as exc:
            raise KeyError(f"node {node} was not part of this query") from exc


def trial_score(trial: QueryResult) -> float:
    if trial.scores is None:
        return float("nan")
    return float(np.sum(trial.scores))


def trial_weights(
    scores: np.ndarray,
    *,
    trial_weighting: TrialWeighting,
    temperature: float | Literal["auto"],
) -> np.ndarray:
    if trial_weighting == "uniform":
        return np.full(len(scores), 1.0 / len(scores), dtype=np.float64)

    if not np.all(np.isfinite(scores)):
        raise ValueError(f"{trial_weighting!r} requires finite trial scores")

    if trial_weighting == "inverse_score":
        if np.any(scores <= 0.0):
            raise ValueError("'inverse_score' requires positive trial scores")
        raw = 1.0 / np.maximum(scores, 1e-12)
        return raw / np.sum(raw)

    if trial_weighting != "score_softmax":
        raise ValueError(
            "trial_weighting must be 'uniform', 'inverse_score', or 'score_softmax'"
        )

    temp = auto_temperature(scores) if temperature == "auto" else float(temperature)
    if not np.isfinite(temp) or temp <= 0.0:
        raise ValueError("temperature must be positive and finite")

    scaled = -(scores - np.min(scores)) / temp
    scaled -= np.max(scaled)
    raw = np.exp(scaled)
    return raw / np.sum(raw)


def auto_temperature(scores: np.ndarray) -> float:
    spread = float(np.std(scores))
    return max(spread, 1e-12)


__all__ = [
    "ConsensusResult",
    "EdgeUpdateStats",
    "QueryResult",
    "Raven",
    "RavenError",
]
