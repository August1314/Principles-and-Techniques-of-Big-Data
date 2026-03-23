from __future__ import annotations

from collections import deque

import numpy as np


class DBSCANClustering:
    def __init__(self, eps: float, min_samples: int = 5) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive")
        if min_samples <= 0:
            raise ValueError("min_samples must be positive")
        self.eps = eps
        self.min_samples = min_samples
        self.labels_: np.ndarray | None = None
        self.core_sample_indices_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "DBSCANClustering":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        n_samples = X.shape[0]
        labels = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)
        neighborhoods = self._build_neighborhoods(X)
        is_core = np.array(
            [len(neighbors) >= self.min_samples for neighbors in neighborhoods],
            dtype=bool,
        )

        cluster_id = 0
        for point_index in range(n_samples):
            if visited[point_index]:
                continue
            visited[point_index] = True
            if not is_core[point_index]:
                continue

            self._expand_cluster(
                point_index=point_index,
                cluster_id=cluster_id,
                labels=labels,
                visited=visited,
                neighborhoods=neighborhoods,
                is_core=is_core,
            )
            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.flatnonzero(is_core)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

    def _build_neighborhoods(self, X: np.ndarray) -> list[np.ndarray]:
        distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        return [np.flatnonzero(distances[i] <= self.eps) for i in range(X.shape[0])]

    def _expand_cluster(
        self,
        point_index: int,
        cluster_id: int,
        labels: np.ndarray,
        visited: np.ndarray,
        neighborhoods: list[np.ndarray],
        is_core: np.ndarray,
    ) -> None:
        queue = deque([point_index])
        labels[point_index] = cluster_id

        while queue:
            current = queue.popleft()
            for neighbor in neighborhoods[current]:
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                if is_core[neighbor]:
                    queue.append(neighbor)
