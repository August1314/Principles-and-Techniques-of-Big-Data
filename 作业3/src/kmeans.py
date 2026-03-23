from __future__ import annotations

import numpy as np


class KMeansClustering:
    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray) -> "KMeansClustering":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples must be >= n_clusters")

        rng = np.random.default_rng(self.random_state)
        centers = self._initialize_centers(X, rng)

        for iteration in range(1, self.max_iter + 1):
            distances = self._pairwise_distances(X, centers)
            labels = np.argmin(distances, axis=1)

            new_centers = centers.copy()
            for cluster_id in range(self.n_clusters):
                members = X[labels == cluster_id]
                if len(members) == 0:
                    new_centers[cluster_id] = X[rng.integers(0, X.shape[0])]
                else:
                    new_centers[cluster_id] = members.mean(axis=0)

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift <= self.tol:
                self.n_iter_ = iteration
                break
        else:
            self.n_iter_ = self.max_iter

        final_distances = self._pairwise_distances(X, centers)
        labels = np.argmin(final_distances, axis=1)
        min_distances = final_distances[np.arange(X.shape[0]), labels]

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = float(np.sum(min_distances ** 2))
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted")
        X = np.asarray(X, dtype=float)
        distances = self._pairwise_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)

    @staticmethod
    def _pairwise_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        return np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)

    def _initialize_centers(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[indices].copy()
