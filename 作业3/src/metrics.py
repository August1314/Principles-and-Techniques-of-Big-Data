from __future__ import annotations

from collections import Counter

import numpy as np


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    mask = y_pred != -1
    if not np.any(mask):
        return 0.0

    mapped_predictions = np.full_like(y_pred, fill_value=-1)
    for cluster_id in np.unique(y_pred[mask]):
        cluster_members = y_true[y_pred == cluster_id]
        majority_label = Counter(cluster_members.tolist()).most_common(1)[0][0]
        mapped_predictions[y_pred == cluster_id] = majority_label

    return float(np.mean(mapped_predictions == y_true))


def silhouette_score_custom(X: np.ndarray, labels: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    valid_mask = labels != -1
    X = X[valid_mask]
    labels = labels[valid_mask]

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(X) <= len(unique_labels):
        return 0.0

    distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    scores = []

    for index in range(len(X)):
        same_cluster = labels == labels[index]
        same_cluster[index] = False
        if np.any(same_cluster):
            a_i = float(np.mean(distances[index, same_cluster]))
        else:
            a_i = 0.0

        b_i = np.inf
        for other_label in unique_labels:
            if other_label == labels[index]:
                continue
            other_cluster = labels == other_label
            b_i = min(b_i, float(np.mean(distances[index, other_cluster])))

        if not np.isfinite(b_i):
            continue
        denominator = max(a_i, b_i)
        scores.append(0.0 if denominator == 0 else (b_i - a_i) / denominator)

    return float(np.mean(scores)) if scores else 0.0


def calinski_harabasz_score_custom(X: np.ndarray, labels: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    valid_mask = labels != -1
    X = X[valid_mask]
    labels = labels[valid_mask]

    unique_labels = np.unique(labels)
    n_samples = len(X)
    n_clusters = len(unique_labels)
    if n_clusters < 2 or n_samples <= n_clusters:
        return 0.0

    overall_mean = np.mean(X, axis=0)
    between_dispersion = 0.0
    within_dispersion = 0.0

    for cluster_label in unique_labels:
        cluster_points = X[labels == cluster_label]
        cluster_mean = np.mean(cluster_points, axis=0)
        between_dispersion += len(cluster_points) * np.sum((cluster_mean - overall_mean) ** 2)
        within_dispersion += np.sum((cluster_points - cluster_mean) ** 2)

    if within_dispersion == 0:
        return 0.0

    return float(
        (between_dispersion / (n_clusters - 1)) / (within_dispersion / (n_samples - n_clusters))
    )
