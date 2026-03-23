from __future__ import annotations

from sklearn.datasets import load_iris


def load_iris_dataset():
    dataset = load_iris()
    return dataset.data, dataset.target, dataset.target_names, dataset.feature_names
