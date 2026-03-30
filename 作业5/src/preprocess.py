from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def to_numpy_series(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float).reshape(-1)


def split_series(values: Iterable[float], test_ratio: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    series = to_numpy_series(values)
    if len(series) < 3:
        raise ValueError("series length must be at least 3")
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    split_index = max(1, min(len(series) - 1, int(round(len(series) * (1 - test_ratio)))))
    train = series[:split_index]
    test = series[split_index:]
    if len(train) < 2 or len(test) < 1:
        raise ValueError("train/test split is too small")
    return train, test


def create_sliding_windows(values: Iterable[float], window_size: int) -> tuple[np.ndarray, np.ndarray]:
    series = to_numpy_series(values)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if len(series) <= window_size:
        raise ValueError("series length must be greater than window_size")

    features = []
    targets = []
    for index in range(len(series) - window_size):
        features.append(series[index : index + window_size])
        targets.append(series[index + window_size])
    return np.asarray(features, dtype=float), np.asarray(targets, dtype=float)


def fit_minmax_scaler(train_values: Iterable[float]) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    train_array = to_numpy_series(train_values).reshape(-1, 1)
    scaler.fit(train_array)
    return scaler


def scale_series(scaler: MinMaxScaler, values: Iterable[float]) -> np.ndarray:
    array = to_numpy_series(values).reshape(-1, 1)
    return scaler.transform(array).reshape(-1)


def inverse_scale_series(scaler: MinMaxScaler, values: Iterable[float]) -> np.ndarray:
    array = to_numpy_series(values).reshape(-1, 1)
    return scaler.inverse_transform(array).reshape(-1)
