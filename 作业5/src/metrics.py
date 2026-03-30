from __future__ import annotations

import math
from typing import Iterable


def _to_float_list(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def _validate_lengths(actual: list[float], predicted: list[float]) -> None:
    if not actual or not predicted:
        raise ValueError("actual and predicted must not be empty")
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")


def mae(actual: Iterable[float], predicted: Iterable[float]) -> float:
    actual_list = _to_float_list(actual)
    predicted_list = _to_float_list(predicted)
    _validate_lengths(actual_list, predicted_list)
    return sum(abs(a - p) for a, p in zip(actual_list, predicted_list)) / len(actual_list)


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    actual_list = _to_float_list(actual)
    predicted_list = _to_float_list(predicted)
    _validate_lengths(actual_list, predicted_list)
    mse = sum((a - p) ** 2 for a, p in zip(actual_list, predicted_list)) / len(actual_list)
    return math.sqrt(mse)


def build_metric_record(
    model_name: str,
    actual: Iterable[float],
    predicted: Iterable[float],
    train_seconds: float,
    infer_seconds: float,
) -> dict[str, float | str]:
    return {
        "model": model_name,
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "train_seconds": float(train_seconds),
        "infer_seconds": float(infer_seconds),
    }
