from __future__ import annotations

import time
import warnings
from itertools import product
from typing import Iterable

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from metrics import build_metric_record
from preprocess import to_numpy_series


def infer_difference_order(series: Iterable[float], max_d: int = 2) -> int:
    current = to_numpy_series(series)
    for degree in range(max_d + 1):
        if len(current) < 8:
            return degree
        try:
            p_value = adfuller(current, autolag="AIC")[1]
            if p_value < 0.05:
                return degree
        except Exception:
            return degree
        current = np.diff(current)
    return max_d


def select_arima_order(
    train_values: Iterable[float],
    p_values: tuple[int, ...] = (0, 1, 2),
    q_values: tuple[int, ...] = (0, 1, 2),
) -> tuple[int, int, int]:
    train_array = to_numpy_series(train_values)
    d_value = infer_difference_order(train_array)
    best_order = (1, d_value, 0)
    best_aic = float("inf")

    for p_value, q_value in product(p_values, q_values):
        order = (p_value, d_value, q_value)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ARIMA(
                    train_array,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()
            if np.isfinite(fitted.aic) and fitted.aic < best_aic:
                best_aic = float(fitted.aic)
                best_order = order
        except Exception:
            continue
    return best_order


def train_evaluate_arima(
    train_values: Iterable[float],
    test_values: Iterable[float],
    future_days: int = 7,
    p_values: tuple[int, ...] = (0, 1, 2),
    q_values: tuple[int, ...] = (0, 1, 2),
) -> dict[str, object]:
    train_array = to_numpy_series(train_values)
    test_array = to_numpy_series(test_values)
    order = select_arima_order(train_array, p_values=p_values, q_values=q_values)

    train_start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ARIMA(
            train_array,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
    train_seconds = time.perf_counter() - train_start

    infer_start = time.perf_counter()
    history = train_array.astype(float).tolist()
    rolling_predictions: list[float] = []
    for observed in test_array:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rolling_model = ARIMA(
                history,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()
        next_prediction = float(rolling_model.forecast(steps=1)[0])
        rolling_predictions.append(next_prediction)
        history.append(float(observed))

    test_prediction = np.asarray(rolling_predictions, dtype=float)
    full_series = np.asarray(history, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        future_model = ARIMA(
            full_series,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
    future_prediction = np.asarray(future_model.forecast(steps=future_days), dtype=float)
    infer_seconds = time.perf_counter() - infer_start

    metrics = build_metric_record(
        model_name=f"ARIMA{order}",
        actual=test_array,
        predicted=test_prediction,
        train_seconds=train_seconds,
        infer_seconds=infer_seconds,
    )
    return {
        "metrics": metrics,
        "order": order,
        "test_prediction": test_prediction,
        "future_prediction": future_prediction,
    }
