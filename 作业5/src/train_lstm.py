from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from metrics import build_metric_record
from preprocess import (
    create_sliding_windows,
    fit_minmax_scaler,
    inverse_scale_series,
    scale_series,
    to_numpy_series,
)


def set_torch_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LSTMRegressor(nn.Module):
    def __init__(self, hidden_size: int = 32, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.lstm(inputs)
        last_hidden = sequence_output[:, -1, :]
        return self.output(last_hidden)


@dataclass
class LSTMConfig:
    window_size: int = 20
    hidden_size: int = 32
    num_layers: int = 1
    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    future_days: int = 7
    seed: int = 42


def _build_test_windows(series: np.ndarray, split_index: int, window_size: int) -> np.ndarray:
    windows = []
    for target_index in range(split_index, len(series)):
        start = target_index - window_size
        if start < 0:
            raise ValueError("window_size is too large for the current split")
        windows.append(series[start:target_index])
    return np.asarray(windows, dtype=float)


def train_evaluate_lstm(
    train_values: Iterable[float],
    test_values: Iterable[float],
    config: LSTMConfig | None = None,
) -> dict[str, object]:
    cfg = config or LSTMConfig()
    set_torch_seed(cfg.seed)

    train_array = to_numpy_series(train_values)
    test_array = to_numpy_series(test_values)
    full_array = np.concatenate([train_array, test_array])
    if len(train_array) <= cfg.window_size:
        raise ValueError("train series is too short for the configured window_size")

    scaler = fit_minmax_scaler(train_array)
    train_scaled = scale_series(scaler, train_array)
    full_scaled = scale_series(scaler, full_array)

    x_train, y_train = create_sliding_windows(train_scaled, window_size=cfg.window_size)
    x_test = _build_test_windows(full_scaled, split_index=len(train_array), window_size=cfg.window_size)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    train_loader = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor),
        batch_size=min(cfg.batch_size, len(x_train_tensor)),
        shuffle=True,
    )

    model = LSTMRegressor(hidden_size=cfg.hidden_size, num_layers=cfg.num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    train_start = time.perf_counter()
    model.train()
    for _ in range(cfg.epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer.step()
    train_seconds = time.perf_counter() - train_start

    infer_start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        test_scaled_prediction = model(torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1)).squeeze(-1).numpy()

    future_sequence = full_scaled[-cfg.window_size :].copy()
    future_scaled_prediction: list[float] = []
    with torch.no_grad():
        for _ in range(cfg.future_days):
            future_tensor = torch.tensor(future_sequence.reshape(1, cfg.window_size, 1), dtype=torch.float32)
            next_value = float(model(future_tensor).item())
            future_scaled_prediction.append(next_value)
            future_sequence = np.concatenate([future_sequence[1:], np.asarray([next_value])])
    infer_seconds = time.perf_counter() - infer_start

    test_prediction = inverse_scale_series(scaler, test_scaled_prediction)
    future_prediction = inverse_scale_series(scaler, future_scaled_prediction)
    metrics = build_metric_record(
        model_name="LSTM",
        actual=test_array,
        predicted=test_prediction,
        train_seconds=train_seconds,
        infer_seconds=infer_seconds,
    )
    return {
        "metrics": metrics,
        "test_prediction": test_prediction,
        "future_prediction": future_prediction,
    }
