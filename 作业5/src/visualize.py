from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["Heiti SC", "Songti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PRIMARY = "#1f5aa6"
SECONDARY = "#e67e22"
ACCENT = "#2ca58d"
RED = "#c0392b"


def _save_current_figure(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_raw_close_series(frame: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 4.8))
    plt.plot(frame["date"], frame["close"], color=PRIMARY, linewidth=1.8)
    plt.title("Apple 股票收盘价时间序列")
    plt.xlabel("日期")
    plt.ylabel("收盘价")
    _save_current_figure(output_path)


def plot_train_test_split(frame: pd.DataFrame, split_index: int, output_path: Path) -> None:
    plt.figure(figsize=(10, 4.8))
    plt.plot(frame["date"].iloc[:split_index], frame["close"].iloc[:split_index], color=PRIMARY, label="训练集")
    plt.plot(frame["date"].iloc[split_index:], frame["close"].iloc[split_index:], color=SECONDARY, label="测试集")
    plt.axvline(frame["date"].iloc[split_index], color=RED, linestyle="--", linewidth=1.2, label="切分点")
    plt.title("训练集 / 测试集时间切分")
    plt.xlabel("日期")
    plt.ylabel("收盘价")
    plt.legend()
    _save_current_figure(output_path)


def plot_prediction_vs_actual(
    dates: pd.Series,
    actual: pd.Series | list[float],
    predicted: pd.Series | list[float],
    title: str,
    output_path: Path,
    predicted_label: str,
) -> None:
    plt.figure(figsize=(10, 4.8))
    plt.plot(dates, actual, color=PRIMARY, linewidth=1.8, label="真实值")
    plt.plot(dates, predicted, color=SECONDARY, linewidth=1.6, label=predicted_label)
    plt.title(title)
    plt.xlabel("日期")
    plt.ylabel("收盘价")
    plt.legend()
    _save_current_figure(output_path)


def plot_model_comparison(
    dates: pd.Series,
    actual: pd.Series | list[float],
    arima_prediction: pd.Series | list[float],
    lstm_prediction: pd.Series | list[float],
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 4.8))
    plt.plot(dates, actual, color=PRIMARY, linewidth=1.8, label="真实值")
    plt.plot(dates, arima_prediction, color=SECONDARY, linewidth=1.6, label="ARIMA")
    plt.plot(dates, lstm_prediction, color=ACCENT, linewidth=1.6, label="LSTM")
    plt.title("ARIMA 与 LSTM 在测试集上的预测对比")
    plt.xlabel("日期")
    plt.ylabel("收盘价")
    plt.legend()
    _save_current_figure(output_path)


def plot_error_bar_chart(results: list[dict], output_path: Path) -> None:
    labels = [item["model"] for item in results]
    mae_values = [item["mae"] for item in results]
    rmse_values = [item["rmse"] for item in results]
    positions = range(len(labels))

    plt.figure(figsize=(8.8, 4.8))
    plt.bar([p - 0.18 for p in positions], mae_values, width=0.36, color=PRIMARY, label="MAE")
    plt.bar([p + 0.18 for p in positions], rmse_values, width=0.36, color=SECONDARY, label="RMSE")
    plt.xticks(list(positions), labels)
    plt.title("模型误差指标对比")
    plt.ylabel("误差值")
    plt.legend()
    _save_current_figure(output_path)


def plot_future_forecast(
    historical_dates: pd.Series,
    historical_close: pd.Series,
    future_dates: pd.Series,
    arima_future: pd.Series | list[float],
    lstm_future: pd.Series | list[float],
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 4.8))
    recent_dates = historical_dates.iloc[-60:]
    recent_close = historical_close.iloc[-60:]
    plt.plot(recent_dates, recent_close, color="#666666", linewidth=1.4, label="最近60个交易日")
    plt.plot(future_dates, arima_future, color=SECONDARY, marker="o", label="ARIMA未来7天")
    plt.plot(future_dates, lstm_future, color=ACCENT, marker="o", label="LSTM未来7天")
    plt.title("未来 7 天价格预测")
    plt.xlabel("日期")
    plt.ylabel("收盘价")
    plt.legend()
    _save_current_figure(output_path)
