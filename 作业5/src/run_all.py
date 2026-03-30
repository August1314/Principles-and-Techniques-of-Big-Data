from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from load_data import load_stock_csv
from preprocess import split_series
from train_arima import train_evaluate_arima
from train_lstm import LSTMConfig, train_evaluate_lstm
from visualize import (
    plot_error_bar_chart,
    plot_future_forecast,
    plot_model_comparison,
    plot_prediction_vs_actual,
    plot_raw_close_series,
    plot_train_test_split,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "stock_history.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_results(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    headers = ["model", "mae", "rmse", "train_seconds", "infer_seconds"]
    lines = [
        "# 作业5：股票价格预测结果汇总（自动生成）",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for result in results:
        row = []
        for header in headers:
            value = result[header]
            row.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(row) + " |")
    (output_dir / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for result in results:
        file_name = result["model"].replace("(", "_").replace(")", "_").replace(",", "_").replace(" ", "").lower()
        (output_dir / f"{file_name}_metrics.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def save_prediction_csv(
    dates: pd.Series,
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
) -> None:
    pd.DataFrame({"date": dates, "actual": actual, "predicted": predicted}).to_csv(output_path, index=False)


def save_future_csv(
    future_dates: pd.Series,
    arima_future: np.ndarray,
    lstm_future: np.ndarray,
    output_path: Path,
) -> None:
    pd.DataFrame(
        {
            "date": future_dates,
            "arima_prediction": arima_future,
            "lstm_prediction": lstm_future,
        }
    ).to_csv(output_path, index=False)


def save_report_tables(results: list[dict], future_frame: pd.DataFrame, output_dir: Path) -> None:
    metrics_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{ARIMA 与 LSTM 测试集指标对比}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "模型 & MAE & RMSE & Train(s) & Infer(s) \\\\",
        "\\midrule",
    ]
    for result in results:
        metrics_lines.append(
            f"{result['model']} & {result['mae']:.4f} & {result['rmse']:.4f} & "
            f"{result['train_seconds']:.2f} & {result['infer_seconds']:.2f} \\\\"
        )
    metrics_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (output_dir / "report_metrics_table.tex").write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")

    future_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{未来 7 天预测结果}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "日期 & ARIMA & LSTM \\\\",
        "\\midrule",
    ]
    for row in future_frame.itertuples(index=False):
        future_lines.append(
            f"{pd.Timestamp(row.date).strftime('%Y-%m-%d')} & "
            f"{float(row.arima_prediction):.4f} & {float(row.lstm_prediction):.4f} \\\\"
        )
    future_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (output_dir / "report_future_table.tex").write_text("\n".join(future_lines) + "\n", encoding="utf-8")


def run_pipeline(
    csv_path: Path = DEFAULT_DATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
    test_ratio: float = 0.2,
    future_days: int = 7,
    lstm_window_size: int = 20,
    lstm_epochs: int = 60,
    lstm_hidden_size: int = 32,
    arima_p_values: tuple[int, ...] = (0, 1, 2),
    arima_q_values: tuple[int, ...] = (0, 1, 2),
) -> list[dict]:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    frame = load_stock_csv(csv_path)
    close_values = frame["close"].to_numpy(dtype=float)
    train_values, test_values = split_series(close_values, test_ratio=test_ratio)
    split_index = len(train_values)
    test_dates = frame["date"].iloc[split_index:].reset_index(drop=True)

    arima_result = train_evaluate_arima(
        train_values=train_values,
        test_values=test_values,
        future_days=future_days,
        p_values=arima_p_values,
        q_values=arima_q_values,
    )
    lstm_result = train_evaluate_lstm(
        train_values=train_values,
        test_values=test_values,
        config=LSTMConfig(
            window_size=lstm_window_size,
            hidden_size=lstm_hidden_size,
            epochs=lstm_epochs,
            future_days=future_days,
            seed=seed,
        ),
    )

    results = [arima_result["metrics"], lstm_result["metrics"]]
    save_results(results, output_dir)

    save_prediction_csv(test_dates, test_values, arima_result["test_prediction"], output_dir / "arima_test_prediction.csv")
    save_prediction_csv(test_dates, test_values, lstm_result["test_prediction"], output_dir / "lstm_test_prediction.csv")

    future_dates = pd.bdate_range(frame["date"].iloc[-1] + pd.Timedelta(days=1), periods=future_days)
    future_frame = pd.DataFrame(
        {
            "date": future_dates,
            "arima_prediction": arima_result["future_prediction"],
            "lstm_prediction": lstm_result["future_prediction"],
        }
    )
    future_frame.to_csv(output_dir / "future_7days_predictions.csv", index=False)
    save_report_tables(results, future_frame, output_dir)

    plot_raw_close_series(frame, figures_dir / "raw_close_series.png")
    plot_train_test_split(frame, split_index, figures_dir / "train_test_split.png")
    plot_prediction_vs_actual(
        test_dates,
        test_values,
        arima_result["test_prediction"],
        "ARIMA 测试集预测对比",
        figures_dir / "arima_vs_actual.png",
        "ARIMA预测",
    )
    plot_prediction_vs_actual(
        test_dates,
        test_values,
        lstm_result["test_prediction"],
        "LSTM 测试集预测对比",
        figures_dir / "lstm_vs_actual.png",
        "LSTM预测",
    )
    plot_model_comparison(
        test_dates,
        test_values,
        arima_result["test_prediction"],
        lstm_result["test_prediction"],
        figures_dir / "model_comparison.png",
    )
    plot_error_bar_chart(results, figures_dir / "error_bar_chart.png")
    plot_future_forecast(
        frame["date"],
        frame["close"],
        future_dates,
        arima_result["future_prediction"],
        lstm_result["future_prediction"],
        figures_dir / "future_7days_forecast.png",
    )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="作业5：股票价格预测")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--future-days", type=int, default=7)
    parser.add_argument("--lstm-window-size", type=int, default=20)
    parser.add_argument("--lstm-epochs", type=int, default=60)
    parser.add_argument("--lstm-hidden-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_pipeline(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        seed=args.seed,
        test_ratio=args.test_ratio,
        future_days=args.future_days,
        lstm_window_size=args.lstm_window_size,
        lstm_epochs=args.lstm_epochs,
        lstm_hidden_size=args.lstm_hidden_size,
    )
    print("\n=== 作业5结果汇总 ===")
    for item in results:
        print(
            f"{item['model']}: "
            f"mae={item['mae']:.4f}, "
            f"rmse={item['rmse']:.4f}, "
            f"train={item['train_seconds']:.2f}s, "
            f"infer={item['infer_seconds']:.2f}s"
        )


if __name__ == "__main__":
    main()
