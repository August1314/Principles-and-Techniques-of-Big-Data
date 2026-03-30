from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_all import run_pipeline


class TestRunAllSmoke(unittest.TestCase):
    def test_run_pipeline_smoke(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "tiny_stock.csv"
            output_dir = tmp_path / "outputs"

            dates = pd.date_range("2024-01-01", periods=40, freq="B")
            close = np.linspace(10.0, 20.0, num=40) + np.sin(np.arange(40)) * 0.2
            pd.DataFrame({"date": dates, "close": close}).to_csv(csv_path, index=False)

            results = run_pipeline(
                csv_path=csv_path,
                output_dir=output_dir,
                seed=42,
                test_ratio=0.2,
                future_days=3,
                lstm_window_size=5,
                lstm_epochs=2,
                lstm_hidden_size=8,
                arima_p_values=(0, 1),
                arima_q_values=(0, 1),
            )

            self.assertEqual(len(results), 2)
            self.assertTrue((output_dir / "metrics_summary.json").exists())
            self.assertTrue((output_dir / "metrics_summary.md").exists())
            self.assertTrue((output_dir / "arima_test_prediction.csv").exists())
            self.assertTrue((output_dir / "lstm_test_prediction.csv").exists())
            self.assertTrue((output_dir / "future_7days_predictions.csv").exists())


if __name__ == "__main__":
    unittest.main()
