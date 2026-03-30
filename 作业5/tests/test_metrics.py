import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from metrics import build_metric_record, mae, rmse


class MetricsTestCase(unittest.TestCase):
    def test_mae_returns_expected_value(self):
        actual = [10.0, 12.0, 14.0]
        predicted = [9.0, 13.0, 15.0]
        self.assertAlmostEqual(mae(actual, predicted), 1.0, places=6)

    def test_rmse_returns_expected_value(self):
        actual = [10.0, 12.0, 14.0]
        predicted = [9.0, 13.0, 15.0]
        self.assertAlmostEqual(rmse(actual, predicted), 1.0, places=6)

    def test_build_metric_record_has_required_fields(self):
        record = build_metric_record(
            model_name="ARIMA(2,1,2)",
            actual=[10.0, 12.0, 14.0],
            predicted=[9.0, 13.0, 15.0],
            train_seconds=0.25,
            infer_seconds=0.05,
        )
        self.assertEqual(record["model"], "ARIMA(2,1,2)")
        self.assertAlmostEqual(record["mae"], 1.0, places=6)
        self.assertAlmostEqual(record["rmse"], 1.0, places=6)
        self.assertAlmostEqual(record["train_seconds"], 0.25, places=6)
        self.assertAlmostEqual(record["infer_seconds"], 0.05, places=6)


if __name__ == "__main__":
    unittest.main()
