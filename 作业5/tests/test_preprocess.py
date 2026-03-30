import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from load_data import load_stock_csv
from preprocess import create_sliding_windows, split_series


class PreprocessTestCase(unittest.TestCase):
    def test_load_stock_csv_sorts_and_keeps_close(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "stock.csv")
            pd.DataFrame(
                {
                    "Date": ["2024-01-03", "2024-01-01", "2024-01-02"],
                    "Close": [12.0, 10.0, 11.0],
                }
            ).to_csv(csv_path, index=False)
            frame = load_stock_csv(csv_path)
            self.assertEqual(
                frame["date"].dt.strftime("%Y-%m-%d").tolist(),
                ["2024-01-01", "2024-01-02", "2024-01-03"],
            )
            self.assertEqual(frame["close"].tolist(), [10.0, 11.0, 12.0])

    def test_split_series_uses_time_order(self):
        train, test = split_series([1, 2, 3, 4, 5], test_ratio=0.4)
        self.assertEqual(train.tolist(), [1, 2, 3])
        self.assertEqual(test.tolist(), [4, 5])

    def test_create_sliding_windows_returns_expected_shapes(self):
        x, y = create_sliding_windows([1, 2, 3, 4, 5, 6], window_size=3)
        self.assertEqual(x.tolist(), [[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        self.assertEqual(y.tolist(), [4, 5, 6])


if __name__ == "__main__":
    unittest.main()
