from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features import build_tfidf_features


class TestTfidf(unittest.TestCase):
    def test_tfidf_shape_and_non_empty(self) -> None:
        train_texts = ["good movie", "bad movie", "excellent plot"]
        test_texts = ["good plot", "bad plot"]

        _, x_train, x_test = build_tfidf_features(train_texts, test_texts)
        self.assertEqual(x_train.shape[0], 3)
        self.assertEqual(x_test.shape[0], 2)
        self.assertGreater(x_train.shape[1], 0)
        self.assertGreater(x_train.nnz, 0)


if __name__ == "__main__":
    unittest.main()
