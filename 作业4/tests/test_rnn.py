from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_rnn import SentimentRNN


class TestRNN(unittest.TestCase):
    def test_rnn_forward_shape_and_backward(self) -> None:
        model = SentimentRNN(vocab_size=100, embed_dim=16, hidden_dim=8)
        batch_x = torch.randint(0, 100, (4, 20), dtype=torch.long)
        batch_y = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)

        logits = model(batch_x)
        self.assertEqual(tuple(logits.shape), (4,))

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, batch_y)
        loss.backward()

        has_grad = any(param.grad is not None for param in model.parameters())
        self.assertTrue(has_grad)


if __name__ == "__main__":
    unittest.main()
