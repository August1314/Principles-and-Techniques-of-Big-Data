from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class MnistNumpy:
    x_train: np.ndarray  # (N, 784) float32 in [0,1]
    y_train: np.ndarray  # (N,) int64
    x_test: np.ndarray  # (N, 784) float32 in [0,1]
    y_test: np.ndarray  # (N,) int64


def load_mnist_numpy(data_dir: str | Path = "data") -> MnistNumpy:
    """
    Loads MNIST via torchvision (download if needed) and returns flattened numpy arrays.
    Kept in a small helper to share between sklearn and torch training.
    """
    try:
        import torch
        from torchvision import datasets, transforms
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "torch/torchvision not available. Install requirements.txt first."
        ) from e

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(str(data_dir), train=False, download=True, transform=tfm)

    def to_numpy(ds) -> Tuple[np.ndarray, np.ndarray]:
        xs = []
        ys = []
        for x, y in ds:
            xs.append(x.numpy())  # (1,28,28)
            ys.append(y)
        x_arr = np.stack(xs, axis=0).astype(np.float32)  # (N,1,28,28)
        y_arr = np.array(ys, dtype=np.int64)
        x_arr = x_arr.reshape(x_arr.shape[0], -1)  # (N,784)
        return x_arr, y_arr

    x_train, y_train = to_numpy(train_ds)
    x_test, y_test = to_numpy(test_ds)

    return MnistNumpy(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

