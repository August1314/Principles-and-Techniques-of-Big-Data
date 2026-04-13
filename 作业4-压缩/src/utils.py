from __future__ import annotations

import random
import time
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def timer() -> float:
    start = time.perf_counter()
    elapsed = [0.0]
    try:
        yield elapsed
    finally:
        elapsed[0] = time.perf_counter() - start
