import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from mnist_data import load_mnist_numpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out", type=str, default="outputs/svm_metrics.json")
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--pca-dim", type=int, default=64)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mnist = load_mnist_numpy(args.data_dir)
    # Use float64 for sklearn PCA/SVM numerical stability on some setups.
    x_train = mnist.x_train[: args.train_size].astype(np.float64, copy=False)
    y_train = mnist.y_train[: args.train_size]
    x_test = mnist.x_test[: args.test_size].astype(np.float64, copy=False)
    y_test = mnist.y_test[: args.test_size]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True)),
            ("pca", PCA(n_components=args.pca_dim, random_state=0, svd_solver="randomized")),
            ("svm", LinearSVC(C=2.0, max_iter=2000, dual="auto", random_state=0)),
        ]
    )

    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    train_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred = model.predict(x_test)
    infer_seconds = time.perf_counter() - t1

    acc = float(accuracy_score(y_test, pred))

    metrics = {
        "method": "PCA+LinearSVC",
        "train_size": int(x_train.shape[0]),
        "test_size": int(x_test.shape[0]),
        "pca_dim": int(args.pca_dim),
        "accuracy": acc,
        "train_seconds": float(train_seconds),
        "infer_seconds": float(infer_seconds),
        "infer_ms_per_image": float(infer_seconds * 1000.0 / x_test.shape[0]),
    }

    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
