from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from metrics_utils import classification_metrics


def train_eval_logistic_regression(
    X_train,
    y_train: list[int],
    X_test,
    y_test: list[int],
) -> dict[str, float | str]:
    model = LogisticRegression(
        solver="liblinear",
        C=4.0,
        max_iter=2000,
        random_state=42,
    )

    import time

    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start_train

    start_infer = time.perf_counter()
    y_pred = model.predict(X_test)
    infer_seconds = time.perf_counter() - start_infer

    metrics = classification_metrics(y_test, y_pred.tolist())
    return {
        "model": "LogisticRegression(TF-IDF)",
        **metrics,
        "train_seconds": float(train_seconds),
        "infer_seconds": float(infer_seconds),
    }
