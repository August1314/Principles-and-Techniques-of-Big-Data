from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dbscan import DBSCANClustering
from kmeans import KMeansClustering
from load_data import load_iris_dataset
from metrics import (
    calinski_harabasz_score_custom,
    clustering_accuracy,
    silhouette_score_custom,
)
from visualize import plot_clusters_2d


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"


def evaluate_clustering(X: np.ndarray, y_true: np.ndarray, labels: np.ndarray) -> dict:
    cluster_count = len({label for label in labels.tolist() if label != -1})
    noise_count = int(np.sum(labels == -1))
    return {
        "accuracy": round(clustering_accuracy(y_true, labels), 4),
        "silhouette_score": round(silhouette_score_custom(X, labels), 4),
        "calinski_harabasz_score": round(calinski_harabasz_score_custom(X, labels), 4),
        "cluster_count": cluster_count,
        "noise_count": noise_count,
    }


def run_kmeans_experiments(X: np.ndarray, y_true: np.ndarray) -> list[dict]:
    results = []
    for k in [2, 3, 4, 5]:
        model = KMeansClustering(n_clusters=k, max_iter=200, tol=1e-6, random_state=42)
        labels = model.fit_predict(X)
        metrics = evaluate_clustering(X, y_true, labels)
        result = {"algorithm": "KMeans", "k": k, **metrics}
        results.append(result)

        plot_clusters_2d(
            X[:, :2],
            labels,
            title=f"K-means (k={k}) on Iris",
            output_path=OUTPUT_DIR / f"kmeans_k{k}.png",
            centers=model.cluster_centers_[:, :2],
        )
    return results


def run_dbscan_experiments(X: np.ndarray, y_true: np.ndarray) -> list[dict]:
    results = []
    parameter_grid = [
        {"eps": 0.3, "min_samples": 3},
        {"eps": 0.5, "min_samples": 4},
        {"eps": 0.6, "min_samples": 5},
        {"eps": 0.8, "min_samples": 5},
    ]

    for config in parameter_grid:
        model = DBSCANClustering(eps=config["eps"], min_samples=config["min_samples"])
        labels = model.fit_predict(X)
        metrics = evaluate_clustering(X, y_true, labels)
        result = {"algorithm": "DBSCAN", **config, **metrics}
        results.append(result)

        plot_clusters_2d(
            X[:, :2],
            labels,
            title=f"DBSCAN (eps={config['eps']}, min_samples={config['min_samples']})",
            output_path=OUTPUT_DIR
            / f"dbscan_eps{str(config['eps']).replace('.', '_')}_min{config['min_samples']}.png",
        )
    return results


def save_results(results: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "metrics_summary.json"
    markdown_path = OUTPUT_DIR / "metrics_summary.md"

    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    headers = [
        "algorithm",
        "k",
        "eps",
        "min_samples",
        "accuracy",
        "silhouette_score",
        "calinski_harabasz_score",
        "cluster_count",
        "noise_count",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for result in results:
        lines.append("| " + " | ".join(str(result.get(header, "")) for header in headers) + " |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    X, y_true, _, _ = load_iris_dataset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    kmeans_results = run_kmeans_experiments(X, y_true)
    dbscan_results = run_dbscan_experiments(X, y_true)
    all_results = kmeans_results + dbscan_results
    save_results(all_results)


if __name__ == "__main__":
    main()
