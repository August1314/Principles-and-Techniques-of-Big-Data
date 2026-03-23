from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_clusters_2d(
    X: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: str | Path,
    centers: np.ndarray | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))

    for color_index, label in enumerate(unique_labels):
        mask = labels == label
        label_name = "Noise" if label == -1 else f"Cluster {label}"
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            s=50,
            alpha=0.8,
            color=colors[color_index % len(colors)],
            label=label_name,
        )

    if centers is not None:
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="X",
            s=180,
            color="black",
            label="Centers",
        )

    ax.set_xlabel("Sepal length")
    ax.set_ylabel("Sepal width")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
