from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from load_data import load_imdb_dataset, tokenize_text

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"


def load_metrics() -> list[dict]:
    path = OUTPUT_DIR / "metrics_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def short_name(model_name: str) -> str:
    if model_name.startswith("LogisticRegression"):
        return "LR(TF-IDF)"
    if model_name.startswith("LinearSVM"):
        return "SVM(TF-IDF)"
    if model_name.startswith("RNN-LSTM"):
        return "RNN-LSTM"
    return model_name


def plot_metric_bars(metrics: list[dict]) -> None:
    models = [short_name(item["model"]) for item in metrics]
    metric_names = ["accuracy", "precision", "recall", "f1"]
    values = np.array([[item[name] for name in metric_names] for item in metrics])

    x = np.arange(len(models))
    width = 0.18

    plt.figure(figsize=(10, 5))
    for i, name in enumerate(metric_names):
        plt.bar(x + (i - 1.5) * width, values[:, i], width=width, label=name)

    plt.xticks(x, models)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_metric_comparison.png", dpi=180)
    plt.close()


def plot_time_bars(metrics: list[dict]) -> None:
    models = [short_name(item["model"]) for item in metrics]
    train_times = [item["train_seconds"] for item in metrics]
    infer_times = [item["infer_seconds"] for item in metrics]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, train_times, width=width, label="train_seconds")
    plt.bar(x + width / 2, infer_times, width=width, label="infer_seconds")
    plt.yscale("log")
    plt.xticks(x, models)
    plt.ylabel("Seconds (log scale)")
    plt.title("Training and Inference Time Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_time_comparison.png", dpi=180)
    plt.close()


def plot_class_distribution() -> None:
    data_dir = ROOT_DIR / "data"
    train_texts, train_labels, test_texts, test_labels = load_imdb_dataset(
        data_dir=data_dir, auto_download=False
    )
    train_pos = sum(train_labels)
    train_neg = len(train_labels) - train_pos
    test_pos = sum(test_labels)
    test_neg = len(test_labels) - test_pos

    labels = ["Train-Pos", "Train-Neg", "Test-Pos", "Test-Neg"]
    values = [train_pos, train_neg, test_pos, test_neg]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values, color=["#4caf50", "#f44336", "#66bb6a", "#ef5350"])
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 100, str(value), ha="center")
    plt.ylabel("Count")
    plt.title("IMDB Dataset Class Distribution")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_class_distribution.png", dpi=180)
    plt.close()

    # length distributions for train set
    pos_lengths = [len(tokenize_text(text)) for text, label in zip(train_texts, train_labels) if label == 1]
    neg_lengths = [len(tokenize_text(text)) for text, label in zip(train_texts, train_labels) if label == 0]

    plt.figure(figsize=(10, 5))
    plt.hist(pos_lengths, bins=60, alpha=0.6, label="positive", density=True)
    plt.hist(neg_lengths, bins=60, alpha=0.6, label="negative", density=True)
    plt.xlim(0, 1000)
    plt.xlabel("Token Count")
    plt.ylabel("Density")
    plt.title("Review Length Distribution (Train Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_length_distribution.png", dpi=180)
    plt.close()

    # top words
    pos_counter: Counter[str] = Counter()
    neg_counter: Counter[str] = Counter()
    for text, label in zip(train_texts, train_labels):
        tokens = tokenize_text(text)
        if label == 1:
            pos_counter.update(tokens)
        else:
            neg_counter.update(tokens)

    for counter, filename, title, color in [
        (pos_counter, "05_top_words_positive.png", "Top Words in Positive Reviews", "#42a5f5"),
        (neg_counter, "06_top_words_negative.png", "Top Words in Negative Reviews", "#ff7043"),
    ]:
        top_items = counter.most_common(20)
        words = [word for word, _ in reversed(top_items)]
        freqs = [freq for _, freq in reversed(top_items)]

        plt.figure(figsize=(8, 6))
        plt.barh(words, freqs, color=color)
        plt.title(title)
        plt.xlabel("Frequency")
        plt.tight_layout()
        plt.savefig(FIG_DIR / filename, dpi=180)
        plt.close()


def estimated_confusion_matrix(metric: dict, total_per_class: int = 12_500) -> np.ndarray:
    precision = float(metric["precision"])
    recall = float(metric["recall"])

    tp = int(round(recall * total_per_class))
    fn = total_per_class - tp

    if precision == 0:
        fp = total_per_class
    else:
        fp = int(round(tp * (1.0 / precision - 1.0)))
    fp = max(0, min(total_per_class, fp))

    tn = total_per_class - fp
    tn = max(0, min(total_per_class, tn))

    return np.array([[tn, fp], [fn, tp]], dtype=int)


def plot_confusion_matrices(metrics: list[dict]) -> None:
    for idx, metric in enumerate(metrics, start=7):
        cm = estimated_confusion_matrix(metric)
        model = short_name(metric["model"])

        plt.figure(figsize=(5.5, 5))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks([0, 1], ["Pred Neg", "Pred Pos"])
        plt.yticks([0, 1], ["True Neg", "True Pos"])
        plt.title(f"Estimated Confusion Matrix - {model}")

        for i in range(2):
            for j in range(2):
                plt.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")

        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{idx:02d}_cm_{model.lower().replace('(', '').replace(')', '').replace('-', '_')}.png", dpi=180)
        plt.close()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()

    plot_metric_bars(metrics)
    plot_time_bars(metrics)
    plot_class_distribution()
    plot_confusion_matrices(metrics)


if __name__ == "__main__":
    main()
