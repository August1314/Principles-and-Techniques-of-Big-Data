from __future__ import annotations

import argparse
import json
from pathlib import Path

from features import build_tfidf_features
from load_data import load_imdb_dataset
from train_lr import train_eval_logistic_regression
from train_rnn import train_eval_rnn
from train_svm import train_eval_linear_svm
from utils import set_seed


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs"


def save_results(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json = output_dir / "metrics_summary.json"
    summary_md = output_dir / "metrics_summary.md"

    summary_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    headers = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "train_seconds",
        "infer_seconds",
    ]
    lines = [
        "# 作业4：电影评论情感分类结果汇总（自动生成）",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for item in results:
        row = []
        for key in headers:
            value = item[key]
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            else:
                row.append(str(value))
        lines.append("| " + " | ".join(row) + " |")

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for result in results:
        file_name = result["model"].split("(")[0].replace(" ", "_").replace("-", "_").lower()
        single_path = output_dir / f"{file_name}_metrics.json"
        single_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    seed: int,
    sample_per_class: int | None = None,
    auto_download: bool = True,
) -> list[dict]:
    set_seed(seed)

    train_texts, train_labels, test_texts, test_labels = load_imdb_dataset(
        data_dir=data_dir,
        seed=seed,
        sample_per_class=sample_per_class,
        auto_download=auto_download,
    )

    _, X_train, X_test = build_tfidf_features(train_texts, test_texts)

    lr_metrics = train_eval_logistic_regression(X_train, train_labels, X_test, test_labels)
    svm_metrics = train_eval_linear_svm(X_train, train_labels, X_test, test_labels)
    rnn_metrics = train_eval_rnn(train_texts, train_labels, test_texts, test_labels)

    results = [lr_metrics, svm_metrics, rnn_metrics]
    save_results(results, output_dir)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="作业4：IMDB 电影评论情感分类")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=None,
        help="每个类别抽样数量（用于快速烟雾测试）",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="不自动下载 IMDB 数据集，仅使用本地 data/aclImdb",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        sample_per_class=args.sample_per_class,
        auto_download=not args.no_download,
    )

    print("\n=== 作业4结果汇总 ===")
    for item in results:
        print(
            f"{item['model']}: "
            f"acc={item['accuracy']:.4f}, "
            f"precision={item['precision']:.4f}, "
            f"recall={item['recall']:.4f}, "
            f"f1={item['f1']:.4f}, "
            f"train={item['train_seconds']:.2f}s, "
            f"infer={item['infer_seconds']:.2f}s"
        )


if __name__ == "__main__":
    main()
