import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    repo_dir = Path(__file__).resolve().parents[1]
    outdir = repo_dir / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    python = sys.executable

    edge_img = repo_dir / "data" / "edge_input.png"
    if not edge_img.exists():
        run([python, str(repo_dir / "src" / "edge_sobel.py"), "--make-demo", str(edge_img)])
    run([python, str(repo_dir / "src" / "edge_sobel.py"), "--image", str(edge_img), "--outdir", str(outdir)])

    svm_out = outdir / "svm_metrics.json"
    cnn_out = outdir / "cnn_metrics.json"
    run([python, str(repo_dir / "src" / "train_svm_mnist.py"), "--out", str(svm_out)])
    run([python, str(repo_dir / "src" / "train_cnn_mnist.py"), "--out", str(cnn_out)])

    svm = load_json(svm_out)
    cnn = load_json(cnn_out)

    comparison = outdir / "comparison.md"
    comparison.write_text(
        "\n".join(
            [
                "# 作业1：分类方法对比（自动生成）",
                "",
                "| 方法 | 准确率 | 训练耗时(s) | 推理/评估耗时(s) | 单张耗时(ms) |",
                "|---|---:|---:|---:|---:|",
                f"| {svm['method']} | {svm['accuracy']:.4f} | {svm['train_seconds']:.2f} | {svm['infer_seconds']:.2f} | {svm['infer_ms_per_image']:.3f} |",
                f"| {cnn['method']} ({cnn['device']}) | {cnn['accuracy']:.4f} | {cnn['train_seconds']:.2f} | {cnn['eval_seconds']:.2f} | {cnn['eval_ms_per_image']:.3f} |",
                "",
                "说明：耗时包含当前机器与当前参数设置下的结果；可在报告中说明硬件与参数。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved: {comparison}")


if __name__ == "__main__":
    main()

