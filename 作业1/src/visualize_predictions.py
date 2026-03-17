import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
    except Exception as e:  # pragma: no cover
        raise RuntimeError("torch/torchvision not available") from e

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=tfm)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    class SimpleCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(32 * 7 * 7, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            return self.net(x)

    # Train quickly (same as train_cnn_mnist but kept local for visualization reproducibility)
    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _epoch in range(args.epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    # Collect predictions
    model.eval()
    all_imgs = []
    all_true = []
    all_pred = []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            pred = torch.argmax(logits, dim=1).cpu()
            all_imgs.append(x.cpu())
            all_true.append(y.cpu())
            all_pred.append(pred)
    imgs = torch.cat(all_imgs, dim=0)  # (N,1,28,28)
    y_true = torch.cat(all_true, dim=0).numpy()
    y_pred = torch.cat(all_pred, dim=0).numpy()

    correct_idx = np.where(y_true == y_pred)[0].tolist()
    wrong_idx = np.where(y_true != y_pred)[0].tolist()
    random.shuffle(correct_idx)
    random.shuffle(wrong_idx)

    def plot_grid(indices: list[int], title: str, out_path: Path) -> None:
        k = min(args.num, len(indices))
        cols = 6
        rows = int(np.ceil(k / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
        axes = np.array(axes).reshape(-1)
        for ax in axes:
            ax.axis("off")
        for i in range(k):
            idx = indices[i]
            ax = axes[i]
            ax.axis("off")
            ax.imshow(imgs[idx, 0].numpy(), cmap="gray")
            ax.set_title(f"t={y_true[idx]}, p={y_pred[idx]}", fontsize=10)
        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

    # Use ASCII titles to avoid missing CJK glyphs on some matplotlib defaults.
    plot_grid(correct_idx, "CNN prediction examples (correct)", outdir / "cnn_examples_correct.png")
    plot_grid(wrong_idx, "CNN prediction examples (wrong)", outdir / "cnn_examples_wrong.png")

    # Confusion matrix
    cm = np.zeros((10, 10), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outdir / "cnn_confusion_matrix.png", dpi=170)
    plt.close(fig)

    summary = {
        "device": str(device),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "test_accuracy": float((y_true == y_pred).mean()),
        "num_correct_examples": int(min(args.num, len(correct_idx))),
        "num_wrong_examples": int(min(args.num, len(wrong_idx))),
    }
    (outdir / "cnn_visual_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
