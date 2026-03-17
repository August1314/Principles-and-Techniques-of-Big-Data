import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out", type=str, default="outputs/cnn_metrics.json")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-size", type=int, default=60000)
    parser.add_argument("--test-size", type=int, default=10000)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, transforms
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "torch/torchvision not available. Install requirements.txt first."
        ) from e

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=tfm)

    if args.train_size < len(train_ds):
        train_ds = Subset(train_ds, list(range(args.train_size)))
    if args.test_size < len(test_ds):
        test_ds = Subset(test_ds, list(range(args.test_size)))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
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

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def eval_accuracy() -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.shape[0])
        return correct / max(total, 1)

    t0 = time.perf_counter()
    epoch_train_loss = []
    epoch_test_acc = []

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        num_batches = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            num_batches += 1

        avg_loss = running_loss / max(num_batches, 1)
        acc = float(eval_accuracy())
        epoch_train_loss.append(avg_loss)
        epoch_test_acc.append(acc)
        model.train()
    train_seconds = time.perf_counter() - t0

    # Time a final evaluation pass for fair "evaluation seconds" reporting.
    t1 = time.perf_counter()
    acc = float(eval_accuracy())
    infer_seconds = time.perf_counter() - t1

    # Training curve
    curve_path = out_path.parent / "cnn_training_curve.png"
    if epoch_train_loss:
        epochs = list(range(1, len(epoch_train_loss) + 1))
        fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
        ax1.plot(epochs, epoch_train_loss, color="#d62728", marker="o", label="Train loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color="#d62728")
        ax1.tick_params(axis="y", labelcolor="#d62728")

        ax2 = ax1.twinx()
        ax2.plot(epochs, epoch_test_acc, color="#1f77b4", marker="s", label="Test accuracy")
        ax2.set_ylabel("Accuracy", color="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#1f77b4")
        ax2.set_ylim(0.0, 1.0)

        fig.suptitle("CNN training curve (MNIST)")
        fig.tight_layout()
        fig.savefig(curve_path, dpi=180)
        plt.close(fig)

    metrics = {
        "method": "SimpleCNN",
        "device": str(device),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "train_size": int(len(train_ds)),
        "test_size": int(len(test_ds)),
        "accuracy": acc,
        "train_seconds": float(train_seconds),
        "eval_seconds": float(infer_seconds),
        "eval_ms_per_image": float(infer_seconds * 1000.0 / len(test_ds)),
        "epoch_train_loss": [float(x) for x in epoch_train_loss],
        "epoch_test_acc": [float(x) for x in epoch_test_acc],
        "training_curve_png": str(curve_path.name),
    }

    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved: {out_path}")
    if curve_path.exists():
        print(f"Saved: {curve_path}")


if __name__ == "__main__":
    main()
