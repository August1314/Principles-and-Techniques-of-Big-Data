from __future__ import annotations

import time
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from load_data import tokenize_text
from metrics_utils import classification_metrics

PAD_ID = 0
UNK_ID = 1


class SentimentRNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(x)
        _, (hidden, _) = self.lstm(embeddings)
        logits = self.classifier(hidden[-1])
        return logits.squeeze(1)


def build_vocab(texts: list[str], vocab_size: int = 20_000) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize_text(text))

    most_common = counter.most_common(max(vocab_size - 2, 0))
    vocab = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
    for idx, (token, _) in enumerate(most_common, start=2):
        vocab[token] = idx
    return vocab


def encode_texts(texts: list[str], vocab: dict[str, int], max_len: int = 300) -> torch.Tensor:
    encoded = torch.full((len(texts), max_len), PAD_ID, dtype=torch.long)
    for i, text in enumerate(texts):
        token_ids = [vocab.get(token, UNK_ID) for token in tokenize_text(text)[:max_len]]
        if token_ids:
            encoded[i, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
    return encoded


def _to_loader(
    inputs: torch.Tensor, labels: list[int], batch_size: int, shuffle: bool
) -> DataLoader:
    y = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(inputs, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_eval_rnn(
    train_texts: list[str],
    y_train: list[int],
    test_texts: list[str],
    y_test: list[int],
    vocab_size: int = 20_000,
    max_len: int = 300,
    batch_size: int = 128,
    epochs: int = 3,
    learning_rate: float = 1e-3,
) -> dict[str, float | str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = build_vocab(train_texts, vocab_size=vocab_size)
    X_train = encode_texts(train_texts, vocab=vocab, max_len=max_len)
    X_test = encode_texts(test_texts, vocab=vocab, max_len=max_len)

    train_loader = _to_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    test_loader = _to_loader(X_test, y_test, batch_size=batch_size, shuffle=False)

    model = SentimentRNN(vocab_size=len(vocab)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_train = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
    train_seconds = time.perf_counter() - start_train

    model.eval()
    preds: list[int] = []
    start_infer = time.perf_counter()
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            batch_preds = (torch.sigmoid(logits) >= 0.5).long().cpu().tolist()
            preds.extend(batch_preds)
    infer_seconds = time.perf_counter() - start_infer

    metrics = classification_metrics(y_test, preds)
    return {
        "model": f"RNN-LSTM(Embedding, device={device.type})",
        **metrics,
        "train_seconds": float(train_seconds),
        "infer_seconds": float(infer_seconds),
    }
