from __future__ import annotations

import random
import re
import tarfile
import urllib.request
from pathlib import Path

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
ARCHIVE_NAME = "aclImdb_v1.tar.gz"
TEXT_ENCODING = "utf-8"


_HTML_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z']+")


def clean_text(text: str) -> str:
    text = _HTML_RE.sub(" ", text)
    text = text.lower()
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def tokenize_text(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _download_archive(archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        return
    urllib.request.urlretrieve(IMDB_URL, archive_path)  # noqa: S310


def ensure_imdb_dataset(data_dir: Path) -> Path:
    dataset_root = data_dir / "aclImdb"
    if dataset_root.exists():
        return dataset_root

    archive_path = data_dir / ARCHIVE_NAME
    _download_archive(archive_path)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir)

    return dataset_root


def _read_text_files(directory: Path) -> list[str]:
    files = sorted(directory.glob("*.txt"))
    texts: list[str] = []
    for path in files:
        texts.append(path.read_text(encoding=TEXT_ENCODING, errors="ignore"))
    return texts


def load_split(dataset_root: Path, split: str) -> tuple[list[str], list[int]]:
    split_dir = dataset_root / split
    pos_texts = _read_text_files(split_dir / "pos")
    neg_texts = _read_text_files(split_dir / "neg")

    texts = [clean_text(text) for text in pos_texts + neg_texts]
    labels = [1] * len(pos_texts) + [0] * len(neg_texts)
    return texts, labels


def _subsample_per_class(
    texts: list[str], labels: list[int], sample_per_class: int, seed: int
) -> tuple[list[str], list[int]]:
    if sample_per_class <= 0:
        raise ValueError("sample_per_class must be positive")

    rng = random.Random(seed)
    class_indices: dict[int, list[int]] = {0: [], 1: []}
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    selected_indices: list[int] = []
    for label in [0, 1]:
        indices = class_indices[label]
        rng.shuffle(indices)
        selected_indices.extend(indices[:sample_per_class])

    rng.shuffle(selected_indices)
    sampled_texts = [texts[idx] for idx in selected_indices]
    sampled_labels = [labels[idx] for idx in selected_indices]
    return sampled_texts, sampled_labels


def load_imdb_dataset(
    data_dir: Path,
    seed: int = 42,
    sample_per_class: int | None = None,
    auto_download: bool = True,
) -> tuple[list[str], list[int], list[str], list[int]]:
    dataset_root = data_dir / "aclImdb"
    if auto_download:
        dataset_root = ensure_imdb_dataset(data_dir)
    elif not dataset_root.exists():
        raise FileNotFoundError(f"未找到数据集目录: {dataset_root}")

    train_texts, train_labels = load_split(dataset_root, "train")
    test_texts, test_labels = load_split(dataset_root, "test")

    if sample_per_class is not None:
        train_texts, train_labels = _subsample_per_class(
            train_texts, train_labels, sample_per_class=sample_per_class, seed=seed
        )
        test_texts, test_labels = _subsample_per_class(
            test_texts, test_labels, sample_per_class=sample_per_class, seed=seed + 1
        )

    return train_texts, train_labels, test_texts, test_labels
