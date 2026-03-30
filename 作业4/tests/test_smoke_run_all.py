from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_all import run_pipeline

POS_TEXT = "This movie is amazing and touching. "
NEG_TEXT = "This movie is boring and terrible. "


def _build_fake_imdb(root: Path, n_per_class: int = 24) -> None:
    for split in ["train", "test"]:
        for label_dir, text in [("pos", POS_TEXT), ("neg", NEG_TEXT)]:
            dir_path = root / "aclImdb" / split / label_dir
            dir_path.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                (dir_path / f"{i}.txt").write_text(text + str(i), encoding="utf-8")


class TestRunAllSmoke(unittest.TestCase):
    def test_run_pipeline_smoke(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _build_fake_imdb(tmp_path, n_per_class=24)
            output_dir = tmp_path / "outputs"

            results = run_pipeline(
                data_dir=tmp_path,
                output_dir=output_dir,
                seed=42,
                sample_per_class=20,
                auto_download=False,
            )

            self.assertEqual(len(results), 3)
            self.assertTrue((output_dir / "metrics_summary.json").exists())
            self.assertTrue((output_dir / "metrics_summary.md").exists())


if __name__ == "__main__":
    unittest.main()
