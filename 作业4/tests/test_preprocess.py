from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from load_data import clean_text, load_split


class TestPreprocess(unittest.TestCase):
    def test_clean_text_removes_html_and_spaces(self) -> None:
        raw = "<br />  This   MOVIE is GREAT!   "
        self.assertEqual(clean_text(raw), "this movie is great!")

    def test_load_split_label_mapping(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pos_dir = tmp_path / "train" / "pos"
            neg_dir = tmp_path / "train" / "neg"
            pos_dir.mkdir(parents=True)
            neg_dir.mkdir(parents=True)

            (pos_dir / "1.txt").write_text("good movie", encoding="utf-8")
            (neg_dir / "1.txt").write_text("bad movie", encoding="utf-8")

            texts, labels = load_split(tmp_path, "train")
            self.assertEqual(len(texts), 2)
            self.assertEqual(sorted(labels), [0, 1])


if __name__ == "__main__":
    unittest.main()
