from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "raw" / "cmd_sample.csv"
OUT_JSONL = ROOT / "data" / "qa.jsonl"


def main() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"找不到原始数据：{RAW_CSV}")

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_CSV, encoding="gb18030")
    # 统一字段名并去空
    df = df.rename(
        columns={
            "department": "department",
            "title": "title",
            "ask": "ask",
            "answer": "answer",
        }
    )
    df = df[["department", "title", "ask", "answer"]].dropna()

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            clean = {k: str(v).strip() for k, v in row.items()}
            if not (clean["title"] and clean["ask"] and clean["answer"]):
                continue
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    print(f"Wrote {len(df):,} rows -> {OUT_JSONL}")


if __name__ == "__main__":
    main()

