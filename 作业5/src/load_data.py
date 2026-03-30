from __future__ import annotations

from pathlib import Path

import pandas as pd


COLUMN_MAP = {
    "date": "date",
    "datetime": "date",
    "trade_date": "date",
    "timestamp": "date",
    "close": "close",
    "adj close": "close",
    "adj_close": "close",
    "adjusted close": "close",
    "adjusted_close": "close",
}


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed: dict[str, str] = {}
    for column in frame.columns:
        key = str(column).strip().lower()
        normalized = COLUMN_MAP.get(key, key)
        if normalized == key and key.endswith(".close"):
            normalized = "close"
        renamed[column] = normalized
    return frame.rename(columns=renamed)


def load_stock_csv(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame = _normalize_columns(frame)
    if "date" not in frame.columns or "close" not in frame.columns:
        raise ValueError("CSV must contain date and close columns")

    result = frame[["date", "close"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result = result.dropna(subset=["date", "close"])
    result = result.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return result
