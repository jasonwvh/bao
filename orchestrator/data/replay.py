from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_replay_dataset(path: str, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".csv":
        return _load_csv(p, max_rows=max_rows)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return _load_parquet(p, max_rows=max_rows)
    raise ValueError(f"Unsupported dataset format: {p.suffix}")


def _load_csv(path: Path, max_rows: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if max_rows is not None and idx >= max_rows:
                break
            rows.append(_row_to_flow(row, idx))
    return rows


def _load_parquet(path: Path, max_rows: Optional[int]) -> List[Dict[str, Any]]:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("Parquet loading requires pandas and parquet backend (pyarrow/fastparquet)") from exc

    df = pd.read_parquet(path)
    if max_rows is not None:
        df = df.head(max_rows)
    rows: List[Dict[str, Any]] = []
    for idx, rec in enumerate(df.to_dict(orient="records")):
        rows.append(_row_to_flow(rec, idx))
    return rows


def _row_to_flow(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    if "label" not in row:
        raise ValueError("Replay row must include 'label' column")

    flow_id = str(row.get("flow_id") or f"flow_{idx:07d}")
    timestamp = _to_float(row.get("timestamp"))
    label = int(float(row["label"]))

    features: Dict[str, float] = {}
    for k, v in row.items():
        if k in {"flow_id", "label", "timestamp"}:
            continue
        f = _to_float(v)
        if f is not None:
            features[k] = f

    return {
        "flow_id": flow_id,
        "timestamp": timestamp,
        "true_label": label,
        "flow_features": features,
    }
