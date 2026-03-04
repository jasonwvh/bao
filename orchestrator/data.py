from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd

from agents.common.preprocessing import UNSWPreprocessor, schema_from_json, transform_row


class ReplayRow(TypedDict):
    flow_id: str
    timestamp: Optional[float]
    true_label: int
    flow_features: Dict[str, Any]


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def load_replay_dataset(path: str, max_rows: Optional[int] = None) -> List[ReplayRow]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".csv":
        frame = pd.read_csv(p)
    elif p.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported dataset format: {p.suffix}")

    if max_rows is not None and int(max_rows) > 0:
        frame = frame.head(int(max_rows))

    if "label" not in frame.columns:
        raise ValueError("Replay dataset must include a 'label' column")

    rows: List[ReplayRow] = []
    records = frame.to_dict(orient="records")
    for idx, rec in enumerate(records):
        flow_id = str(rec.get("flow_id") or f"flow_{idx:07d}")
        timestamp = _to_float(rec.get("timestamp"))
        label = int(float(rec["label"]))

        features: Dict[str, Any] = {}
        for key, val in rec.items():
            if key in {"flow_id", "label", "timestamp", "id", "attack_cat"}:
                continue
            f = _to_float(val)
            if f is not None:
                features[key] = f
                continue
            if val is None:
                continue
            s = str(val).strip()
            if s:
                features[key] = s

        rows.append(
            ReplayRow(
                flow_id=flow_id,
                timestamp=timestamp,
                true_label=label,
                flow_features=features,
            )
        )
    return rows


class DataAdapter:
    def __init__(self, schema_path: Optional[str | Path] = None):
        self.schema_path = Path(schema_path).resolve() if schema_path else None
        self.preprocessor: Optional[UNSWPreprocessor] = None

        if self.schema_path and self.schema_path.exists():
            payload = schema_from_json(self.schema_path)
            pp_payload = payload.get("preprocessor")
            if isinstance(pp_payload, dict):
                self.preprocessor = UNSWPreprocessor.from_dict(pp_payload)

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def transform(self, flow_features: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(flow_features)

        spkts = self._to_float(out.get("spkts", 0.0))
        dpkts = self._to_float(out.get("dpkts", 0.0))
        sbytes = self._to_float(out.get("sbytes", 0.0))
        dbytes = self._to_float(out.get("dbytes", 0.0))
        dur = self._to_float(out.get("dur", 0.0))

        out.setdefault("packet_count", spkts + dpkts)
        out.setdefault("byte_count", sbytes + dbytes)
        out.setdefault("flow_duration", dur)

        if self.preprocessor is None:
            return out

        n, c = transform_row(self.preprocessor, out)
        for i, col in enumerate(self.preprocessor.numeric_cols):
            out[f"pp_num_{col}"] = float(n[i])
        for i, col in enumerate(self.preprocessor.categorical_cols):
            out[f"pp_cat_{col}"] = int(c[i])
        return out
