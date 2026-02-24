from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

EXCLUDE_COLUMNS = {"id", "label", "attack_cat", "flow_id", "timestamp"}
DEFAULT_CATEGORICAL = ["proto", "service", "state"]


@dataclass
class UNSWPreprocessor:
    numeric_cols: List[str]
    categorical_cols: List[str]
    vocabularies: Dict[str, Dict[str, int]]
    log1p_cols: List[str]
    medians: Dict[str, float]
    iqrs: Dict[str, float]
    iqr_floor: float = 1.0
    clip_min: float = -15.0
    clip_max: float = 15.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "vocabularies": self.vocabularies,
            "log1p_cols": self.log1p_cols,
            "medians": self.medians,
            "iqrs": self.iqrs,
            "iqr_floor": self.iqr_floor,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "UNSWPreprocessor":
        return cls(
            numeric_cols=list(payload.get("numeric_cols", [])),
            categorical_cols=list(payload.get("categorical_cols", [])),
            vocabularies={k: dict(v) for k, v in payload.get("vocabularies", {}).items()},
            log1p_cols=list(payload.get("log1p_cols", [])),
            medians={k: float(v) for k, v in payload.get("medians", {}).items()},
            iqrs={k: float(v) for k, v in payload.get("iqrs", {}).items()},
            iqr_floor=float(payload.get("iqr_floor", 1.0)),
            clip_min=float(payload.get("clip_min", -15.0)),
            clip_max=float(payload.get("clip_max", 15.0)),
        )


def normalize_columns(cols: List[str]) -> List[str]:
    return [str(c).replace("\ufeff", "").strip() for c in cols]


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = normalize_columns(list(df.columns))
    return df


def fit_preprocessor(
    df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
    iqr_floor: float = 1.0,
    clip_min: float = -15.0,
    clip_max: float = 15.0,
) -> UNSWPreprocessor:
    c_cols = categorical_cols or [c for c in DEFAULT_CATEGORICAL if c in df.columns]

    numeric_cols: List[str] = []
    for col in df.columns:
        if col in EXCLUDE_COLUMNS or col in c_cols:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() > 0:
            numeric_cols.append(col)

    vocabularies: Dict[str, Dict[str, int]] = {}
    for col in c_cols:
        if col not in df.columns:
            continue
        vals = sorted({str(v).strip() for v in df[col].fillna("UNK").astype(str)})
        vocab = {"<UNK>": 0}
        for i, v in enumerate(vals, start=1):
            if v not in vocab:
                vocab[v] = i
        vocabularies[col] = vocab

    medians: Dict[str, float] = {}
    iqrs: Dict[str, float] = {}
    log1p_cols: List[str] = []

    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32")
        if float(s.min()) >= 0.0:
            log1p_cols.append(col)
            s = np.log1p(s)
        q1, q3 = np.percentile(s, [25, 75])
        iqr = float(max(q3 - q1, float(iqr_floor)))
        medians[col] = float(np.median(s))
        iqrs[col] = iqr

    return UNSWPreprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=[c for c in c_cols if c in df.columns],
        vocabularies=vocabularies,
        log1p_cols=log1p_cols,
        medians=medians,
        iqrs=iqrs,
        iqr_floor=float(iqr_floor),
        clip_min=float(clip_min),
        clip_max=float(clip_max),
    )


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def transform_row(pre: UNSWPreprocessor, row: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    n = np.zeros(len(pre.numeric_cols), dtype=np.float32)
    c = np.zeros(len(pre.categorical_cols), dtype=np.int64)

    for i, col in enumerate(pre.numeric_cols):
        x = _to_float(row.get(col, 0.0))
        if col in pre.log1p_cols and x >= 0.0:
            x = float(np.log1p(x))
        x = (x - pre.medians.get(col, 0.0)) / max(pre.iqrs.get(col, 1.0), 1e-6)
        x = float(np.clip(x, pre.clip_min, pre.clip_max))
        n[i] = x

    for i, col in enumerate(pre.categorical_cols):
        vocab = pre.vocabularies.get(col, {"<UNK>": 0})
        token = str(row.get(col, "<UNK>")).strip()
        c[i] = int(vocab.get(token, 0))

    return n, c


def transform_frame(pre: UNSWPreprocessor, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    num = np.zeros((len(df), len(pre.numeric_cols)), dtype=np.float32)
    cat = np.zeros((len(df), len(pre.categorical_cols)), dtype=np.int64)

    for idx, rec in enumerate(df.to_dict(orient="records")):
        n, c = transform_row(pre, rec)
        num[idx] = n
        cat[idx] = c

    return num, cat


def build_sequences(
    num: np.ndarray,
    cat: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if window_size <= 1:
        return num[:, None, :], cat[:, None, :], labels.astype(np.int64)

    xs_num: List[np.ndarray] = []
    xs_cat: List[np.ndarray] = []
    ys: List[int] = []

    step = max(1, stride)
    for start in range(0, max(1, len(num) - window_size + 1), step):
        end = start + window_size
        if end > len(num):
            break
        xs_num.append(num[start:end])
        xs_cat.append(cat[start:end])
        ys.append(int(labels[end - 1]))

    if not xs_num:
        return np.zeros((0, window_size, num.shape[1]), dtype=np.float32), np.zeros(
            (0, window_size, cat.shape[1]), dtype=np.int64
        ), np.zeros((0,), dtype=np.int64)

    return np.stack(xs_num), np.stack(xs_cat), np.array(ys, dtype=np.int64)


def schema_to_json(pre: UNSWPreprocessor, out_path: str | Path, extra: Optional[Dict[str, Any]] = None) -> None:
    payload = {"version": "v2", "preprocessor": pre.to_dict()}
    if extra:
        payload.update(extra)
    Path(out_path).write_text(json.dumps(payload, indent=2))


def schema_from_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())
