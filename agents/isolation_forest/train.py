from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest as SklearnIsolationForest

EXCLUDE_COLUMNS = {"id", "label", "attack_cat"}


def _normalize_columns(cols: list[str]) -> list[str]:
    return [str(c).replace("\ufeff", "").strip() for c in cols]


def _load_training_frame(dataset: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset)
    df.columns = _normalize_columns(list(df.columns))

    numeric_cols = []
    for col in df.columns:
        if col in EXCLUDE_COLUMNS:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() > 0:
            df[col] = s.fillna(0.0).astype("float32")
            numeric_cols.append(col)

    if not numeric_cols:
        raise RuntimeError("No numeric feature columns found in dataset")

    return df[numeric_cols]


def train(dataset: Path, output: Path, seed: int, n_estimators: int, contamination: str | float) -> None:
    x = _load_training_frame(dataset)
    model = SklearnIsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x)

    anomaly_scores = -model.score_samples(x)
    score_mean = float(np.mean(anomaly_scores))
    score_std = float(np.std(anomaly_scores) + 1e-9)
    score_p95 = float(np.percentile(anomaly_scores, 95.0))

    payload = {
        "feature_names": list(x.columns),
        "model": model,
        "score_mean": score_mean,
        "score_std": score_std,
        "score_p95": score_p95,
        "meta": {
            "dataset": str(dataset),
            "rows": int(x.shape[0]),
            "features": int(x.shape[1]),
            "seed": int(seed),
            "n_estimators": int(n_estimators),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(payload, f)

    print(json.dumps({"saved_model": str(output), "rows": int(x.shape[0]), "features": int(x.shape[1])}, indent=2))


def parse_args() -> argparse.Namespace:
    default_data = Path(__file__).resolve().parents[1] / "../data" / "UNSW_NB15_training-set.csv"
    default_model = Path(__file__).resolve().parent / "models" / "isolation_forest.pkl"

    p = argparse.ArgumentParser(description="Train IsolationForest on UNSW-NB15")
    p.add_argument("--dataset", default=str(default_data), help="Path to UNSW training CSV")
    p.add_argument("--output", default=str(default_model), help="Output .pkl model path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--contamination", default="auto", help="IsolationForest contamination")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    contamination: str | float
    if args.contamination == "auto":
        contamination = "auto"
    else:
        contamination = float(args.contamination)
    train(
        dataset=Path(args.dataset),
        output=Path(args.output),
        seed=int(args.seed),
        n_estimators=int(args.n_estimators),
        contamination=contamination,
    )


if __name__ == "__main__":
    main()
