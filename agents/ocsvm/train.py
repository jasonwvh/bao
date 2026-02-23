from __future__ import annotations

import argparse
import os
import sys
# When running this script directly, ensure the project root is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import OneClassSVM

from agents.common.preprocessing import fit_preprocessor, load_csv, transform_frame


def _vectorize(num: np.ndarray, cat: np.ndarray, cat_cardinalities: list[int]) -> np.ndarray:
    num = num.astype(np.float32)
    if cat.size == 0 or len(cat_cardinalities) == 0:
        return num

    n = cat.shape[0]
    total_cat_dim = int(sum(cat_cardinalities))
    cat_oh = np.zeros((n, total_cat_dim), dtype=np.float32)

    offset = 0
    for i, card in enumerate(cat_cardinalities):
        idx = np.clip(cat[:, i].astype(np.int64), 0, max(card - 1, 0))
        rows = np.arange(n)
        cat_oh[rows, offset + idx] = 1.0
        offset += card

    return np.concatenate([num, cat_oh], axis=1)


def train(dataset: Path, output: Path, seed: int, max_train_normal: int) -> None:
    df = load_csv(dataset)
    if "id" in df.columns:
        df = df.sort_values("id").reset_index(drop=True)

    y = df["label"].astype(int).to_numpy()
    pre = fit_preprocessor(df)
    num, cat = transform_frame(pre, df)
    cat_cardinalities = [len(pre.vocabularies[c]) for c in pre.categorical_cols]
    x_all = _vectorize(num, cat, cat_cardinalities)

    x_train = x_all[y == 0]
    if len(x_train) == 0:
        raise RuntimeError("No benign rows to train One-Class SVM")
    if len(x_train) > max_train_normal:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x_train), size=max_train_normal, replace=False)
        x_train = x_train[idx]

    grid_nu = [0.01, 0.03, 0.05, 0.1]
    grid_gamma = ["scale", "auto", 0.01, 0.05]

    best_model = None
    best_f1 = -1.0
    best_scores = None
    best_cfg = None

    for nu in grid_nu:
        for gamma in grid_gamma:
            model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
            model.fit(x_train)
            scores = -model.decision_function(x_all)

            thr = float(np.percentile(scores[y == 0], 95.0)) if np.any(y == 0) else float(np.percentile(scores, 95.0))
            preds = (scores >= thr).astype(int)
            f1 = float(f1_score(y, preds, zero_division=0))

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_scores = scores
                best_cfg = {"nu": nu, "gamma": gamma, "threshold": thr}

    if best_model is None or best_scores is None or best_cfg is None:
        raise RuntimeError("Failed to fit One-Class SVM")

    calibrator = None
    if len(np.unique(y)) > 1:
        calibrator = LogisticRegression(max_iter=1000, random_state=seed)
        calibrator.fit(best_scores.reshape(-1, 1), y)

    benign = best_scores[y == 0] if np.any(y == 0) else best_scores
    malicious = best_scores[y == 1] if np.any(y == 1) else best_scores

    payload = {
        "model": best_model,
        "calibrator": calibrator,
        "preprocessor": pre.to_dict(),
        "cat_cardinalities": cat_cardinalities,
        "score_stats": {
            "mean": float(np.mean(best_scores)),
            "std": float(np.std(best_scores) + 1e-9),
            "p95": float(np.percentile(best_scores, 95.0)),
            "benign_mean": float(np.mean(benign)),
            "benign_std": float(np.std(benign) + 1e-9),
            "mal_mean": float(np.mean(malicious)),
            "mal_std": float(np.std(malicious) + 1e-9),
        },
        "model_config": best_cfg,
        "meta": {
            "dataset": str(dataset),
            "rows": int(len(df)),
            "features": int(x_all.shape[1]),
            "seed": int(seed),
            "best_f1": float(best_f1),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(payload, f)

    print(json.dumps({"saved_model": str(output), "rows": int(len(df)), "features": int(x_all.shape[1])}, indent=2))


def parse_args() -> argparse.Namespace:
    default_data = Path(__file__).resolve().parents[1] / "../data" / "UNSW_NB15_training-set.csv"
    default_model = Path(__file__).resolve().parent / "models" / "ocsvm.pkl"

    p = argparse.ArgumentParser(description="Train One-Class SVM on UNSW-NB15")
    p.add_argument("--dataset", default=str(default_data), help="Path to UNSW training CSV")
    p.add_argument("--output", default=str(default_model), help="Output .pkl model path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-normal", type=int, default=12000)
    return p.parse_args()


def main() -> None:
    print("Starting One-Class SVM training...")
    args = parse_args()
    train(
        dataset=Path(args.dataset),
        output=Path(args.output),
        seed=int(args.seed),
        max_train_normal=int(args.max_train_normal),
    )


if __name__ == "__main__":
    main()
