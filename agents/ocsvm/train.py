from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM

from agents.common.calibration import fit_logistic_calibrator, logistic_probability, select_probability_threshold
from agents.common.preprocessing import fit_preprocessor, load_csv, transform_frame
from agents.common.training_config import load_agent_training_config


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


def train(dataset: Path, output: Path, seed: int, config_path: Path | None) -> None:
    cfg = load_agent_training_config(config_path)
    shared = dict(cfg.get("shared", {}))
    pre_cfg = dict(shared.get("preprocessing", {}))
    cal_cfg = dict(shared.get("calibration", {}))
    agent_cfg = dict(cfg.get("ocsvm", {}))

    val_fraction = float(shared.get("validation_fraction", 0.2))
    val_fraction = min(0.4, max(0.05, val_fraction))

    probability_clip = list(cal_cfg.get("probability_clip", [0.001, 0.999]))
    p_lo = float(probability_clip[0])
    p_hi = float(probability_clip[1])

    max_train_normal = int(agent_cfg.get("max_train_normal", 30000))
    nu_grid = list(agent_cfg.get("nu_grid", [0.01, 0.03, 0.05, 0.1]))
    gamma_grid = list(agent_cfg.get("gamma_grid", ["scale", "auto", 0.005, 0.01, 0.03]))

    df = load_csv(dataset)
    if "id" in df.columns:
        df = df.sort_values("id").reset_index(drop=True)

    y = df["label"].astype(int).to_numpy()
    train_idx, cal_idx = train_test_split(
        np.arange(len(df)),
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    cal_df = df.iloc[cal_idx].reset_index(drop=True)
    y_cal = cal_df["label"].astype(int).to_numpy()

    pre = fit_preprocessor(
        train_df,
        categorical_cols=list(pre_cfg.get("categorical_cols", ["proto", "service", "state"])),
        iqr_floor=float(pre_cfg.get("iqr_floor", 1.0)),
        clip_min=float(pre_cfg.get("clip_min", -15.0)),
        clip_max=float(pre_cfg.get("clip_max", 15.0)),
    )

    train_num, train_cat = transform_frame(pre, train_df)
    cal_num, cal_cat = transform_frame(pre, cal_df)
    cat_cardinalities = [len(pre.vocabularies[c]) for c in pre.categorical_cols]

    x_train = _vectorize(train_num, train_cat, cat_cardinalities)
    x_cal = _vectorize(cal_num, cal_cat, cat_cardinalities)

    y_train = train_df["label"].astype(int).to_numpy()
    x_train_normal = x_train[y_train == 0]
    if len(x_train_normal) == 0:
        raise RuntimeError("No benign rows to train One-Class SVM")

    if len(x_train_normal) > max_train_normal:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x_train_normal), size=max_train_normal, replace=False)
        x_train_normal = x_train_normal[idx]

    best_model = None
    best_calibrator = None
    best_threshold = 0.5
    best_ba = -1.0
    best_scores = None
    best_cfg = None

    for nu in nu_grid:
        for gamma in gamma_grid:
            model = OneClassSVM(kernel="rbf", nu=float(nu), gamma=gamma)
            model.fit(x_train_normal)
            scores_cal = -model.decision_function(x_cal)
            calibrator = fit_logistic_calibrator(scores_cal, y_cal, seed=seed)
            probs_cal = logistic_probability(scores_cal, calibrator, clip_lo=p_lo, clip_hi=p_hi)
            thr, ba = select_probability_threshold(np.asarray(probs_cal), y_cal)
            if ba > best_ba:
                best_ba = float(ba)
                best_model = model
                best_calibrator = calibrator
                best_threshold = float(thr)
                best_scores = scores_cal
                best_cfg = {
                    "nu": float(nu),
                    "gamma": gamma,
                    "balanced_accuracy": float(ba),
                }

    if best_model is None or best_scores is None or best_cfg is None:
        raise RuntimeError("Failed to fit One-Class SVM")

    benign = best_scores[y_cal == 0] if np.any(y_cal == 0) else best_scores
    malicious = best_scores[y_cal == 1] if np.any(y_cal == 1) else best_scores

    payload = {
        "model": best_model,
        "calibrator": best_calibrator,
        "threshold_probability": best_threshold,
        "probability_clip": [p_lo, p_hi],
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
            "train_rows": int(len(train_df)),
            "calibration_rows": int(len(cal_df)),
            "features": int(x_train.shape[1]),
            "seed": int(seed),
            "validation_fraction": float(val_fraction),
            "best_balanced_accuracy": float(best_ba),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        import pickle

        pickle.dump(payload, f)

    print(
        json.dumps(
            {
                "saved_model": str(output),
                "rows": int(len(df)),
                "features": int(x_train.shape[1]),
                "balanced_accuracy": float(best_ba),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    default_data = Path(__file__).resolve().parents[1] / "../data" / "UNSW_NB15_training-set.csv"
    default_model = Path(__file__).resolve().parent / "models" / "ocsvm.pkl"
    default_cfg = Path(__file__).resolve().parents[2] / "config" / "agent_training.yaml"

    p = argparse.ArgumentParser(description="Train One-Class SVM on UNSW-NB15")
    p.add_argument("--dataset", default=str(default_data), help="Path to UNSW training CSV")
    p.add_argument("--output", default=str(default_model), help="Output .pkl model path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default=str(default_cfg), help="Path to agent training YAML")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(
        dataset=Path(args.dataset),
        output=Path(args.output),
        seed=int(args.seed),
        config_path=Path(args.config),
    )


if __name__ == "__main__":
    main()
