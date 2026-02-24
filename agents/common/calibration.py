from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np


def _clip_probability(p: np.ndarray | float, lo: float, hi: float) -> np.ndarray | float:
    return np.clip(p, lo, hi)


def fit_logistic_calibrator(scores: np.ndarray, labels: np.ndarray, seed: int) -> Dict[str, float]:
    from sklearn.linear_model import LogisticRegression

    y = labels.astype(int).reshape(-1)
    x = scores.astype(np.float64).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return {"coef": 0.0, "intercept": 0.0, "is_fitted": 0.0}

    model = LogisticRegression(max_iter=2000, random_state=int(seed))
    model.fit(x, y)
    return {
        "coef": float(model.coef_[0][0]),
        "intercept": float(model.intercept_[0]),
        "is_fitted": 1.0,
    }


def logistic_probability(
    scores: np.ndarray | float,
    calibrator: Dict[str, float] | object | None,
    clip_lo: float = 0.001,
    clip_hi: float = 0.999,
) -> np.ndarray | float:
    # Backward compatibility for legacy payloads that stored sklearn calibrators.
    if calibrator is not None and hasattr(calibrator, "predict_proba"):
        x = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        p = calibrator.predict_proba(x)[:, 1]
        p = _clip_probability(p, float(clip_lo), float(clip_hi))
        if np.isscalar(scores):
            return float(p[0])
        return p

    if calibrator is None:
        c = {"coef": 1.0, "intercept": 0.0}
    else:
        c = calibrator

    coef = float(c.get("coef", 1.0))
    intercept = float(c.get("intercept", 0.0))
    z = coef * np.asarray(scores, dtype=np.float64) + intercept
    p = 1.0 / (1.0 + np.exp(-z))
    p = _clip_probability(p, float(clip_lo), float(clip_hi))
    if np.isscalar(scores):
        return float(p)
    return p


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = y_true.astype(int).reshape(-1)
    p = y_pred.astype(int).reshape(-1)

    pos = max(1, int((y == 1).sum()))
    neg = max(1, int((y == 0).sum()))
    tpr = float(((p == 1) & (y == 1)).sum()) / float(pos)
    tnr = float(((p == 0) & (y == 0)).sum()) / float(neg)
    return 0.5 * (tpr + tnr)


def select_probability_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    quantiles: Iterable[float] = np.linspace(0.01, 0.99, 199),
) -> Tuple[float, float]:
    probs = probabilities.astype(np.float64).reshape(-1)
    y = labels.astype(int).reshape(-1)
    candidates = np.quantile(probs, list(quantiles))
    best_thr = 0.5
    best_score = -1.0

    for thr in candidates:
        pred = (probs >= float(thr)).astype(int)
        ba = balanced_accuracy(y, pred)
        if ba > best_score:
            best_score = ba
            best_thr = float(thr)

    return best_thr, best_score


def entropy_from_probability(p: float) -> float:
    p_clip = float(np.clip(p, 1e-9, 1.0 - 1e-9))
    return -(p_clip * math.log(p_clip) + (1.0 - p_clip) * math.log(1.0 - p_clip))


def align_probability_threshold(probability: float, threshold_probability: float) -> float:
    """
    Re-center calibrated probabilities so the selected operating threshold maps to 0.5.
    """
    p = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    t = float(np.clip(threshold_probability, 1e-6, 1.0 - 1e-6))
    logit_p = math.log(p / (1.0 - p))
    logit_t = math.log(t / (1.0 - t))
    centered = 1.0 / (1.0 + math.exp(-(logit_p - logit_t)))
    return float(np.clip(centered, 1e-6, 1.0 - 1e-6))
