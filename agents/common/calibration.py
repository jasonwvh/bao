from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

LN2 = math.log(2.0)


def _clip_probability(p: np.ndarray | float, lo: float, hi: float) -> np.ndarray | float:
    return np.clip(p, lo, hi)


def fit_logistic_calibrator(scores: np.ndarray, labels: np.ndarray, seed: int) -> Dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    y = labels.astype(int).reshape(-1)
    x = scores.astype(np.float64).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return {"type": "logistic", "coef": 0.0, "intercept": 0.0, "is_fitted": 0.0}

    model = LogisticRegression(max_iter=2000, random_state=int(seed))
    model.fit(x, y)
    return {
        "type": "logistic",
        "coef": float(model.coef_[0][0]),
        "intercept": float(model.intercept_[0]),
        "is_fitted": 1.0,
    }


def fit_isotonic_calibrator(scores: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    from sklearn.isotonic import IsotonicRegression

    y = labels.astype(int).reshape(-1)
    x = scores.astype(np.float64).reshape(-1)
    if len(np.unique(y)) < 2:
        return {"type": "isotonic", "x": [0.0, 1.0], "y": [0.5, 0.5], "is_fitted": 0.0}

    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(x, y)
    x_thr = np.asarray(getattr(model, "X_thresholds_", np.array([x.min(), x.max()])), dtype=np.float64)
    y_thr = np.asarray(getattr(model, "y_thresholds_", np.array([0.5, 0.5])), dtype=np.float64)
    if x_thr.size == 0 or y_thr.size == 0:
        x_thr = np.array([x.min(), x.max()], dtype=np.float64)
        y_thr = np.array([0.5, 0.5], dtype=np.float64)
    return {
        "type": "isotonic",
        "x": [float(v) for v in x_thr.tolist()],
        "y": [float(v) for v in y_thr.tolist()],
        "is_fitted": 1.0,
    }


def logistic_probability(
    scores: np.ndarray | float,
    calibrator: Dict[str, Any] | object | None,
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

    c = calibrator or {"type": "logistic", "coef": 1.0, "intercept": 0.0}
    c_type = str(c.get("type", "logistic")).strip().lower() if isinstance(c, dict) else "logistic"

    if c_type == "isotonic" and isinstance(c, dict):
        xs = np.asarray(c.get("x", [0.0, 1.0]), dtype=np.float64).reshape(-1)
        ys = np.asarray(c.get("y", [0.5, 0.5]), dtype=np.float64).reshape(-1)
        if xs.size < 2 or ys.size < 2:
            xs = np.asarray([0.0, 1.0], dtype=np.float64)
            ys = np.asarray([0.5, 0.5], dtype=np.float64)
        p = np.interp(np.asarray(scores, dtype=np.float64), xs, ys, left=float(ys[0]), right=float(ys[-1]))
    else:
        # Backward-compatible path for legacy calibrator payloads without "type".
        coef = float(c.get("coef", 1.0)) if isinstance(c, dict) else 1.0
        intercept = float(c.get("intercept", 0.0)) if isinstance(c, dict) else 0.0
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


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    p = probabilities.astype(np.float64).reshape(-1)
    y = labels.astype(int).reshape(-1)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    p = probabilities.astype(np.float64).reshape(-1)
    y = labels.astype(int).reshape(-1)
    if len(p) == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, max(2, int(bins) + 1))
    total = float(len(p))
    ece = 0.0
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == len(edges) - 2:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        ece += (float(np.sum(mask)) / total) * abs(acc - conf)
    return float(ece)


def fit_best_calibrator(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    seed: int,
    clip_lo: float,
    clip_hi: float,
) -> Dict[str, Any]:
    candidates: List[Tuple[str, Dict[str, Any]]] = [
        ("logistic", fit_logistic_calibrator(scores, labels, seed=seed)),
        ("isotonic", fit_isotonic_calibrator(scores, labels)),
    ]

    best: Dict[str, Any] | None = None
    diagnostics: List[Dict[str, Any]] = []

    for name, calibrator in candidates:
        probs = np.asarray(logistic_probability(scores, calibrator, clip_lo=clip_lo, clip_hi=clip_hi), dtype=np.float64)
        thr, ba = select_probability_threshold(probs, labels)
        ece = expected_calibration_error(probs, labels)
        brier = brier_score(probs, labels)
        row = {
            "name": name,
            "balanced_accuracy": float(ba),
            "ece": float(ece),
            "brier": float(brier),
            "threshold_probability": float(thr),
            "calibrator": calibrator,
        }
        diagnostics.append(row)

        if best is None:
            best = row
            continue

        ba_best = float(best["balanced_accuracy"])
        ece_best = float(best["ece"])
        brier_best = float(best["brier"])
        if (float(ba) > ba_best) or (
            abs(float(ba) - ba_best) <= 1e-9 and (float(ece) < ece_best or (abs(float(ece) - ece_best) <= 1e-12 and float(brier) < brier_best))
        ):
            best = row

    if best is None:
        fallback = {"type": "logistic", "coef": 0.0, "intercept": 0.0, "is_fitted": 0.0}
        probs = np.asarray(logistic_probability(scores, fallback, clip_lo=clip_lo, clip_hi=clip_hi), dtype=np.float64)
        thr, ba = select_probability_threshold(probs, labels)
        best = {
            "name": "logistic",
            "balanced_accuracy": float(ba),
            "ece": float(expected_calibration_error(probs, labels)),
            "brier": float(brier_score(probs, labels)),
            "threshold_probability": float(thr),
            "calibrator": fallback,
        }

    return {
        "calibrator": best["calibrator"],
        "selected": str(best["name"]),
        "threshold_probability": float(best["threshold_probability"]),
        "balanced_accuracy": float(best["balanced_accuracy"]),
        "ece": float(best["ece"]),
        "brier": float(best["brier"]),
        "diagnostics": [
            {
                "name": str(d["name"]),
                "balanced_accuracy": float(d["balanced_accuracy"]),
                "ece": float(d["ece"]),
                "brier": float(d["brier"]),
                "threshold_probability": float(d["threshold_probability"]),
            }
            for d in diagnostics
        ],
    }


def entropy_from_probability(p: float) -> float:
    p_clip = float(np.clip(p, 1e-9, 1.0 - 1e-9))
    return -(p_clip * math.log(p_clip) + (1.0 - p_clip) * math.log(1.0 - p_clip))


def class_uncertainty_from_probability(p: float) -> float:
    """Symmetric confidence gap mapped to [0, 1], highest at p=0.5."""
    p_clip = float(np.clip(p, 1e-9, 1.0 - 1e-9))
    return float(max(0.0, min(1.0, 1.0 - min(abs(2.0 * p_clip - 1.0), 1.0))))


def normalize_uncertainty(epistemic: float, aleatoric: float) -> Dict[str, float]:
    """Normalize uncertainty channels and enforce consistent total entropy semantics."""
    ep = float(np.clip(epistemic, 0.0, 1.0))
    al = float(np.clip(aleatoric, 0.0, LN2))
    total = float(max(al, ep * LN2))
    return {
        "epistemic": ep,
        "aleatoric": al,
        "total_entropy": total,
    }


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
