from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


DEFAULT_BINS = 128
DEFAULT_SMOOTHING = 1.0


def _as_float_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    return np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64).reshape(-1)


def _normalize_histogram(counts: np.ndarray, smoothing: float) -> np.ndarray:
    vals = np.asarray(counts, dtype=np.float64).reshape(-1)
    alpha = max(0.0, float(smoothing))
    if vals.size == 0:
        return np.ones((1,), dtype=np.float64)
    total = float(np.sum(vals))
    if total <= 0.0 and alpha <= 0.0:
        return np.full(vals.shape, 1.0 / float(vals.size), dtype=np.float64)
    numer = vals + alpha
    denom = float(np.sum(vals)) + (alpha * float(vals.size))
    if denom <= 0.0:
        return np.full(vals.shape, 1.0 / float(vals.size), dtype=np.float64)
    return numer / denom


def build_class_conditional_likelihood_model(
    *,
    scores: Iterable[float] | np.ndarray,
    labels: Iterable[int] | np.ndarray,
    bins: int = DEFAULT_BINS,
    smoothing: float = DEFAULT_SMOOTHING,
) -> Dict[str, Any]:
    vals = _as_float_array(scores)
    y = np.asarray(list(labels) if not isinstance(labels, np.ndarray) else labels, dtype=np.int64).reshape(-1)
    if vals.size == 0 or y.size == 0 or vals.size != y.size:
        return {
            "bin_edges": [0.0, 1.0],
            "p_obs_given_attack_bins": [0.5],
            "p_obs_given_clean_bins": [0.5],
            "bins": 1,
            "smoothing": float(smoothing),
            "attack_count": int(np.sum(y == 1)),
            "clean_count": int(np.sum(y == 0)),
        }

    num_bins = max(8, int(bins))
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6

    attack = vals[y == 1]
    clean = vals[y == 0]
    attack_hist, edges = np.histogram(attack if attack.size else np.asarray([lo], dtype=np.float64), bins=num_bins, range=(lo, hi))
    clean_hist, _ = np.histogram(clean if clean.size else np.asarray([hi], dtype=np.float64), bins=num_bins, range=(lo, hi))

    p_attack = _normalize_histogram(attack_hist.astype(np.float64), smoothing=float(smoothing))
    p_clean = _normalize_histogram(clean_hist.astype(np.float64), smoothing=float(smoothing))

    return {
        "bin_edges": [float(x) for x in edges.tolist()],
        "p_obs_given_attack_bins": [float(x) for x in p_attack.tolist()],
        "p_obs_given_clean_bins": [float(x) for x in p_clean.tolist()],
        "bins": int(num_bins),
        "smoothing": float(smoothing),
        "attack_count": int(attack.size),
        "clean_count": int(clean.size),
    }


def validate_likelihood_model(model: Dict[str, Any] | None) -> Optional[Dict[str, Any]]:
    if not isinstance(model, dict):
        return None
    edges = _as_float_array(model.get("bin_edges", []))
    attack = _as_float_array(model.get("p_obs_given_attack_bins", []))
    clean = _as_float_array(model.get("p_obs_given_clean_bins", []))
    if edges.size < 2 or attack.size == 0 or clean.size == 0 or attack.size != clean.size or attack.size != edges.size - 1:
        return None
    attack_sum = float(np.sum(attack))
    clean_sum = float(np.sum(clean))
    if attack_sum <= 0.0 or clean_sum <= 0.0:
        return None
    return {
        "bin_edges": [float(x) for x in edges.tolist()],
        "p_obs_given_attack_bins": [float(x) for x in (attack / attack_sum).tolist()],
        "p_obs_given_clean_bins": [float(x) for x in (clean / clean_sum).tolist()],
        "bins": int(attack.size),
        "smoothing": float(model.get("smoothing", DEFAULT_SMOOTHING)),
        "attack_count": int(model.get("attack_count", 0)),
        "clean_count": int(model.get("clean_count", 0)),
    }


def lookup_likelihoods(score: float, model: Dict[str, Any] | None, eps: float = 1e-9) -> Optional[Tuple[float, float, int]]:
    validated = validate_likelihood_model(model)
    if validated is None:
        return None
    edges = np.asarray(validated["bin_edges"], dtype=np.float64)
    attack = np.asarray(validated["p_obs_given_attack_bins"], dtype=np.float64)
    clean = np.asarray(validated["p_obs_given_clean_bins"], dtype=np.float64)
    idx = int(np.searchsorted(edges, float(score), side="right") - 1)
    idx = int(np.clip(idx, 0, len(attack) - 1))
    return (
        float(max(float(eps), attack[idx])),
        float(max(float(eps), clean[idx])),
        idx,
    )
