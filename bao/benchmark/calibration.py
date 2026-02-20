from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


def _fit_logit_scale(raw_scores: List[float], labels: List[int]) -> Tuple[float, float]:
    # Tiny Platt-like fit by 1D search over temperature and bias; avoids extra deps.
    import math

    if not raw_scores:
        return 1.0, 0.0

    best = (1.0, 0.0)
    best_loss = float("inf")

    for t in [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]:
        for b in [-0.6, -0.3, 0.0, 0.3, 0.6]:
            loss = 0.0
            for p, y in zip(raw_scores, labels):
                p = max(1e-6, min(1 - 1e-6, p))
                logit = math.log(p / (1 - p))
                q = 1.0 / (1.0 + math.exp(-(logit / t + b)))
                q = max(1e-6, min(1 - 1e-6, q))
                loss += -(y * math.log(q) + (1 - y) * math.log(1 - q))
            if loss < best_loss:
                best_loss = loss
                best = (t, b)

    return best


@dataclass
class SharedCalibrator:
    params: Dict[str, Tuple[float, float]]

    def calibrate(self, agent_name: str, raw_score: float) -> float:
        import math

        t, b = self.params.get(agent_name, (1.0, 0.0))
        p = max(1e-6, min(1 - 1e-6, raw_score))
        logit = math.log(p / (1 - p))
        q = 1.0 / (1.0 + math.exp(-(logit / t + b)))
        return max(0.001, min(0.999, q))


def fit_shared_calibrator(
    calibration_rows: Iterable[dict],
    agents: Dict[str, object],
    seed: int,
) -> SharedCalibrator:
    rows = list(calibration_rows)
    if not rows:
        return SharedCalibrator(params={})

    params: Dict[str, Tuple[float, float]] = {}
    labels = [int(r["label"]) for r in rows]

    for name, agent in agents.items():
        raw_scores = [agent.predict_proba(r["features"], seed=seed) for r in rows]
        params[name] = _fit_logit_scale(raw_scores, labels)

    return SharedCalibrator(params=params)
