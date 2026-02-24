from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class DecisionCosts:
    c_fn: float
    c_fp: float
    c_h: float


def expected_action_costs(p_mal: float, costs: DecisionCosts) -> Dict[str, float]:
    p = max(0.0, min(1.0, float(p_mal)))
    return {
        "accept": p * float(costs.c_fn),
        "reject": (1.0 - p) * float(costs.c_fp),
        "defer": float(costs.c_h),
    }


def select_expected_cost_action(p_mal: float, costs: DecisionCosts) -> tuple[str, Dict[str, float]]:
    action_costs = expected_action_costs(p_mal, costs)
    action = min(action_costs, key=action_costs.get)
    return action, action_costs


def min_expected_action_cost(p_mal: float, costs: DecisionCosts) -> float:
    return min(expected_action_costs(p_mal, costs).values())


def perfect_information_cost(p_mal: float, costs: DecisionCosts) -> float:
    p = max(0.0, min(1.0, float(p_mal)))
    # True state s=1 (malicious): reject is optimal with zero cost.
    min_if_attack = min(float(costs.c_fn), 0.0, float(costs.c_h))
    # True state s=0 (benign): accept is optimal with zero cost.
    min_if_benign = min(0.0, float(costs.c_fp), float(costs.c_h))
    return p * min_if_attack + (1.0 - p) * min_if_benign


def approximate_voi(p_mal: float, costs: DecisionCosts, rho: float) -> float:
    rho_clipped = max(0.0, min(1.0, float(rho)))
    current = min_expected_action_cost(p_mal, costs)
    perfect = perfect_information_cost(p_mal, costs)
    return rho_clipped * max(0.0, current - perfect)


def realized_action_cost(
    *,
    decision: Optional[str],
    prediction: Optional[int],
    true_label: Optional[int],
    costs: DecisionCosts,
) -> float:
    if true_label is None:
        return 0.0

    y = int(true_label)
    if decision is not None:
        d = str(decision).strip().lower()
        if d == "defer":
            return float(costs.c_h)
        if d == "accept":
            return float(costs.c_fn) if y == 1 else 0.0
        if d == "reject":
            return float(costs.c_fp) if y == 0 else 0.0

    if prediction is None:
        return 0.0
    p = int(prediction)
    if p == 0 and y == 1:
        return float(costs.c_fn)
    if p == 1 and y == 0:
        return float(costs.c_fp)
    return 0.0
