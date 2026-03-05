from __future__ import annotations

import math
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


def select_decision(p_mal: float, costs: DecisionCosts) -> tuple[str, Dict[str, float]]:
    action_costs = expected_action_costs(p_mal, costs)
    action = min(action_costs, key=action_costs.get)
    return action, action_costs


def min_expected_action_cost(p_mal: float, costs: DecisionCosts) -> float:
    return min(expected_action_costs(p_mal, costs).values())


def perfect_information_cost(p_mal: float, costs: DecisionCosts) -> float:
    p = max(0.0, min(1.0, float(p_mal)))
    min_if_attack = min(float(costs.c_fn), 0.0, float(costs.c_h))
    min_if_benign = min(0.0, float(costs.c_fp), float(costs.c_h))
    return p * min_if_attack + (1.0 - p) * min_if_benign


def approximate_voi(p_mal: float, costs: DecisionCosts, rho: float) -> float:
    rho_clipped = max(0.0, min(1.0, float(rho)))
    current = min_expected_action_cost(p_mal, costs)
    perfect = perfect_information_cost(p_mal, costs)
    return rho_clipped * max(0.0, current - perfect)


def expected_cost_reduction(
    *,
    p_mal: float,
    costs: DecisionCosts,
    reliability: float,
    epistemic_uncertainty: float,
    rho: float = 1.0,
) -> float:
    p = max(0.0, min(1.0, float(p_mal)))
    rel = max(0.0, min(1.0, float(reliability)))
    u = max(0.0, min(1.0, float(epistemic_uncertainty)))
    rho_clipped = max(0.0, min(1.0, float(rho)))
    current = min_expected_action_cost(p, costs)
    perfect = perfect_information_cost(p, costs)
    max_reduction = max(0.0, current - perfect)
    return rho_clipped * rel * u * max_reduction


def realized_action_cost(
    *,
    decision: Optional[str],
    true_label: Optional[int],
    costs: DecisionCosts,
) -> float:
    if true_label is None:
        return 0.0

    y = int(true_label)
    d = str(decision or "").strip().lower()
    if d == "defer":
        return float(costs.c_h)
    if d == "accept":
        return float(costs.c_fn) if y == 1 else 0.0
    if d == "reject":
        return float(costs.c_fp) if y == 0 else 0.0
    return 0.0


def probability_to_prediction(p_mal: float) -> int:
    return 1 if float(p_mal) >= 0.5 else 0


def entropy_nats(p_mal: float) -> float:
    p = max(1e-9, min(1.0 - 1e-9, float(p_mal)))
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))
