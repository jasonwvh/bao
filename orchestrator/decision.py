from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


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


def perfect_information_cost(p_mal: float, costs: DecisionCosts) -> float:
    p = max(0.0, min(1.0, float(p_mal)))
    # True state s=1 (malicious): reject is optimal with zero cost.
    min_if_attack = min(float(costs.c_fn), 0.0, float(costs.c_h))
    # True state s=0 (benign): accept is optimal with zero cost.
    min_if_benign = min(0.0, float(costs.c_fp), float(costs.c_h))
    return p * min_if_attack + (1.0 - p) * min_if_benign


def approximate_voi(p_mal: float, costs: DecisionCosts, rho: float) -> float:
    rho_clipped = max(0.0, min(1.0, float(rho)))
    current = min(expected_action_costs(p_mal, costs).values())
    perfect = perfect_information_cost(p_mal, costs)
    return rho_clipped * max(0.0, current - perfect)
