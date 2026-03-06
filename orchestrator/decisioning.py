from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


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


def posterior_from_likelihood_ratio(
    *,
    p_mal: float,
    p_obs_given_attack: float,
    p_obs_given_clean: float,
    reliability_weight: float = 1.0,
    eps: float = 1e-9,
) -> float:
    p = max(float(eps), min(1.0 - float(eps), float(p_mal)))
    attack = max(float(eps), float(p_obs_given_attack))
    clean = max(float(eps), float(p_obs_given_clean))
    logit_prior = math.log(p / (1.0 - p))
    delta = float(reliability_weight) * math.log(attack / clean)
    posterior = 1.0 / (1.0 + math.exp(-(logit_prior + delta)))
    return max(float(eps), min(1.0 - float(eps), posterior))


def expected_cost_reduction_from_likelihood_model(
    *,
    p_mal: float,
    costs: DecisionCosts,
    likelihood_model: Dict[str, object],
    reliability_weight: float = 1.0,
    rho: float = 1.0,
    eps: float = 1e-9,
) -> float:
    attack_bins = np.asarray(likelihood_model.get("p_obs_given_attack_bins", []), dtype=np.float64).reshape(-1)
    clean_bins = np.asarray(likelihood_model.get("p_obs_given_clean_bins", []), dtype=np.float64).reshape(-1)
    if attack_bins.size == 0 or clean_bins.size == 0 or attack_bins.size != clean_bins.size:
        return 0.0

    attack_bins = np.maximum(float(eps), attack_bins)
    clean_bins = np.maximum(float(eps), clean_bins)
    attack_bins = attack_bins / float(np.sum(attack_bins))
    clean_bins = clean_bins / float(np.sum(clean_bins))

    p = max(float(eps), min(1.0 - float(eps), float(p_mal)))
    current = min_expected_action_cost(p, costs)
    future = 0.0
    for p_attack_bin, p_clean_bin in zip(attack_bins.tolist(), clean_bins.tolist()):
        p_obs = (p * float(p_attack_bin)) + ((1.0 - p) * float(p_clean_bin))
        posterior = posterior_from_likelihood_ratio(
            p_mal=p,
            p_obs_given_attack=float(p_attack_bin),
            p_obs_given_clean=float(p_clean_bin),
            reliability_weight=float(reliability_weight),
            eps=float(eps),
        )
        future += float(p_obs) * min_expected_action_cost(posterior, costs)
    reduction = max(0.0, current - future)
    return max(0.0, min(1.0, float(rho))) * reduction


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
