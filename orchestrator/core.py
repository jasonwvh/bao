from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from orchestrator.decision import DecisionCosts, min_expected_action_cost
from orchestrator.types import AgentRuntimeHandle


@dataclass(frozen=True)
class UtilizationTarget:
    agent_id: str
    min_rate: float
    max_rate: float
    penalty_under: float
    penalty_over: float


def resolve_first_agent(
    *,
    candidates: list[str],
    agent_handles: Dict[str, AgentRuntimeHandle],
    strategy: str,
    explicit_first_agent: Optional[str],
) -> Optional[str]:
    if not candidates:
        return None

    if str(strategy).strip().lower() == "explicit" and explicit_first_agent in candidates:
        return str(explicit_first_agent)

    best_agent = candidates[0]
    best_cost = float(agent_handles[best_agent].cost)
    for aid in candidates[1:]:
        cost = float(agent_handles[aid].cost)
        if cost < best_cost:
            best_agent = aid
            best_cost = cost
    return best_agent


def order_candidates(candidates: Iterable[str], first_agent: Optional[str]) -> list[str]:
    ordered = list(candidates)
    if first_agent is None or first_agent not in ordered:
        return ordered
    return [first_agent] + [aid for aid in ordered if aid != first_agent]


def classify_from_probability(probability: float) -> str:
    return "reject" if float(probability) >= 0.5 else "accept"


def compute_utilization_rates(
    *,
    agent_calls: Dict[str, int],
    flows_processed: int,
) -> Dict[str, float]:
    denom = float(max(1, int(flows_processed)))
    return {aid: float(calls) / denom for aid, calls in agent_calls.items()}


def compute_utilization_penalty(
    *,
    agent_id: str,
    agent_calls: Dict[str, int],
    flows_processed: int,
    targets: Dict[str, UtilizationTarget],
    warmup_flows: int,
) -> float:
    if int(flows_processed) < int(max(0, warmup_flows)):
        return 0.0

    target = targets.get(str(agent_id))
    if target is None:
        return 0.0

    rate = float(agent_calls.get(agent_id, 0)) / float(max(1, int(flows_processed)))
    under = max(0.0, float(target.min_rate) - rate)
    over = max(0.0, rate - float(target.max_rate))
    return (under * float(target.penalty_under)) + (over * float(target.penalty_over))


def build_utilization_penalties(
    *,
    candidate_agents: Iterable[str],
    agent_calls: Dict[str, int],
    flows_processed: int,
    targets: Dict[str, UtilizationTarget],
    warmup_flows: int,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for aid in candidate_agents:
        out[str(aid)] = compute_utilization_penalty(
            agent_id=str(aid),
            agent_calls=agent_calls,
            flows_processed=flows_processed,
            targets=targets,
            warmup_flows=warmup_flows,
        )
    return out


def apply_fusion_update(
    *,
    belief: Any,
    belief_manager: Any,
    agent_output: Dict[str, Any],
    agent_id: str,
    p_agent: float,
    queried_probabilities: Dict[str, float],
    fusion_method: str,
    update_mode: str,
    agent_weight: float,
    eps: float,
    likelihood_sanity_gate: bool,
    decision_costs: DecisionCosts,
    min_cost_fn: Callable[[float, DecisionCosts], float] = min_expected_action_cost,
) -> Tuple[Dict[str, float], str]:
    method = str(fusion_method).strip().lower()

    if method == "logit_pool":
        updated = belief.update_from_agent_output(
            agent_output=agent_output,
            agent_id=agent_id,
            update_mode=update_mode,
            weight=agent_weight,
            eps=eps,
            likelihood_sanity_gate=likelihood_sanity_gate,
        )
        return updated, "logit_pool"

    if method == "handoff_latest":
        belief.set_compromise_prob(p_agent)
        belief.var = max(1e-4, min(4.0, 1.0 / max(agent_weight, eps)))
        return {
            "mu": belief.mu,
            "var": belief.get_variance(),
            "compromise_prob": belief.get_compromise_prob(),
            "epistemic_uncertainty": belief.get_epistemic_uncertainty(),
        }, "handoff_latest"

    queried_probabilities[agent_id] = p_agent
    best_agent = agent_id
    best_proxy = float("inf")
    best_prob = p_agent
    for aid, prob in queried_probabilities.items():
        reliability = max(eps, float(belief_manager.get_global_reliability(aid)))
        proxy_cost = float(min_cost_fn(prob, decision_costs)) / reliability
        if proxy_cost < best_proxy:
            best_proxy = proxy_cost
            best_agent = aid
            best_prob = prob

    belief.set_compromise_prob(best_prob)
    belief.var = max(1e-4, min(4.0, 1.0 / max(agent_weight, eps)))
    return {
        "mu": belief.mu,
        "var": belief.get_variance(),
        "compromise_prob": belief.get_compromise_prob(),
        "epistemic_uncertainty": belief.get_epistemic_uncertainty(),
    }, f"utility_select:{best_agent}"
