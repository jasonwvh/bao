from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from orchestrator.state import SQLiteState


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float, eps: float) -> float:
    clipped = _clip(p, eps, 1.0 - eps)
    return math.log(clipped / (1.0 - clipped))


@dataclass
class BayesianBelief:
    flow_id: str
    mu: float
    var: float
    eps: float
    pooled_logit_sum: float
    pooled_weight_sum: float

    @classmethod
    def new(cls, flow_id: str, prior_attack_rate: float, eps: float) -> "BayesianBelief":
        p = _clip(float(prior_attack_rate), eps, 1.0 - eps)
        mu = _logit(p, eps)
        return cls(
            flow_id=flow_id,
            mu=mu,
            var=1.0,
            eps=eps,
            pooled_logit_sum=0.0,
            pooled_weight_sum=0.0,
        )

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BayesianBelief":
        return cls(
            flow_id=str(payload["flow_id"]),
            mu=float(payload.get("mu", 0.0)),
            var=float(payload.get("var", 1.0)),
            eps=float(payload.get("eps", 1e-6)),
            pooled_logit_sum=float(payload.get("pooled_logit_sum", 0.0)),
            pooled_weight_sum=float(payload.get("pooled_weight_sum", 0.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "mu": self.mu,
            "var": self.var,
            "eps": self.eps,
            "pooled_logit_sum": self.pooled_logit_sum,
            "pooled_weight_sum": self.pooled_weight_sum,
        }

    def probability(self) -> float:
        return _clip(_sigmoid(self.mu), self.eps, 1.0 - self.eps)

    def entropy(self) -> float:
        p = self.probability()
        return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))

    def update_from_agent_probability(self, p_mal: float, weight: float) -> float:
        w = max(self.eps, float(weight))
        p = _clip(float(p_mal), self.eps, 1.0 - self.eps)
        prev_weight = self.pooled_weight_sum
        logit_p = _logit(p, self.eps)
        self.pooled_logit_sum += w * logit_p
        self.pooled_weight_sum += w
        if prev_weight <= self.eps:
            pooled_logit = logit_p
        else:
            pooled_logit = self.pooled_logit_sum / max(self.pooled_weight_sum, self.eps)
        self.mu = pooled_logit
        self.var = _clip(1.0 / max(self.pooled_weight_sum, self.eps), 1e-4, 4.0)
        return self.probability()


class BeliefManager:
    def __init__(self, state: SQLiteState, eps: float = 1e-6):
        self.state = state
        self.eps = max(1e-9, float(eps))
        self.cache: Dict[str, BayesianBelief] = {}

    def get_or_create(self, flow_id: str, prior_attack_rate: float) -> BayesianBelief:
        if flow_id in self.cache:
            return self.cache[flow_id]
        saved = self.state.load_belief(flow_id)
        if isinstance(saved, dict):
            belief = BayesianBelief.from_dict(saved)
        else:
            belief = BayesianBelief.new(flow_id, prior_attack_rate, self.eps)
        self.cache[flow_id] = belief
        return belief

    def persist(self, flow_id: str) -> None:
        belief = self.cache.get(flow_id)
        if belief is None:
            return
        self.state.save_belief(flow_id, belief.to_dict())

    def update_global_reliability(self, agent_id: str, correct: bool) -> None:
        self.state.update_global_reliability(agent_id, correct)

    def get_global_reliability(self, agent_id: str) -> float:
        alpha, beta = self.state.get_global_reliability(agent_id)
        return float(alpha) / float(alpha + beta)
