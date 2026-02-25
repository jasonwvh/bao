from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from orchestrator.decision import DecisionCosts, expected_action_costs
from orchestrator.types import AgentRuntimeHandle


def _clip_probability(value: float, eps: float = 1e-6) -> float:
    return max(eps, min(1.0 - eps, float(value)))


def _min_expected_action_cost(p_mal: float, costs: DecisionCosts) -> float:
    return min(expected_action_costs(p_mal, costs).values())


@dataclass(frozen=True)
class RouterScore:
    agent_id: str
    expected_gain: float
    expected_gain_adjusted: float
    current_action_cost: float
    expected_action_cost_after: float
    query_cost: float
    utilization_penalty: float
    reliability: float
    estimated_candidate_prob: float
    estimated_p_after: float
    profile_source: str

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "agent_id": self.agent_id,
            "expected_gain": self.expected_gain,
            "expected_gain_adjusted": self.expected_gain_adjusted,
            "current_action_cost": self.current_action_cost,
            "expected_action_cost_after": self.expected_action_cost_after,
            "query_cost": self.query_cost,
            "utilization_penalty": self.utilization_penalty,
            "reliability": self.reliability,
            "estimated_candidate_prob": self.estimated_candidate_prob,
            "estimated_p_after": self.estimated_p_after,
            "profile_source": self.profile_source,
        }


class AdaptiveRouter:
    """Adaptive next-agent selector based on expected utility gain."""

    def __init__(
        self,
        *,
        decision_costs: DecisionCosts,
        profile_path: Optional[Path] = None,
        min_samples_per_bin: int = 20,
    ):
        self.decision_costs = decision_costs
        self.profile_path = profile_path
        self.min_samples_per_bin = max(1, int(min_samples_per_bin))
        self.profile: Dict[str, Any] = {}
        self._global_stats: Dict[str, Dict[str, float]] = {}
        self._pairwise_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}

        if profile_path is not None and profile_path.exists():
            self.profile = json.loads(profile_path.read_text())
            self._global_stats = dict(self.profile.get("global", {}) or {})
            self._pairwise_stats = dict(self.profile.get("pairwise", {}) or {})

    def _global_probability_and_accuracy(self, agent_id: str) -> Tuple[Optional[float], Optional[float]]:
        entry = self._global_stats.get(agent_id) or {}
        p = entry.get("mean_probability")
        a = entry.get("accuracy")
        out_p = None if p is None else _clip_probability(float(p))
        out_a = None if a is None else max(0.0, min(1.0, float(a)))
        return out_p, out_a

    def _pairwise_bin_lookup(
        self,
        *,
        source_agent: str,
        target_agent: str,
        source_probability: float,
    ) -> Tuple[Optional[float], Optional[float], str]:
        by_source = self._pairwise_stats.get(source_agent) or {}
        edge = by_source.get(target_agent) or {}
        bins = edge.get("bins") or []
        if not bins:
            return None, None, "none"

        p = _clip_probability(source_probability)
        for item in bins:
            lo = float(item.get("lo", 0.0))
            hi = float(item.get("hi", 1.0))
            if not (lo <= p < hi or (p == 1.0 and hi >= 1.0)):
                continue
            count = int(item.get("count", 0))
            if count < self.min_samples_per_bin:
                return None, None, "pairwise_low_count"
            mean_target_probability = item.get("mean_target_probability")
            target_accuracy = item.get("target_accuracy")
            if mean_target_probability is None:
                return None, None, "pairwise_empty"
            out_p = _clip_probability(float(mean_target_probability))
            out_a = None if target_accuracy is None else max(0.0, min(1.0, float(target_accuracy)))
            return out_p, out_a, "pairwise"
        return None, None, "pairwise_missing_bin"

    def _estimate_candidate_probability(
        self,
        *,
        source_agent: str,
        target_agent: str,
        source_probability: float,
    ) -> Tuple[float, Optional[float], str]:
        pair_prob, pair_acc, pair_src = self._pairwise_bin_lookup(
            source_agent=source_agent,
            target_agent=target_agent,
            source_probability=source_probability,
        )
        if pair_prob is not None:
            return pair_prob, pair_acc, pair_src

        global_prob, global_acc = self._global_probability_and_accuracy(target_agent)
        if global_prob is not None:
            return global_prob, global_acc, "global"

        return _clip_probability(source_probability), None, "fallback"

    def _resolve_reliability(
        self,
        *,
        agent_id: str,
        belief_manager: Any,
        profile_accuracy: Optional[float],
    ) -> float:
        backend_rel = 0.5
        try:
            backend_rel = float(belief_manager.get_global_reliability(agent_id))
        except Exception:
            backend_rel = 0.5
        backend_rel = max(0.0, min(1.0, backend_rel))

        if profile_accuracy is None:
            return backend_rel
        return max(0.0, min(1.0, 0.5 * backend_rel + 0.5 * float(profile_accuracy)))

    def score_candidates(
        self,
        *,
        current_probability: float,
        source_agent: str,
        candidate_agents: Iterable[str],
        agent_handles: Dict[str, AgentRuntimeHandle],
        belief_manager: Any,
        utilization_penalties: Optional[Dict[str, float]] = None,
    ) -> Dict[str, RouterScore]:
        p_current = _clip_probability(current_probability)
        current_cost = _min_expected_action_cost(p_current, self.decision_costs)
        scores: Dict[str, RouterScore] = {}

        for aid in candidate_agents:
            handle = agent_handles[aid]
            est_prob, profile_acc, profile_src = self._estimate_candidate_probability(
                source_agent=source_agent,
                target_agent=aid,
                source_probability=p_current,
            )
            reliability = self._resolve_reliability(
                agent_id=aid,
                belief_manager=belief_manager,
                profile_accuracy=profile_acc,
            )
            p_after = _clip_probability((reliability * est_prob) + ((1.0 - reliability) * p_current))
            after_cost = _min_expected_action_cost(p_after, self.decision_costs)
            query_cost = float(handle.cost)
            gain = current_cost - after_cost - query_cost
            util_penalty = float((utilization_penalties or {}).get(aid, 0.0))
            gain_adjusted = gain - util_penalty
            scores[aid] = RouterScore(
                agent_id=aid,
                expected_gain=float(gain),
                expected_gain_adjusted=float(gain_adjusted),
                current_action_cost=float(current_cost),
                expected_action_cost_after=float(after_cost),
                query_cost=query_cost,
                utilization_penalty=float(util_penalty),
                reliability=float(reliability),
                estimated_candidate_prob=float(est_prob),
                estimated_p_after=float(p_after),
                profile_source=profile_src,
            )
        return scores

    def select_next_agent(
        self,
        *,
        current_probability: float,
        source_agent: str,
        candidate_agents: list[str],
        agent_handles: Dict[str, AgentRuntimeHandle],
        belief_manager: Any,
        min_expected_gain: float,
        utilization_penalties: Optional[Dict[str, float]] = None,
    ) -> Tuple[Optional[str], Dict[str, RouterScore]]:
        scores = self.score_candidates(
            current_probability=current_probability,
            source_agent=source_agent,
            candidate_agents=candidate_agents,
            agent_handles=agent_handles,
            belief_manager=belief_manager,
            utilization_penalties=utilization_penalties,
        )
        if not scores:
            return None, {}

        # Deterministic tie-break: candidate ordering from agent_sequence.
        best_agent: Optional[str] = None
        best_score: Optional[RouterScore] = None
        for aid in candidate_agents:
            score = scores[aid]
            if best_score is None or score.expected_gain_adjusted > best_score.expected_gain_adjusted:
                best_agent = aid
                best_score = score

        if best_agent is None or best_score is None:
            return None, scores
        if float(best_score.expected_gain_adjusted) <= float(min_expected_gain):
            return None, scores
        return best_agent, scores
