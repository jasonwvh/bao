from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .belief_state import BayesianBeliefState


class VOIRouter:
    def __init__(
        self,
        agents: Dict[str, Any],
        observation_models: Dict[str, Any],
        c_fn: float = 1000000.0,  # Cost of false negative: letting hacker in ($1M)
        c_fp: float = 50.0,       # Cost of false positive: blocking safe user ($50)
        c_h: float = 100000.0,     # Cost of human deferral: expensive analyst time ($100k)
        use_surrogate: bool = True,
        allow_exact: bool = False,
        capability_filter: Optional[Callable[[List[str], Dict[str, Any]], List[str]]] = None,
    ):
        self.agents = agents
        self.observation_models = observation_models
        self.c_fn = c_fn
        self.c_fp = c_fp
        self.c_h = c_h
        self.use_surrogate = use_surrogate
        self.allow_exact = allow_exact
        self.capability_filter = capability_filter
        self._surrogate_samples: list[tuple[np.ndarray, float]] = []
        self._surrogate_weights: Optional[np.ndarray] = None

    def _features(self, agent_id: str, belief_state: BayesianBeliefState, flow_features: Dict[str, Any]) -> np.ndarray:
        rel, _ = belief_state.get_reliability_estimate(agent_id)
        return np.array(
            [
                belief_state.get_compromise_prob(),
                belief_state.get_epistemic_uncertainty(),
                belief_state.get_variance(),
                rel,
                float(self.agents[agent_id].cost),
                float(flow_features.get("packet_count", 0.0)) / 1000.0,
                float(flow_features.get("byte_count", 0.0)) / 100000.0,
                float(flow_features.get("flow_duration", 0.0)) / 60.0,
                1.0,
            ],
            dtype=float,
        )

    def compute_expected_loss(self, belief_state: BayesianBeliefState) -> Dict[str, float | str]:
        p = belief_state.get_compromise_prob()
        cost_accept = p * self.c_fn
        cost_reject = (1.0 - p) * self.c_fp
        cost_defer = self.c_h
        losses = {"accept": cost_accept, "reject": cost_reject, "defer": cost_defer}
        action = min(losses, key=losses.get)
        return {"loss": float(losses[action]), "optimal_action": action}

    def estimate_voi_exact(
        self,
        agent_id: str,
        belief_state: BayesianBeliefState,
        flow_features: Dict[str, Any],
        n_samples: int = 20,
    ) -> float:
        agent = self.agents[agent_id]
        model = self.observation_models.get(agent_id)
        if model is None:
            # Conservative cold-start heuristic: allow cheap first query under high uncertainty.
            return (2.0 * belief_state.get_epistemic_uncertainty()) - float(agent.cost)

        current_loss = float(self.compute_expected_loss(belief_state)["loss"])
        p_mal = belief_state.get_compromise_prob()
        rng = np.random.default_rng(int(flow_features.get("seed", 0)) + len(flow_features))

        expected_future_loss = 0.0
        for _ in range(max(2, n_samples)):
            y = 1 if rng.random() < p_mal else 0
            z = model.sample_observation(y, rng)
            temp = BayesianBeliefState.from_dict(belief_state.to_dict())
            temp.variational_update({"proba": [1.0 - z, z]}, agent_id=agent_id, learning_rate=0.35)
            expected_future_loss += float(self.compute_expected_loss(temp)["loss"]) / max(2, n_samples)

        return (current_loss - expected_future_loss) - float(agent.cost)

    def estimate_voi_surrogate(
        self,
        agent_id: str,
        belief_state: BayesianBeliefState,
        flow_features: Dict[str, Any],
    ) -> float:
        x = self._features(agent_id, belief_state, flow_features)
        if self._surrogate_weights is None:
            return -0.2 * float(self.agents[agent_id].cost) + 0.8 * belief_state.get_epistemic_uncertainty()
        return float(np.dot(x, self._surrogate_weights))

    def estimate_voi(
        self,
        agent_id: str,
        belief_state: BayesianBeliefState,
        flow_features: Dict[str, Any],
    ) -> float:
        if self.use_surrogate:
            return self.estimate_voi_surrogate(agent_id, belief_state, flow_features)
        if not self.allow_exact:
            # Lightweight deterministic heuristic if exact VOI is disabled.
            return self.estimate_voi_surrogate(agent_id, belief_state, flow_features)
        return self.estimate_voi_exact(agent_id, belief_state, flow_features)

    def _update_surrogate(self, features: np.ndarray, voi_value: float) -> None:
        self._surrogate_samples.append((features, voi_value))
        if len(self._surrogate_samples) < 24:
            return
        xs = np.stack([s[0] for s in self._surrogate_samples[-512:]])
        ys = np.array([s[1] for s in self._surrogate_samples[-512:]], dtype=float)
        self._surrogate_weights, *_ = np.linalg.lstsq(xs, ys, rcond=None)

    def select_best_agent(
        self,
        belief_state: BayesianBeliefState,
        flow_features: Dict[str, Any],
        queried_agents: List[str],
    ) -> Tuple[Optional[str], Optional[float], Dict[str, float]]:
        available = [a for a in self.agents.keys() if a not in queried_agents]
        if self.capability_filter is not None:
            available = self.capability_filter(available, flow_features)
        if not available:
            return None, None, {}

        voi_scores: Dict[str, float] = {}
        for aid in available:
            score = self.estimate_voi(aid, belief_state, flow_features)
            voi_scores[aid] = score
            self._update_surrogate(self._features(aid, belief_state, flow_features), score)

        best = max(voi_scores, key=voi_scores.get)
        best_voi = voi_scores[best]
        if best_voi <= 0:
            return None, None, voi_scores
        return best, best_voi, voi_scores
