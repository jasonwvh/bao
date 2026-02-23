from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .belief_state import BayesianBeliefState


class VOIRouter:
    def __init__(
        self,
        agents: Dict[str, Any],
        observation_models: Dict[str, Any],
        c_fn: float = 50.0,
        c_fp: float = 5.0,
        c_h: float = 500.0,
        use_surrogate: bool = True,
        allow_exact: bool = False,
        capability_filter: Optional[Callable[[List[str], Dict[str, Any]], List[str]]] = None,
        lazy_exact_interval: int = 100,
        exact_uncertainty_trigger: float = 0.85,
        cache_bins: int = 64,
    ):
        self.agents = agents
        self.observation_models = observation_models
        self.c_fn = c_fn
        self.c_fp = c_fp
        self.c_h = c_h
        self.use_surrogate = use_surrogate
        self.allow_exact = allow_exact
        self.capability_filter = capability_filter

        self.lazy_exact_interval = max(1, int(lazy_exact_interval))
        self.exact_uncertainty_trigger = float(exact_uncertainty_trigger)
        self.cache_bins = max(8, int(cache_bins))

        self._surrogate_samples: list[tuple[np.ndarray, float]] = []
        self._surrogate_weights: Optional[np.ndarray] = None
        self._surrogate_cache: Dict[tuple[str, int, int, int], float] = {}
        self._exact_counters = defaultdict(int)

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _features(self, agent_id: str, belief_state: BayesianBeliefState, flow_features: Dict[str, Any]) -> np.ndarray:
        rel, _ = belief_state.get_reliability_estimate(agent_id)
        packet_count = self._safe_float(flow_features.get("packet_count", flow_features.get("spkts", 0.0)))
        byte_count = self._safe_float(flow_features.get("byte_count", flow_features.get("sbytes", 0.0)))
        flow_duration = self._safe_float(flow_features.get("flow_duration", flow_features.get("dur", 0.0)))

        return np.array(
            [
                belief_state.get_compromise_prob(),
                belief_state.get_epistemic_uncertainty(),
                belief_state.get_variance(),
                rel,
                float(self.agents[agent_id].cost),
                packet_count / 1000.0,
                byte_count / 100000.0,
                flow_duration / 60.0,
                1.0,
            ],
            dtype=float,
        )

    def _cache_key(self, agent_id: str, belief_state: BayesianBeliefState) -> tuple[str, int, int, int]:
        p = int(np.clip(belief_state.get_compromise_prob() * self.cache_bins, 0, self.cache_bins))
        h = int(np.clip(belief_state.get_epistemic_uncertainty() * self.cache_bins, 0, self.cache_bins))
        v = int(np.clip(belief_state.get_variance() * self.cache_bins, 0, self.cache_bins))
        return agent_id, p, h, v

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
        key = self._cache_key(agent_id, belief_state)
        if key in self._surrogate_cache:
            return self._surrogate_cache[key]

        x = self._features(agent_id, belief_state, flow_features)
        if self._surrogate_weights is None:
            score = -0.2 * float(self.agents[agent_id].cost) + 0.8 * belief_state.get_epistemic_uncertainty()
        else:
            score = float(np.dot(x, self._surrogate_weights))

        self._surrogate_cache[key] = score
        return score

    def estimate_voi(
        self,
        agent_id: str,
        belief_state: BayesianBeliefState,
        flow_features: Dict[str, Any],
    ) -> float:
        if self.use_surrogate or not self.allow_exact:
            return self.estimate_voi_surrogate(agent_id, belief_state, flow_features)
        return self.estimate_voi_exact(agent_id, belief_state, flow_features)

    def _update_surrogate(self, features: np.ndarray, voi_value: float) -> None:
        self._surrogate_samples.append((features, voi_value))
        if len(self._surrogate_samples) < 128:
            return
        xs = np.stack([s[0] for s in self._surrogate_samples[-512:]])
        ys = np.array([s[1] for s in self._surrogate_samples[-512:]], dtype=float)
        self._surrogate_weights, *_ = np.linalg.lstsq(xs, ys, rcond=None)
        if len(self._surrogate_cache) > 4096:
            self._surrogate_cache.clear()

    def _should_refresh_exact(self, agent_id: str, belief_state: BayesianBeliefState) -> bool:
        if not self.allow_exact:
            return False
        self._exact_counters[agent_id] += 1
        if belief_state.get_epistemic_uncertainty() >= self.exact_uncertainty_trigger:
            return True
        return (self._exact_counters[agent_id] % self.lazy_exact_interval) == 0

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

            # Lazy exact refresh keeps hot path fast while still anchoring surrogate quality.
            if self._should_refresh_exact(aid, belief_state):
                exact_voi = self.estimate_voi_exact(aid, belief_state, flow_features)
                self._update_surrogate(self._features(aid, belief_state, flow_features), exact_voi)

        best = max(voi_scores, key=voi_scores.get)
        best_voi = voi_scores[best]

        exploration_rate = 0.1
        rng = np.random.default_rng()

        if best_voi <= 0:
            if rng.random() <= exploration_rate:
                return best, 0.01, voi_scores
            return None, None, voi_scores

        return best, best_voi, voi_scores
