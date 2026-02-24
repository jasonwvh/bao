from __future__ import annotations

import copy
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _logit(p: float, eps: float) -> float:
    clipped = _clip(p, eps, 1.0 - eps)
    return math.log(clipped / (1.0 - clipped))


@dataclass
class DriftStats:
    drift_score: float
    drift_detected: bool
    recent_entropy: float
    previous_entropy: float


class BayesianBeliefState:
    def __init__(
        self,
        flow_id: str,
        prior_mu: float = 0.0,
        prior_var: float = 1.0,
        drift_window: int = 10,
        drift_threshold: float = 0.08,
        eps: float = 1e-6,
    ):
        self.flow_id = flow_id
        self.mu = float(prior_mu)
        self.var = float(max(1e-4, prior_var))
        self.drift_window = max(2, int(drift_window))
        self.drift_threshold = float(drift_threshold)
        self.eps = max(1e-9, float(eps))
        self._p_cached = _clip(_sigmoid(self.mu), self.eps, 1.0 - self.eps)

        # Aggregation state used by posterior-first logit pooling.
        self._pooled_logit_sum = 0.0
        self._pooled_weight_sum = 0.0

        self.agent_reliabilities: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"alpha": 2.0, "beta": 1.0}
        )
        self.evidence_history: list[Dict[str, Any]] = []
        self._entropy_history: Deque[float] = deque(maxlen=self.drift_window * 2)

    def get_compromise_prob(self) -> float:
        if self._p_cached is None:
            self._p_cached = _clip(_sigmoid(self.mu), self.eps, 1.0 - self.eps)
        return _clip(float(self._p_cached), self.eps, 1.0 - self.eps)

    def get_epistemic_uncertainty(self) -> float:
        p = _clip(self.get_compromise_prob(), self.eps, 1.0 - self.eps)
        return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))

    def get_variance(self) -> float:
        return self.var

    def set_compromise_prob(self, p_mal: float) -> None:
        p = _clip(float(p_mal), self.eps, 1.0 - self.eps)
        self.mu = _logit(p, self.eps)
        self._p_cached = p
        self._entropy_history.append(self.get_epistemic_uncertainty())

    def get_reliability_estimate(self, agent_id: str) -> Tuple[float, float]:
        rel = self.agent_reliabilities[agent_id]
        alpha = rel["alpha"]
        beta = rel["beta"]
        mean = alpha / (alpha + beta)
        var = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
        return mean, math.sqrt(max(0.0, var))

    def _extract_probability(self, agent_output: Dict[str, Any]) -> float:
        proba = agent_output.get("proba", [0.5, 0.5])
        if isinstance(proba, (list, tuple)):
            p = float(proba[1] if len(proba) > 1 else proba[0])
        else:
            p = float(proba)
        return _clip(p, self.eps, 1.0 - self.eps)

    def _extract_likelihood_ratio(self, agent_output: Dict[str, Any]) -> float:
        if "likelihood_ratio" in agent_output:
            lr = float(agent_output.get("likelihood_ratio", 1.0))
            return _clip(lr, self.eps, 1.0 / self.eps)

        likelihoods = agent_output.get("likelihoods") or {}
        p_a = float(
            likelihoods.get(
                "p_obs_given_attack",
                likelihoods.get("p_attack", likelihoods.get("p_given_attack", 0.5)),
            )
        )
        p_c = float(
            likelihoods.get(
                "p_obs_given_clean",
                likelihoods.get("p_clean", likelihoods.get("p_given_clean", 0.5)),
            )
        )
        p_a = _clip(p_a, self.eps, 1.0)
        p_c = _clip(p_c, self.eps, 1.0)
        return _clip(p_a / p_c, self.eps, 1.0 / self.eps)

    def posterior_from_agent_proba(self, p_mal: float, weight: float = 1.0) -> float:
        w = max(self.eps, float(weight))
        p = _clip(float(p_mal), self.eps, 1.0 - self.eps)
        prev_weight = self._pooled_weight_sum
        logit_p = _logit(p, self.eps)
        self._pooled_logit_sum += w * logit_p
        self._pooled_weight_sum += w
        pooled_prob: float
        if prev_weight <= self.eps:
            # Preserve exact single-agent posterior parity.
            pooled_prob = p
            pooled_logit = logit_p
        else:
            pooled_logit = self._pooled_logit_sum / max(self._pooled_weight_sum, self.eps)
            pooled_prob = _sigmoid(pooled_logit)
        self.mu = pooled_logit
        self._p_cached = _clip(pooled_prob, self.eps, 1.0 - self.eps)
        self.var = _clip(1.0 / max(self._pooled_weight_sum, self.eps), 1e-4, 4.0)
        return self._p_cached

    def update_from_agent_output(
        self,
        *,
        agent_output: Dict[str, Any],
        agent_id: str,
        update_mode: str,
        weight: float,
        eps: float,
        likelihood_sanity_gate: bool,
    ) -> Dict[str, float]:
        mode = str(update_mode).strip().lower()
        eps = max(1e-9, float(eps))
        p_obs = self._extract_probability(agent_output)

        used_mode = mode
        used_likelihood = False
        llr = 0.0

        if mode == "likelihood_strict":
            lr = self._extract_likelihood_ratio(agent_output)
            llr = math.log(lr)
            llr_from_proba = _logit(p_obs, eps)

            if likelihood_sanity_gate:
                # If likelihood evidence points in opposite direction to model posterior,
                # treat likelihoods as unreliable and fall back to posterior update.
                if llr * llr_from_proba < 0.0:
                    used_mode = "posterior_first"
                else:
                    used_likelihood = True
            else:
                used_likelihood = True

            if used_likelihood:
                self.mu += max(eps, float(weight)) * llr
                self.var = _clip(self.var + abs(llr) * 0.03, 1e-4, 4.0)
                self._p_cached = _clip(_sigmoid(self.mu), self.eps, 1.0 - self.eps)

        if used_mode == "posterior_first":
            self.posterior_from_agent_proba(p_obs, weight=weight)

        comp_prob = self.get_compromise_prob()
        entropy = self.get_epistemic_uncertainty()

        self.evidence_history.append(
            {
                "agent_id": agent_id,
                "update_mode": used_mode,
                "used_likelihood": used_likelihood,
                "weight": float(weight),
                "observed_p": p_obs,
                "observed_llr": llr,
                "compromise_prob": comp_prob,
            }
        )
        self._entropy_history.append(entropy)

        return {
            "mu": self.mu,
            "var": self.var,
            "compromise_prob": comp_prob,
            "epistemic_uncertainty": entropy,
        }

    def variational_update(
        self,
        agent_output: Dict[str, Any],
        agent_id: str,
        learning_rate: float = 1.0,
        use_natural_gradient: bool = True,
    ) -> Dict[str, float]:
        # Compatibility shim for older VOI router code paths.
        return self.update_from_agent_output(
            agent_output=agent_output,
            agent_id=agent_id,
            update_mode="likelihood_strict",
            weight=max(self.eps, float(learning_rate)),
            eps=self.eps,
            likelihood_sanity_gate=False,
        )

    def update_agent_reliability(self, agent_id: str, prediction: int, true_label: int) -> None:
        correct = int(prediction == true_label)
        if correct:
            self.agent_reliabilities[agent_id]["alpha"] += 1.0
        else:
            self.agent_reliabilities[agent_id]["beta"] += 1.0

    def detect_drift(self) -> DriftStats:
        if len(self._entropy_history) < self.drift_window * 2:
            return DriftStats(0.0, False, 0.0, 0.0)

        vals = list(self._entropy_history)
        recent = vals[-self.drift_window :]
        previous = vals[-2 * self.drift_window : -self.drift_window]

        recent_mean = sum(recent) / len(recent)
        prev_mean = sum(previous) / len(previous)
        score = abs(recent_mean - prev_mean)
        return DriftStats(
            drift_score=score,
            drift_detected=score > self.drift_threshold,
            recent_entropy=recent_mean,
            previous_entropy=prev_mean,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "mu": self.mu,
            "var": self.var,
            "eps": self.eps,
            "p_cached": self._p_cached,
            "pooled_logit_sum": self._pooled_logit_sum,
            "pooled_weight_sum": self._pooled_weight_sum,
            "agent_reliabilities": copy.deepcopy(dict(self.agent_reliabilities)),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BayesianBeliefState":
        obj = cls(
            flow_id=str(data["flow_id"]),
            prior_mu=float(data.get("mu", 0.0)),
            prior_var=float(data.get("var", 1.0)),
            eps=float(data.get("eps", 1e-6)),
        )
        obj._p_cached = _clip(
            float(data.get("p_cached", _sigmoid(obj.mu))),
            obj.eps,
            1.0 - obj.eps,
        )
        obj._pooled_logit_sum = float(data.get("pooled_logit_sum", 0.0))
        obj._pooled_weight_sum = float(data.get("pooled_weight_sum", 0.0))

        rel = defaultdict(lambda: {"alpha": 1.0, "beta": 1.0})
        for key, value in data.get("agent_reliabilities", {}).items():
            rel[key] = {
                "alpha": float(value.get("alpha", 1.0)),
                "beta": float(value.get("beta", 1.0)),
            }
        obj.agent_reliabilities = rel
        return obj


class BeliefStateManager:
    def __init__(
        self,
        drift_window: int = 10,
        drift_threshold: float = 0.08,
        backend: Optional[Any] = None,
        eps: float = 1e-6,
    ):
        self.beliefs: Dict[str, BayesianBeliefState] = {}
        self.backend = backend
        self.global_agent_reliabilities: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"alpha": 2.0, "beta": 1.0}
        )
        self._drift_window = int(drift_window)
        self._drift_threshold = float(drift_threshold)
        self._eps = max(1e-9, float(eps))

    def get_or_create_belief(self, flow_id: str, reset: bool = False, prior_attack_rate: float = 0.5) -> BayesianBeliefState:
        if reset:
            self.beliefs.pop(flow_id, None)

        if flow_id not in self.beliefs:
            belief: BayesianBeliefState
            loaded_from_backend = False
            if self.backend is not None and not reset:
                saved = self.backend.load_flow_belief(flow_id)
                if saved is not None:
                    belief = BayesianBeliefState.from_dict(saved)
                    loaded_from_backend = True
                else:
                    belief = BayesianBeliefState(
                        flow_id=flow_id,
                        drift_window=self._drift_window,
                        drift_threshold=self._drift_threshold,
                        eps=self._eps,
                    )
            else:
                belief = BayesianBeliefState(
                    flow_id=flow_id,
                    drift_window=self._drift_window,
                    drift_threshold=self._drift_threshold,
                    eps=self._eps,
                )

            if reset or not loaded_from_backend:
                prior = _clip(float(prior_attack_rate), self._eps, 1.0 - self._eps)
                belief.set_compromise_prob(prior)

            belief.agent_reliabilities = copy.deepcopy(self.global_agent_reliabilities)
            self.beliefs[flow_id] = belief

        return self.beliefs[flow_id]

    def get_belief(self, flow_id: str) -> Optional[BayesianBeliefState]:
        return self.beliefs.get(flow_id)

    def delete_belief(self, flow_id: str) -> None:
        self.beliefs.pop(flow_id, None)

    def update_global_reliabilities(self, agent_id: str, correct: bool) -> None:
        if self.backend is not None:
            alpha, beta = self.backend.update_global_reliability(agent_id, correct)
            self.global_agent_reliabilities[agent_id]["alpha"] = alpha
            self.global_agent_reliabilities[agent_id]["beta"] = beta
            return

        if correct:
            self.global_agent_reliabilities[agent_id]["alpha"] += 1.0
        else:
            self.global_agent_reliabilities[agent_id]["beta"] += 1.0

    def get_global_reliability(self, agent_id: str) -> float:
        if self.backend is not None:
            alpha, beta = self.backend.get_global_reliability(agent_id)
            self.global_agent_reliabilities[agent_id]["alpha"] = alpha
            self.global_agent_reliabilities[agent_id]["beta"] = beta
            return alpha / (alpha + beta)

        rel = self.global_agent_reliabilities[agent_id]
        return rel["alpha"] / (rel["alpha"] + rel["beta"])

    def persist_belief(self, flow_id: str) -> None:
        if self.backend is None:
            return
        belief = self.beliefs.get(flow_id)
        if belief is None:
            return
        self.backend.save_flow_belief(flow_id, belief.to_dict())
