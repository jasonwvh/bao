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
    ):
        self.flow_id = flow_id
        self.mu = float(prior_mu)
        self.var = float(max(1e-4, prior_var))
        self.drift_window = max(2, int(drift_window))
        self.drift_threshold = float(drift_threshold)

        self.agent_reliabilities: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"alpha": 1.0, "beta": 1.0}
        )
        self.evidence_history: list[Dict[str, Any]] = []
        self._entropy_history: Deque[float] = deque(maxlen=self.drift_window * 2)

    def get_compromise_prob(self) -> float:
        return _sigmoid(self.mu)

    def get_epistemic_uncertainty(self) -> float:
        p = _clip(self.get_compromise_prob(), 1e-9, 1 - 1e-9)
        return -(p * math.log(p) + (1 - p) * math.log(1 - p))

    def get_variance(self) -> float:
        return self.var

    def set_compromise_prob(self, p_mal: float) -> None:
        p = _clip(p_mal, 1e-6, 1 - 1e-6)
        self.mu = math.log(p / (1.0 - p))
        self._entropy_history.append(self.get_epistemic_uncertainty())

    def get_reliability_estimate(self, agent_id: str) -> Tuple[float, float]:
        rel = self.agent_reliabilities[agent_id]
        alpha = rel["alpha"]
        beta = rel["beta"]
        mean = alpha / (alpha + beta)
        var = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
        return mean, math.sqrt(max(0.0, var))

    def variational_update(
        self,
        agent_output: Dict[str, Any],
        agent_id: str,
        learning_rate: float = 0.25,
        use_natural_gradient: bool = True,
    ) -> Dict[str, float]:
        # Prefer explicit likelihood elicitation when available: agents may return
        # P(obs|attack) and P(obs|clean) (under key "likelihoods") or a
        # pre-computed likelihood_ratio. Convert to log-likelihood ratio and
        # apply as additive evidence in log-odds space.
        obs_logit: float
        if "likelihood_ratio" in agent_output:
            lr = float(agent_output.get("likelihood_ratio", 1.0))
            lr = _clip(lr, 1e-9, 1e9)
            obs_logit = math.log(lr)
            obs_record = lr
        elif "likelihoods" in agent_output:
            l = agent_output.get("likelihoods") or {}
            p_a = float(l.get("p_obs_given_attack", l.get("p_attack", l.get("p_given_attack", 0.5))))
            p_c = float(l.get("p_obs_given_clean", l.get("p_clean", l.get("p_given_clean", 0.5))))
            p_a = _clip(p_a, 1e-9, 1.0)
            p_c = _clip(p_c, 1e-9, 1.0)
            lr = p_a / p_c if p_c > 0 else 1.0
            lr = _clip(lr, 1e-9, 1e9)
            obs_logit = math.log(lr)
            obs_record = lr
        else:
            proba = agent_output.get("proba", [0.5, 0.5])
            if isinstance(proba, (list, tuple)):
                obs_p = float(proba[1] if len(proba) > 1 else proba[0])
            else:
                obs_p = float(proba)
            obs_p = _clip(obs_p, 1e-4, 1 - 1e-4)

            # Convert posterior-style agent proba into a pseudo log-odds evidence
            obs_logit = math.log(obs_p / (1.0 - obs_p))
            obs_record = obs_p

        reliability, _ = self.get_reliability_estimate(agent_id)
        # Lightweight closed-form additive update in log-odds space.
        delta = learning_rate * reliability * obs_logit
        self.mu += delta
        self.var = _clip(self.var + learning_rate * abs(delta) * 0.02, 1e-4, 4.0)

        self.evidence_history.append(
            {
                "agent_id": agent_id,
                "observation": obs_record,
                "reliability": reliability,
                "compromise_prob": self.get_compromise_prob(),
            }
        )
        self._entropy_history.append(self.get_epistemic_uncertainty())

        return {
            "mu": self.mu,
            "var": self.var,
            "compromise_prob": self.get_compromise_prob(),
            "epistemic_uncertainty": self.get_epistemic_uncertainty(),
        }

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
            "agent_reliabilities": copy.deepcopy(dict(self.agent_reliabilities)),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BayesianBeliefState":
        obj = cls(flow_id=str(data["flow_id"]), prior_mu=float(data["mu"]), prior_var=float(data["var"]))
        rel = defaultdict(lambda: {"alpha": 1.0, "beta": 1.0})
        for k, v in data.get("agent_reliabilities", {}).items():
            rel[k] = {"alpha": float(v["alpha"]), "beta": float(v["beta"])}
        obj.agent_reliabilities = rel
        return obj


class BeliefStateManager:
    def __init__(
        self,
        drift_window: int = 10,
        drift_threshold: float = 0.08,
        backend: Optional[Any] = None,
    ):
        self.beliefs: Dict[str, BayesianBeliefState] = {}
        self.backend = backend
        self.global_agent_reliabilities: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"alpha": 1.0, "beta": 1.0}
        )
        self._drift_window = drift_window
        self._drift_threshold = drift_threshold

    def get_or_create_belief(self, flow_id: str) -> BayesianBeliefState:
        if flow_id not in self.beliefs:
            if self.backend is not None:
                saved = self.backend.load_flow_belief(flow_id)
                if saved is not None:
                    belief = BayesianBeliefState.from_dict(saved)
                else:
                    belief = BayesianBeliefState(
                        flow_id=flow_id,
                        drift_window=self._drift_window,
                        drift_threshold=self._drift_threshold,
                    )
            else:
                belief = BayesianBeliefState(
                    flow_id=flow_id,
                    drift_window=self._drift_window,
                    drift_threshold=self._drift_threshold,
                )

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
