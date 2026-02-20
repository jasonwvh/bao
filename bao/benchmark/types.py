from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Sequence


@dataclass(frozen=True)
class BenchmarkConfig:
    # Shared decision thresholds
    p_accept: float = 0.30
    p_reject: float = 0.70
    uncertainty_threshold: float = 0.60

    # Baseline-specific control thresholds
    cascade_conf_low: float = 0.55
    cascade_conf_high: float = 0.85
    escalation_conf_threshold: float = 0.75
    escalation_entropy_threshold: float = 0.60

    # Cost model
    c_fn: float = 100.0
    c_fp: float = 1.0
    c_h: float = 5.0
    agent_costs: Dict[str, float] = field(
        default_factory=lambda: {
            "agent_a": 1.0,
            "agent_b": 5.0,
        }
    )

    # Runner settings
    calibration_fraction: float = 0.20
    bootstrap_samples: int = 1000


@dataclass
class FlowRecord:
    flow_id: str
    label: int
    features: Dict[str, float]


@dataclass
class PolicyResult:
    policy_name: str
    seed: int
    labels: List[int]
    scores: List[float]
    decisions: List[str]
    defer_flags: List[int]
    costs: List[float]
    agent_calls: List[int]
    latencies_ms: List[float]


class AgentProtocol(Protocol):
    name: str
    cost: float

    def predict_proba(self, features: Dict[str, float], seed: int) -> float:
        ...


class CalibratorProtocol(Protocol):
    def calibrate(self, agent_name: str, raw_score: float) -> float:
        ...


class PolicyProtocol(Protocol):
    name: str

    def run_flow(
        self,
        state: FlowRecord,
        agents: Dict[str, AgentProtocol],
        calibrator: CalibratorProtocol,
        seed: int,
    ) -> Dict[str, Any]:
        ...


def entropy_binary(p: float) -> float:
    import math

    p = max(1e-9, min(1.0 - 1e-9, p))
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def decision_from_prob(p_mal: float, uncertainty: float, cfg: BenchmarkConfig) -> str:
    if uncertainty >= cfg.uncertainty_threshold:
        return "defer"
    if p_mal < cfg.p_accept:
        return "accept"
    if p_mal > cfg.p_reject:
        return "reject"
    return "defer"
