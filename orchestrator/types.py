from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, TypedDict


class AgentOutput(TypedDict, total=False):
    proba: List[float]
    prediction: Dict[str, Any]
    uncertainty: Dict[str, float]
    cost: float
    agent_id: str
    metadata: Dict[str, Any]


class AgentProtocol(Protocol):
    agent_id: str
    cost: float

    def predict_with_uncertainty(
        self,
        flow_features: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> AgentOutput:
        ...


@dataclass(frozen=True)
class AgentRegistryEntry:
    id: str
    enabled: bool
    endpoint: str
    transport: str
    timeout_ms: int
    cost: float
    capabilities: List[str]
    health_path: str
    infer_path: str
    capabilities_path: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class AgentRuntimeHandle:
    agent_id: str
    endpoint: str
    transport: str
    timeout_ms: int
    cost: float
    capabilities: List[str]
    health_path: str
    infer_path: str
    capabilities_path: str
    meta: Dict[str, Any]


class A2AInferRequest(TypedDict, total=False):
    request_id: str
    flow_id: str
    timestamp: float
    flow_features: Dict[str, Any]
    context: Dict[str, Any]


class A2AInferResponse(TypedDict, total=False):
    agent_id: str
    proba: List[float]
    prediction: Dict[str, Any]
    uncertainty: Dict[str, float]
    cost: float
    latency_ms: float
    metadata: Dict[str, Any]


class A2AHealthResponse(TypedDict, total=False):
    status: str
    agent_id: str
    version: str


class A2ACapabilitiesResponse(TypedDict, total=False):
    agent_id: str
    capabilities: List[str]
    cost: float


@dataclass(frozen=True)
class DecisionThresholds:
    p_accept: float = 0.30
    p_reject: float = 0.70
    uncertainty: float = 0.60

    def decide(self, p_mal: float, uncertainty: float, has_more_agents: bool = False) -> str:
        if p_mal < self.p_accept:
            return "accept"
        if p_mal > self.p_reject:
            return "reject"
        # In gray zone [p_accept, p_reject]
        if uncertainty >= self.uncertainty:
            if has_more_agents:
                return "more_agents"
            return "defer"
        if has_more_agents:
            return "more_agents"
        # Agents exhausted, low uncertainty in gray zone: make best guess
        if p_mal < 0.5:
            return "accept"
        return "reject"


class FullBAOState(TypedDict, total=False):
    flow_id: str
    flow_features: Dict[str, Any]
    timestamp: float
    true_label: Optional[int]

    agents_available: List[str]
    agents_queried: List[str]
    agent_outputs: List[AgentOutput]
    voi_scores: Dict[str, float]
    selected_agent: Optional[str]
    selected_voi: Optional[float]
    iteration: int
    max_iterations: int

    belief_mu: float
    belief_var: float
    compromise_prob: float
    epistemic_uncertainty: float
    agent_reliabilities: Dict[str, float]

    drift_detected: bool
    drift_score: float
    needs_calibration: bool
    consensus_triggered: bool
    consensus_result: Dict[str, Any]

    decision: Optional[str]
    decision_reasoning: List[str]
    hitl_context: Optional[Dict[str, Any]]

    inference_time_ms: float
    cumulative_cost: float
    confidence: float
    total_time_ms: float
