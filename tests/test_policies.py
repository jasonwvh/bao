from __future__ import annotations

from dataclasses import dataclass

from bao.benchmark.policies import (
    BAOPolicy,
    ConfidenceEscalationPolicy,
    FixedCascadePolicy,
    QueryAllEnsemblePolicy,
    SingleCheapestPolicy,
)
from bao.benchmark.types import BenchmarkConfig, FlowRecord


@dataclass
class StubAgent:
    name: str
    p: float
    cost: float

    def predict_proba(self, features, seed):
        return self.p


class IdentityCalibrator:
    def calibrate(self, agent_name, raw_score):
        return raw_score


def test_fixed_cascade_routes_to_b_when_conf_in_band():
    cfg = BenchmarkConfig(cascade_conf_low=0.55, cascade_conf_high=0.9)
    agents = {"agent_a": StubAgent("a", 0.6, 1.0), "agent_b": StubAgent("b", 0.9, 5.0)}
    policy = FixedCascadePolicy(cfg=cfg)
    flow = FlowRecord(flow_id="f1", label=1, features={})

    out = policy.run_flow(flow, agents, IdentityCalibrator(), seed=1)

    assert out["agent_calls"] == 2
    assert out["cost"] == 6.0


def test_confidence_escalation_routes_based_on_entropy_or_confidence():
    cfg = BenchmarkConfig(escalation_conf_threshold=0.75, escalation_entropy_threshold=0.5)
    agents = {"agent_a": StubAgent("a", 0.51, 1.0), "agent_b": StubAgent("b", 0.95, 5.0)}
    policy = ConfidenceEscalationPolicy(cfg=cfg)
    flow = FlowRecord(flow_id="f1", label=1, features={})

    out = policy.run_flow(flow, agents, IdentityCalibrator(), seed=1)

    assert out["agent_calls"] == 2
    assert out["cost"] == 6.0


def test_single_cheapest_only_calls_agent_a():
    cfg = BenchmarkConfig()
    agents = {"agent_a": StubAgent("a", 0.2, 1.0), "agent_b": StubAgent("b", 0.9, 5.0)}
    policy = SingleCheapestPolicy(cfg=cfg)
    flow = FlowRecord(flow_id="f1", label=0, features={})

    out = policy.run_flow(flow, agents, IdentityCalibrator(), seed=1)

    assert out["agent_calls"] == 1
    assert out["cost"] == 1.0


def test_query_all_always_calls_both_agents():
    cfg = BenchmarkConfig()
    agents = {"agent_a": StubAgent("a", 0.2, 1.0), "agent_b": StubAgent("b", 0.9, 5.0)}
    policy = QueryAllEnsemblePolicy(cfg=cfg, weights={"agent_a": 1.0, "agent_b": 1.0})
    flow = FlowRecord(flow_id="f1", label=1, features={})

    out = policy.run_flow(flow, agents, IdentityCalibrator(), seed=1)

    assert out["agent_calls"] == 2
    assert out["cost"] == 6.0


def test_bao_policy_is_deterministic_given_seed_and_inputs():
    cfg = BenchmarkConfig()
    agents = {"agent_a": StubAgent("a", 0.51, 1.0), "agent_b": StubAgent("b", 0.85, 5.0)}
    policy = BAOPolicy(cfg=cfg)
    flow = FlowRecord(flow_id="f1", label=1, features={})

    out1 = policy.run_flow(flow, agents, IdentityCalibrator(), seed=7)
    out2 = policy.run_flow(flow, agents, IdentityCalibrator(), seed=7)

    assert out1["score"] == out2["score"]
    assert out1["decision"] == out2["decision"]
