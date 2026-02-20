from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict

from .types import (
    BenchmarkConfig,
    FlowRecord,
    PolicyProtocol,
    entropy_binary,
    decision_from_prob,
)


def _confidence(p: float) -> float:
    return max(p, 1.0 - p)


def _expected_loss(p_mal: float, cfg: BenchmarkConfig) -> float:
    # Expected loss if forced to binary action now (ignores defer action directly).
    cost_accept = p_mal * cfg.c_fn
    cost_reject = (1.0 - p_mal) * cfg.c_fp
    return min(cost_accept, cost_reject)


def _result(policy: str, p_mal: float, decision: str, cost: float, agent_calls: int, latency_ms: float) -> Dict[str, Any]:
    return {
        "policy": policy,
        "score": max(0.001, min(0.999, p_mal)),
        "decision": decision,
        "cost": float(cost),
        "agent_calls": int(agent_calls),
        "latency_ms": float(latency_ms),
        "defer": int(decision == "defer"),
    }


@dataclass
class BAOPolicy(PolicyProtocol):
    cfg: BenchmarkConfig
    name: str = "bao"

    def run_flow(self, state: FlowRecord, agents: Dict[str, Any], calibrator: Any, seed: int) -> Dict[str, Any]:
        start = time.perf_counter()

        p_a = calibrator.calibrate("agent_a", agents["agent_a"].predict_proba(state.features, seed))
        h_a = entropy_binary(p_a)
        total_cost = self.cfg.agent_costs["agent_a"]
        calls = 1

        # VOI proxy: expected loss reduction from consulting B minus cost of B.
        loss_now = _expected_loss(p_a, self.cfg)
        should_query_b = h_a > self.cfg.uncertainty_threshold * 0.8

        p = p_a
        if should_query_b:
            p_b = calibrator.calibrate("agent_b", agents["agent_b"].predict_proba(state.features, seed))
            calls += 1
            total_cost += self.cfg.agent_costs["agent_b"]

            # Reliability-biased fusion; B has higher modeled reliability.
            p_fused = 0.35 * p_a + 0.65 * p_b
            loss_after = _expected_loss(p_fused, self.cfg)
            voi = loss_now - loss_after - self.cfg.agent_costs["agent_b"]
            p = p_fused if voi > 0 else p_a

        h = entropy_binary(p)
        decision = decision_from_prob(p, h, self.cfg)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return _result(self.name, p, decision, total_cost, calls, latency_ms)


@dataclass
class QueryAllEnsemblePolicy(PolicyProtocol):
    cfg: BenchmarkConfig
    weights: Dict[str, float]
    name: str = "baseline_query_all"

    def run_flow(self, state: FlowRecord, agents: Dict[str, Any], calibrator: Any, seed: int) -> Dict[str, Any]:
        start = time.perf_counter()

        p_a = calibrator.calibrate("agent_a", agents["agent_a"].predict_proba(state.features, seed))
        p_b = calibrator.calibrate("agent_b", agents["agent_b"].predict_proba(state.features, seed))

        w_a = self.weights.get("agent_a", 0.5)
        w_b = self.weights.get("agent_b", 0.5)
        p = (w_a * p_a + w_b * p_b) / max(w_a + w_b, 1e-9)

        decision = decision_from_prob(p, entropy_binary(p), self.cfg)
        latency_ms = (time.perf_counter() - start) * 1000.0
        total_cost = self.cfg.agent_costs["agent_a"] + self.cfg.agent_costs["agent_b"]
        return _result(self.name, p, decision, total_cost, 2, latency_ms)


@dataclass
class FixedCascadePolicy(PolicyProtocol):
    cfg: BenchmarkConfig
    name: str = "baseline_fixed_cascade"

    def run_flow(self, state: FlowRecord, agents: Dict[str, Any], calibrator: Any, seed: int) -> Dict[str, Any]:
        start = time.perf_counter()

        p_a = calibrator.calibrate("agent_a", agents["agent_a"].predict_proba(state.features, seed))
        conf_a = _confidence(p_a)

        p = p_a
        calls = 1
        total_cost = self.cfg.agent_costs["agent_a"]

        if self.cfg.cascade_conf_low <= conf_a <= self.cfg.cascade_conf_high:
            p = calibrator.calibrate("agent_b", agents["agent_b"].predict_proba(state.features, seed))
            calls += 1
            total_cost += self.cfg.agent_costs["agent_b"]

        decision = decision_from_prob(p, entropy_binary(p), self.cfg)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return _result(self.name, p, decision, total_cost, calls, latency_ms)


@dataclass
class ConfidenceEscalationPolicy(PolicyProtocol):
    cfg: BenchmarkConfig
    name: str = "baseline_confidence_escalation"

    def run_flow(self, state: FlowRecord, agents: Dict[str, Any], calibrator: Any, seed: int) -> Dict[str, Any]:
        start = time.perf_counter()

        p_a = calibrator.calibrate("agent_a", agents["agent_a"].predict_proba(state.features, seed))
        conf_a = _confidence(p_a)
        ent_a = entropy_binary(p_a)

        p = p_a
        calls = 1
        total_cost = self.cfg.agent_costs["agent_a"]

        if ent_a > self.cfg.escalation_entropy_threshold or conf_a < self.cfg.escalation_conf_threshold:
            p = calibrator.calibrate("agent_b", agents["agent_b"].predict_proba(state.features, seed))
            calls += 1
            total_cost += self.cfg.agent_costs["agent_b"]

        decision = decision_from_prob(p, entropy_binary(p), self.cfg)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return _result(self.name, p, decision, total_cost, calls, latency_ms)


@dataclass
class SingleCheapestPolicy(PolicyProtocol):
    cfg: BenchmarkConfig
    name: str = "baseline_single_cheapest"

    def run_flow(self, state: FlowRecord, agents: Dict[str, Any], calibrator: Any, seed: int) -> Dict[str, Any]:
        start = time.perf_counter()

        p = calibrator.calibrate("agent_a", agents["agent_a"].predict_proba(state.features, seed))
        decision = decision_from_prob(p, entropy_binary(p), self.cfg)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return _result(self.name, p, decision, self.cfg.agent_costs["agent_a"], 1, latency_ms)


def compute_query_all_weights(calibration_rows: list[FlowRecord], agents: Dict[str, Any], calibrator: Any, seed: int) -> Dict[str, float]:
    def brier(agent_name: str) -> float:
        if not calibration_rows:
            return 0.25
        err = 0.0
        for row in calibration_rows:
            p = calibrator.calibrate(agent_name, agents[agent_name].predict_proba(row.features, seed))
            y = float(row.label)
            err += (p - y) ** 2
        return err / len(calibration_rows)

    b_a = brier("agent_a")
    b_b = brier("agent_b")

    inv_a = 1.0 / max(b_a, 1e-6)
    inv_b = 1.0 / max(b_b, 1e-6)
    return {"agent_a": inv_a, "agent_b": inv_b}


def build_policies(cfg: BenchmarkConfig, calibration_rows: list[FlowRecord], agents: Dict[str, Any], calibrator: Any, seed: int) -> Dict[str, PolicyProtocol]:
    weights = compute_query_all_weights(calibration_rows, agents, calibrator, seed)
    return {
        "bao": BAOPolicy(cfg=cfg),
        "baseline_query_all": QueryAllEnsemblePolicy(cfg=cfg, weights=weights),
        "baseline_fixed_cascade": FixedCascadePolicy(cfg=cfg),
        "baseline_confidence_escalation": ConfidenceEscalationPolicy(cfg=cfg),
        "baseline_single_cheapest": SingleCheapestPolicy(cfg=cfg),
    }
