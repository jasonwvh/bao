from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from typing import Dict, List

import json
import yaml

from orchestrator.integrated_system import IntegratedBAOSystem


class _FakeA2A:
    def __init__(self, probs_by_agent: Dict[str, List[float]]):
        self.probs_by_agent = {k: list(v) for k, v in probs_by_agent.items()}
        self.calls: List[str] = []

    def infer(self, handle, payload):
        aid = handle.agent_id
        self.calls.append(aid)
        if not self.probs_by_agent.get(aid):
            p = 0.5
        else:
            p = float(self.probs_by_agent[aid].pop(0))
        return {
            "agent_id": aid,
            "proba": [1.0 - p, p],
            "prediction": {"label": "malicious" if p >= 0.5 else "benign", "probability": p},
            "uncertainty": {"epistemic": 1.0 - min(abs(2.0 * p - 1.0), 1.0), "aleatoric": 0.0, "total_entropy": 0.0},
            "cost": float(handle.cost),
            "metadata": {},
        }


class IntegratedSystemTests(unittest.IsolatedAsyncioTestCase):
    def _write_files(
        self,
        tmp: Path,
        sequence: List[str],
        max_agents: int,
        *,
        query_policy: str = "strict_cascade",
        fusion_method: str = "logit_pool",
        first_agent: str | None = None,
        first_agent_strategy: str = "explicit",
        min_expected_gain: float = 0.0,
        profile_path: str | None = None,
        costs: Dict[str, float] | None = None,
        query_overrides: Dict[str, object] | None = None,
        decision_overrides: Dict[str, object] | None = None,
    ) -> Path:
        ids = list(dict.fromkeys(sequence or ["agent_a", "agent_b"]))
        if len(ids) < 2:
            ids = ids + ["agent_b"]

        agents = []
        for i, aid in enumerate(ids):
            agents.append(
                {
                    "id": aid,
                    "enabled": True,
                    "endpoint": f"http://example-{aid}",
                    "transport": "http-json",
                    "timeout_ms": 1000,
                    "cost": float((costs or {}).get(aid, 1.0)),
                    "capabilities": ["flow_tabular"],
                    "health_path": "/a2a/health",
                    "infer_path": "/a2a/infer",
                    "capabilities_path": "/a2a/capabilities",
                }
            )

        registry_path = tmp / "agents.yaml"
        registry = {
            "version": "v1",
            "agents": agents,
            "routing": {
                "default_agents": ids,
                "require_healthy": False,
                "fallback_strategy": "skip_unhealthy",
            },
        }
        registry_path.write_text(yaml.dump(registry))

        query_cfg = {
            "policy": query_policy,
            "uncertainty_threshold": 0.6,
            "apply_uncertainty_gate_in_adaptive": True,
            "max_agents": max_agents,
            "min_expected_gain": min_expected_gain,
            "first_agent": first_agent,
            "force_under_target_topup": True,
            "exploration_enabled": False,
            "exploration_seed": 7,
            "exploration_base_rate": 0.0,
            "exploration_max_rate": 0.1,
            "exploration_uncertainty_threshold": 0.6,
            "escalation_ordered": True,
            "utilization_targets": [],
            "utilization_warmup_flows": 500,
        }
        if query_overrides:
            query_cfg.update(query_overrides)

        decision_cfg = {
            "policy": "expected_cost_min",
            "costs": {"c_fn": 500.0, "c_fp": 5.0, "c_h": 5000.0},
            "defer_policy": {
                "enabled": True,
                "uncertainty_threshold": 0.66,
                "margin_from_half": 0.08,
                "require_all_agents_exhausted": True,
            },
            "accuracy_floor_delta": 0.01,
            "cost_calibration": {"enabled": False, "mode": "validation_derived"},
        }
        if decision_overrides:
            decision_cfg.update(decision_overrides)

        cfg = {
            "orchestration": {
                "seed": 7,
                "agent_registry_path": str(registry_path),
                "update_mode": "posterior_first",
                "engine": "deterministic",
                "first_agent_strategy": first_agent_strategy,
                "agent_sequence": sequence,
            },
            "belief": {
                "prior_attack_rate": 0.5,
                "eps": 1e-6,
                "likelihood_sanity_gate": True,
            },
            "fusion": {"method": fusion_method, "agent_weights": {}},
            "decision": decision_cfg,
            "query": query_cfg,
            "voi": {"enabled": False, "rho": 0.7},
            "routing": {
                "profile_path": profile_path,
                "bin_count": 20,
                "min_samples_per_bin": 1,
                "tie_break": "agent_sequence",
                "langgraph_perf_guardrail_overhead": 0.05,
                "parity_tolerance": 1e-6,
            },
            "benchmark": {"reset_state": True, "prediction_source": "probability", "write_manifest": False},
            "a2a": {"retries": 0},
            "state": {"sqlite_path": str(tmp / "state.sqlite")},
            "logging": {"jsonl_path": str(tmp / "flows.jsonl"), "enable_mlflow": False},
        }
        cfg_path = tmp / "config.yaml"
        cfg_path.write_text(yaml.dump(cfg))
        return cfg_path

    async def test_agent_order_follows_configured_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = self._write_files(Path(td), sequence=["agent_a", "agent_b"], max_agents=2)
            system = IntegratedBAOSystem(cfg_path)
            fake = _FakeA2A({"agent_a": [0.5], "agent_b": [0.9]})
            system.a2a = fake

            res = await system.process_flow(
                flow_features={"packet_count": 10.0},
                flow_id="flow-1",
                timestamp=time.time(),
                true_label=1,
            )

            self.assertEqual(fake.calls, ["agent_a", "agent_b"])
            self.assertEqual(res["agents_queried"], ["agent_a", "agent_b"])

    async def test_stop_condition_respects_max_agents(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = self._write_files(Path(td), sequence=["agent_a", "agent_b"], max_agents=1)
            system = IntegratedBAOSystem(cfg_path)
            fake = _FakeA2A({"agent_a": [0.5], "agent_b": [0.9]})
            system.a2a = fake

            res = await system.process_flow(
                flow_features={"packet_count": 10.0},
                flow_id="flow-2",
                timestamp=time.time(),
                true_label=1,
            )

            self.assertEqual(fake.calls, ["agent_a"])
            self.assertEqual(res["agents_queried"], ["agent_a"])

    async def test_dynamic_n_agent_sequence_supported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sequence = ["agent_a", "agent_b", "agent_c", "agent_d"]
            cfg_path = self._write_files(Path(td), sequence=sequence, max_agents=4)
            system = IntegratedBAOSystem(cfg_path)
            fake = _FakeA2A(
                {
                    "agent_a": [0.5],
                    "agent_b": [0.51],
                    "agent_c": [0.49],
                    "agent_d": [0.8],
                }
            )
            system.a2a = fake

            res = await system.process_flow(
                flow_features={"packet_count": 10.0},
                flow_id="flow-4",
                timestamp=time.time(),
                true_label=1,
            )

            self.assertEqual(fake.calls, sequence)
            self.assertEqual(res["agents_queried"], sequence)

    async def test_first_agent_forced_from_query_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sequence = ["agent_b", "agent_a", "agent_c"]
            cfg_path = self._write_files(
                Path(td),
                sequence=sequence,
                max_agents=2,
                first_agent="agent_a",
            )
            system = IntegratedBAOSystem(cfg_path)
            fake = _FakeA2A({"agent_a": [0.5], "agent_b": [0.7], "agent_c": [0.9]})
            system.a2a = fake

            res = await system.process_flow(
                flow_features={"packet_count": 10.0},
                flow_id="flow-first-agent",
                timestamp=time.time(),
                true_label=1,
            )

            self.assertEqual(fake.calls[0], "agent_a")
            self.assertEqual(res["agents_queried"][0], "agent_a")

    async def test_dynamic_cheapest_first_agent_selected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sequence = ["agent_a", "agent_b", "agent_c"]
            cfg_path = self._write_files(
                Path(td),
                sequence=sequence,
                max_agents=1,
                first_agent=None,
                first_agent_strategy="dynamic_cheapest",
                costs={"agent_a": 3.0, "agent_b": 1.0, "agent_c": 5.0},
            )
            system = IntegratedBAOSystem(cfg_path)
            fake = _FakeA2A({"agent_a": [0.9], "agent_b": [0.4], "agent_c": [0.7]})
            system.a2a = fake

            res = await system.process_flow(
                flow_features={"packet_count": 10.0},
                flow_id="flow-cheapest-first",
                timestamp=time.time(),
                true_label=0,
            )

            self.assertEqual(fake.calls[0], "agent_b")
            self.assertEqual(res["agents_queried"][0], "agent_b")

    async def test_adaptive_router_selects_positive_gain_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            profile = {
                "version": "v1",
                "global": {
                    "agent_b": {"accuracy": 1.0, "mean_probability": 0.9},
                    "agent_c": {"accuracy": 1.0, "mean_probability": 0.51},
                },
                "pairwise": {
                    "agent_a": {
                        "agent_b": {
                            "bins": [
                                {
                                    "lo": 0.0,
                                    "hi": 1.0,
                                    "count": 100,
                                    "mean_target_probability": 0.9,
                                    "target_accuracy": 1.0,
                                }
                            ]
                        },
                        "agent_c": {
                            "bins": [
                                {
                                    "lo": 0.0,
                                    "hi": 1.0,
                                    "count": 100,
                                    "mean_target_probability": 0.51,
                                    "target_accuracy": 1.0,
                                }
                            ]
                        },
                    },
                    "agent_b": {
                        "agent_c": {
                            "bins": [
                                {
                                    "lo": 0.0,
                                    "hi": 1.0,
                                    "count": 100,
                                    "mean_target_probability": 0.52,
                                    "target_accuracy": 1.0,
                                }
                            ]
                        }
                    },
                },
            }
            profile_path = tmp / "router_profile.json"
            profile_path.write_text(json.dumps(profile))

            cfg_path = self._write_files(
                tmp,
                sequence=["agent_a", "agent_b", "agent_c"],
                max_agents=3,
                query_policy="adaptive_router",
                fusion_method="handoff_latest",
                first_agent="agent_a",
                min_expected_gain=0.0,
                profile_path=str(profile_path),
            )
            system = IntegratedBAOSystem(cfg_path)
            fake = _FakeA2A({"agent_a": [0.5], "agent_b": [0.9], "agent_c": [0.51]})
            system.a2a = fake

            res = await system.process_flow(
                flow_features={"packet_count": 10.0},
                flow_id="flow-adaptive",
                timestamp=time.time(),
                true_label=1,
            )

            self.assertEqual(fake.calls, ["agent_a", "agent_b"])
            self.assertEqual(res["agents_queried"], ["agent_a", "agent_b"])

    async def test_adaptive_ordered_escalation_blocks_direct_skip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            profile = {
                "version": "v1",
                "global": {
                    "agent_b": {"accuracy": 1.0, "mean_probability": 0.51},
                    "agent_c": {"accuracy": 1.0, "mean_probability": 0.99},
                },
                "pairwise": {
                    "agent_a": {
                        "agent_b": {
                            "bins": [
                                {
                                    "lo": 0.0,
                                    "hi": 1.0,
                                    "count": 100,
                                    "mean_target_probability": 0.51,
                                    "target_accuracy": 1.0,
                                }
                            ]
                        },
                        "agent_c": {
                            "bins": [
                                {
                                    "lo": 0.0,
                                    "hi": 1.0,
                                    "count": 100,
                                    "mean_target_probability": 0.99,
                                    "target_accuracy": 1.0,
                                }
                            ]
                        },
                    }
                },
            }
            profile_path = tmp / "router_profile.json"
            profile_path.write_text(json.dumps(profile))

            cfg_path = self._write_files(
                tmp,
                sequence=["agent_a", "agent_b", "agent_c"],
                max_agents=2,
                query_policy="adaptive_router",
                fusion_method="handoff_latest",
                first_agent="agent_a",
                min_expected_gain=-100.0,
                profile_path=str(profile_path),
                query_overrides={"escalation_ordered": True},
            )
            system = IntegratedBAOSystem(cfg_path)
            fake = _FakeA2A({"agent_a": [0.5], "agent_b": [0.51], "agent_c": [0.99]})
            system.a2a = fake

            res = await system.process_flow(
                flow_features={"packet_count": 10.0},
                flow_id="flow-ordered",
                timestamp=time.time(),
                true_label=1,
            )

            self.assertEqual(fake.calls, ["agent_a", "agent_b"])
            self.assertEqual(res["agents_queried"], ["agent_a", "agent_b"])

    async def test_adaptive_topup_targets_second_and_third_agent_rates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = self._write_files(
                Path(td),
                sequence=["agent_a", "agent_b", "agent_c"],
                max_agents=3,
                query_policy="adaptive_router",
                fusion_method="handoff_latest",
                first_agent="agent_a",
                min_expected_gain=100.0,
                query_overrides={
                    "apply_uncertainty_gate_in_adaptive": False,
                    "exploration_enabled": False,
                    "force_under_target_topup": True,
                    "utilization_warmup_flows": 0,
                    "utilization_targets": [
                        {
                            "agent_id": "agent_b",
                            "min_rate": 0.20,
                            "max_rate": 0.30,
                            "bonus_under": 8.0,
                            "penalty_over": 16.0,
                        },
                        {
                            "agent_id": "agent_c",
                            "min_rate": 0.05,
                            "max_rate": 0.10,
                            "bonus_under": 10.0,
                            "penalty_over": 20.0,
                        },
                    ],
                },
            )
            system = IntegratedBAOSystem(cfg_path)
            fake = _FakeA2A(
                {
                    "agent_a": [0.6] * 200,
                    "agent_b": [0.6] * 200,
                    "agent_c": [0.6] * 200,
                }
            )
            system.a2a = fake

            for i in range(200):
                await system.process_flow(
                    flow_features={"packet_count": 10.0},
                    flow_id=f"flow-topup-{i}",
                    timestamp=time.time(),
                    true_label=1,
                )

            summary = system.get_system_statistics()
            util = summary["agent_utilization"]
            self.assertAlmostEqual(float(util["agent_a"]), 1.0, places=12)
            self.assertGreaterEqual(float(util["agent_b"]), 0.18)
            self.assertLessEqual(float(util["agent_b"]), 0.32)
            self.assertGreaterEqual(float(util["agent_c"]), 0.04)
            self.assertLessEqual(float(util["agent_c"]), 0.12)

    async def test_single_agent_parity_in_posterior_first_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = self._write_files(Path(td), sequence=["agent_a", "agent_b"], max_agents=1)
            system = IntegratedBAOSystem(cfg_path)
            fake = _FakeA2A({"agent_a": [0.83]})
            system.a2a = fake

            res = await system.process_flow(
                flow_features={"packet_count": 10.0},
                flow_id="flow-3",
                timestamp=time.time(),
                true_label=1,
            )

            self.assertAlmostEqual(float(res["compromise_prob"]), 0.83, places=12)

    async def test_defer_policy_applies_when_all_agents_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = self._write_files(
                Path(td),
                sequence=["agent_a", "agent_b"],
                max_agents=2,
                query_policy="adaptive_router",
                fusion_method="handoff_latest",
                first_agent="agent_a",
                min_expected_gain=-100.0,
                query_overrides={
                    "apply_uncertainty_gate_in_adaptive": False,
                    "exploration_enabled": False,
                    "force_under_target_topup": False,
                    "utilization_targets": [],
                },
                decision_overrides={
                    "defer_policy": {
                        "enabled": True,
                        "uncertainty_threshold": 0.66,
                        "margin_from_half": 0.1,
                        "require_all_agents_exhausted": True,
                    }
                },
            )
            system = IntegratedBAOSystem(cfg_path)
            fake = _FakeA2A({"agent_a": [0.5], "agent_b": [0.5]})
            system.a2a = fake

            res = await system.process_flow(
                flow_features={"packet_count": 10.0},
                flow_id="flow-defer",
                timestamp=time.time(),
                true_label=0,
            )

            self.assertEqual(res["action_decision"], "defer")
            self.assertEqual(res["agents_queried"], ["agent_a", "agent_b"])


if __name__ == "__main__":
    unittest.main()
