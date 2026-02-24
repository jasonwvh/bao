from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from typing import Dict, List

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
    def _write_files(self, tmp: Path, sequence: List[str], max_agents: int) -> Path:
        registry_path = tmp / "agents.yaml"
        registry = {
            "version": "v1",
            "agents": [
                {
                    "id": "agent_a",
                    "enabled": True,
                    "endpoint": "http://example-a",
                    "transport": "http-json",
                    "timeout_ms": 1000,
                    "cost": 1.0,
                    "capabilities": ["flow_tabular"],
                    "health_path": "/a2a/health",
                    "infer_path": "/a2a/infer",
                    "capabilities_path": "/a2a/capabilities",
                },
                {
                    "id": "agent_b",
                    "enabled": True,
                    "endpoint": "http://example-b",
                    "transport": "http-json",
                    "timeout_ms": 1000,
                    "cost": 1.0,
                    "capabilities": ["flow_tabular"],
                    "health_path": "/a2a/health",
                    "infer_path": "/a2a/infer",
                    "capabilities_path": "/a2a/capabilities",
                },
            ],
            "routing": {
                "default_agents": ["agent_a", "agent_b"],
                "require_healthy": False,
                "fallback_strategy": "skip_unhealthy",
            },
        }
        registry_path.write_text(yaml.dump(registry))

        cfg = {
            "orchestration": {
                "seed": 7,
                "agent_registry_path": str(registry_path),
                "update_mode": "posterior_first",
                "agent_sequence": sequence,
            },
            "belief": {
                "prior_attack_rate": 0.5,
                "eps": 1e-6,
                "likelihood_sanity_gate": True,
            },
            "fusion": {"method": "logit_pool", "agent_weights": {}},
            "decision": {"policy": "expected_cost_min", "costs": {"c_fn": 500.0, "c_fp": 5.0, "c_h": 5000.0}},
            "query": {"uncertainty_threshold": 0.6, "max_agents": max_agents},
            "voi": {"enabled": True, "rho": 0.7},
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


if __name__ == "__main__":
    unittest.main()
