from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

import yaml

from agents.ocsvm.service import OCSVMAgent
from benchmark.metrics import compute_metrics
from orchestrator.data.replay import load_replay_dataset
from orchestrator.integrated_system import IntegratedBAOSystem


class _LocalAgentA2A:
    def __init__(self, agent):
        self.agent = agent

    def infer(self, handle, payload):
        out = self.agent.predict_with_uncertainty(payload.get("flow_features", {}))
        return out


class ReplayRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_agent_replay_matches_agent_probabilities(self) -> None:
        rows = load_replay_dataset("data/UNSW_NB15_testing-set.csv", max_rows=300)
        self.assertGreater(len(rows), 0)

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            registry = {
                "version": "v1",
                "agents": [
                    {
                        "id": "ocsvm",
                        "enabled": True,
                        "endpoint": "http://placeholder",
                        "transport": "http-json",
                        "timeout_ms": 1000,
                        "cost": 1.0,
                        "capabilities": ["flow_tabular", "unsw_nb15", "anomaly_score", "one_class"],
                        "health_path": "/a2a/health",
                        "infer_path": "/a2a/infer",
                        "capabilities_path": "/a2a/capabilities",
                    }
                ],
                "routing": {
                    "default_agents": ["ocsvm"],
                    "require_healthy": False,
                    "fallback_strategy": "skip_unhealthy",
                },
            }
            registry_path = tmp / "agents.yaml"
            registry_path.write_text(yaml.dump(registry))

            cfg = {
                "orchestration": {
                    "seed": 7,
                    "agent_registry_path": str(registry_path),
                    "update_mode": "posterior_first",
                    "agent_sequence": ["ocsvm"],
                },
                "belief": {"prior_attack_rate": 0.5, "eps": 1e-6, "likelihood_sanity_gate": True},
                "fusion": {"method": "logit_pool", "agent_weights": {}},
                "decision": {"policy": "expected_cost_min", "costs": {"c_fn": 500.0, "c_fp": 5.0, "c_h": 5000.0}},
                "query": {"uncertainty_threshold": 0.6, "max_agents": 1},
                "voi": {"enabled": True, "rho": 0.7},
                "benchmark": {"reset_state": True, "prediction_source": "probability", "write_manifest": False},
                "a2a": {"retries": 0},
                "state": {"sqlite_path": str(tmp / "state.sqlite")},
                "logging": {"jsonl_path": str(tmp / "flows.jsonl"), "enable_mlflow": False},
            }
            cfg_path = tmp / "config.yaml"
            cfg_path.write_text(yaml.dump(cfg))

            system = IntegratedBAOSystem(cfg_path)

            local_agent = OCSVMAgent("agents/ocsvm/models/ocsvm.pkl", cost=1.0)
            system.a2a = _LocalAgentA2A(local_agent)

            labels = []
            probs_agent = []
            probs_bao = []

            for row in rows:
                out = local_agent.predict_with_uncertainty(row["flow_features"])
                p_agent = float(out["proba"][1])

                res = await system.process_flow(
                    flow_features=row["flow_features"],
                    flow_id=row["flow_id"],
                    timestamp=row.get("timestamp") or time.time(),
                    true_label=row.get("true_label"),
                )
                p_bao = float(res["compromise_prob"])

                labels.append(int(row["true_label"]))
                probs_agent.append(p_agent)
                probs_bao.append(p_bao)

            for p_a, p_b in zip(probs_agent, probs_bao):
                self.assertAlmostEqual(p_a, p_b, places=12)

            preds_agent = [1 if p >= 0.5 else 0 for p in probs_agent]
            preds_bao = [1 if p >= 0.5 else 0 for p in probs_bao]

            m_agent = compute_metrics(preds_agent, labels, probs_agent, [1.0] * len(labels), approach="agent")
            m_bao = compute_metrics(preds_bao, labels, probs_bao, [1.0] * len(labels), approach="bao")

            self.assertEqual(m_agent["accuracy"], m_bao["accuracy"])
            self.assertEqual(m_agent["precision"], m_bao["precision"])
            self.assertEqual(m_agent["recall"], m_bao["recall"])
            self.assertEqual(m_agent["f1"], m_bao["f1"])


if __name__ == "__main__":
    unittest.main()
