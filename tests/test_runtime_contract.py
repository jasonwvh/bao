from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

import yaml

from orchestrator.runtime import BAORuntime


class _FakeA2A:
    def __init__(self, probs):
        self.probs = probs

    def infer(self, handle, payload):
        p = float(self.probs.get(handle.agent_id, 0.5))
        ep = 1.0 - min(abs(2.0 * p - 1.0), 1.0)
        return {
            "agent_id": handle.agent_id,
            "proba": [1.0 - p, p],
            "prediction": {"label": "malicious" if p >= 0.5 else "benign", "probability": p},
            "uncertainty": {"epistemic": ep, "aleatoric": 0.0, "total_entropy": 0.0},
            "cost": float(handle.cost),
        }

    def metadata(self):
        return {"official_a2a_sdk_available": False, "transport": "fake"}


def _write_registry(path: Path) -> Path:
    payload = {
        "version": "v1",
        "agents": [
            {
                "id": "ocsvm",
                "enabled": True,
                "endpoint": "http://localhost:8081",
                "transport": "http-json",
                "timeout_ms": 1000,
                "cost": 1.0,
                "capabilities": ["flow_tabular"],
                "health_path": "/a2a/health",
                "infer_path": "/a2a/infer",
                "capabilities_path": "/a2a/capabilities",
            },
            {
                "id": "lstm_autoencoder",
                "enabled": True,
                "endpoint": "http://localhost:8082",
                "transport": "http-json",
                "timeout_ms": 1000,
                "cost": 3.0,
                "capabilities": ["flow_tabular"],
                "health_path": "/a2a/health",
                "infer_path": "/a2a/infer",
                "capabilities_path": "/a2a/capabilities",
            },
        ],
    }
    path.write_text(yaml.dump(payload))
    return path


def _write_config(path: Path, registry_path: Path, sqlite_path: Path) -> Path:
    payload = {
        "orchestration": {
            "seed": 7,
            "agent_registry_path": str(registry_path),
            "agent_sequence": ["ocsvm", "lstm_autoencoder"],
        },
        "belief": {"prior_attack_rate": 0.5, "eps": 1e-6},
        "fusion": {
            "uncertainty_weight_gamma": 1.5,
            "weight_floor": 0.1,
            "agent_weights": {"ocsvm": 0.6, "lstm_autoencoder": 1.4},
        },
        "decision": {
            "costs": {"c_fn": 25.0, "c_fp": 2.0, "c_h": 2.0},
            "defer_policy": {
                "enabled": True,
                "uncertainty_threshold": 0.69308,
                "margin_from_half": 0.01,
                "require_all_agents_exhausted": True,
            },
        },
        "query": {
            "first_agent": "ocsvm",
            "uncertainty_threshold": 0.6,
            "max_agents": 2,
            "min_expected_gain": -3.0,
        },
        "voi": {"enabled": True, "rho": 0.7},
        "benchmark": {"reset_state": True, "write_manifest": True},
        "a2a": {"retries": 0},
        "state": {"sqlite_path": str(sqlite_path)},
    }
    path.write_text(yaml.dump(payload))
    return path


class RuntimeContractTests(unittest.TestCase):
    def test_single_decision_field(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            registry = _write_registry(root / "agents.yaml")
            config = _write_config(root / "config.yaml", registry, root / "state.sqlite")

            runtime = BAORuntime(str(config))
            runtime.a2a = _FakeA2A({"ocsvm": 0.5, "lstm_autoencoder": 0.85})

            res = runtime.process_flow(
                flow_features={"dur": 1.0, "spkts": 1.0, "dpkts": 1.0},
                flow_id="flow-1",
                timestamp=time.time(),
                true_label=1,
            )

            self.assertIn("decision", res)
            self.assertNotIn("action_decision", res)
            self.assertIn(res["decision"], {"accept", "reject", "defer"})
            self.assertIn("combined_uncertainty", res)
            self.assertIn("agents_queried", res)


if __name__ == "__main__":
    unittest.main()
