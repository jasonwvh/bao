from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from orchestrator.config import load_orchestrator_config


class OrchestratorConfigTests(unittest.TestCase):
    def test_legacy_stage_threshold_keys_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = {
                "orchestration": {
                    "seed": 7,
                    "agent_registry_path": "agents.yaml",
                    "update_mode": "posterior_first",
                    "agent_sequence": ["a1"],
                },
                "belief": {"prior_attack_rate": 0.5, "eps": 1e-6},
                "fusion": {"method": "logit_pool", "agent_weights": {}},
                "decision": {
                    "policy": "expected_cost_min",
                    "costs": {"c_fn": 500.0, "c_fp": 5.0, "c_h": 5000.0},
                    "accuracy_floor_delta": 0.02,
                    "cost_calibration": {"enabled": True, "mode": "validation_derived"},
                },
                "query": {
                    "policy": "adaptive_router",
                    "first_agent": "a1",
                    "uncertainty_threshold": 0.64,
                    "uncertainty_threshold_stage1": 0.2,
                    "uncertainty_threshold_stage2": 0.1,
                    "min_expected_gain": 0.3,
                    "max_agents": 3,
                },
                "voi": {"enabled": True, "rho": 0.7},
                "routing": {
                    "profile_path": str(tmp / "router_profile.json"),
                    "bin_count": 16,
                    "min_samples_per_bin": 8,
                    "tie_break": "agent_sequence",
                },
                "benchmark": {"reset_state": True, "prediction_source": "probability", "write_manifest": False},
                "state": {"sqlite_path": str(tmp / "state.sqlite")},
                "logging": {"jsonl_path": str(tmp / "flows.jsonl"), "enable_mlflow": False},
            }
            path = tmp / "cfg.yaml"
            path.write_text(yaml.dump(cfg))
            loaded = load_orchestrator_config(path)
            self.assertEqual(loaded.query.policy, "adaptive_router")
            self.assertAlmostEqual(float(loaded.query.uncertainty_threshold), 0.64, places=12)
            self.assertEqual(int(loaded.query.max_agents), 3)
            self.assertEqual(loaded.query.first_agent, "a1")
            self.assertAlmostEqual(float(loaded.query.min_expected_gain), 0.3, places=12)
            self.assertEqual(loaded.fusion.method, "logit_pool")
            self.assertAlmostEqual(float(loaded.decision.accuracy_floor_delta), 0.02, places=12)
            self.assertTrue(bool(loaded.decision.cost_calibration.enabled))
            self.assertEqual(loaded.decision.cost_calibration.mode, "validation_derived")
            self.assertEqual(int(loaded.routing.bin_count), 16)


if __name__ == "__main__":
    unittest.main()
