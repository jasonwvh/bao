from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import yaml

from orchestrator.belief import BayesianBelief, reliability_weight_from_beta_params
from orchestrator.decisioning import DecisionCosts, expected_cost_reduction_from_likelihood_model
from orchestrator.runtime import BAORuntime


def _write_registry(path: Path) -> Path:
    payload = {
        "version": "v1",
        "agents": [
            {
                "id": "ocsvm",
                "enabled": True,
                "endpoint": "http://localhost:8081",
                "timeout_ms": 1000,
                "cost": 1.0,
                "capabilities": ["flow_tabular"],
            },
            {
                "id": "lstm_autoencoder",
                "enabled": True,
                "endpoint": "http://localhost:8082",
                "timeout_ms": 1000,
                "cost": 0.1,
                "capabilities": ["flow_tabular"],
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
        "belief": {
            "prior_attack_rate": 0.5,
            "eps": 1e-6,
            "update_mode": "likelihood_ratio",
            "reliability_strength": 1.0,
        },
        "fusion": {
            "uncertainty_weight_gamma": 1.5,
            "weight_floor": 0.1,
            "agent_weights": {"ocsvm": 1.0, "lstm_autoencoder": 1.0},
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
        },
        "voi": {
            "enabled": True,
            "rho": 1.0,
            "mode": "expected_cost_reduction",
            "min_net_gain": 0.0,
        },
        "metrics": {"warnings_enabled": True},
        "benchmark": {"reset_state": True, "write_manifest": True},
        "a2a": {"retries": 0},
        "state": {"sqlite_path": str(sqlite_path)},
    }
    path.write_text(yaml.dump(payload))
    return path


class BayesianAndVOITests(unittest.TestCase):
    def test_likelihood_ratio_update_direction(self) -> None:
        belief = BayesianBelief.new("flow-1", prior_attack_rate=0.5, eps=1e-6)
        p0 = belief.probability()
        belief.update_from_likelihood_ratio(0.8, 0.2, k=1.0)
        p1 = belief.probability()
        self.assertGreater(p1, p0)

        belief = BayesianBelief.new("flow-2", prior_attack_rate=0.5, eps=1e-6)
        p0 = belief.probability()
        belief.update_from_likelihood_ratio(0.2, 0.8, k=1.0)
        p1 = belief.probability()
        self.assertLess(p1, p0)

    def test_reliability_scaling_weight_monotonic(self) -> None:
        low = reliability_weight_from_beta_params(alpha=2.0, beta=2.0, reliability_strength=1.0)
        high = reliability_weight_from_beta_params(alpha=20.0, beta=2.0, reliability_strength=1.0)
        self.assertGreater(high, low)

    def test_voi_gate_blocks_negative_and_allows_positive_net_gain(self) -> None:
        class _FakeA2A:
            def capabilities(self, handle):
                return {
                    "agent_id": handle.agent_id,
                    "capabilities": ["flow_tabular"],
                    "cost": float(handle.cost),
                    "metadata": {
                        "likelihood_model": {
                            "bin_edges": [0.0, 0.5, 1.0],
                            "p_obs_given_attack_bins": [0.2, 0.8],
                            "p_obs_given_clean_bins": [0.8, 0.2],
                        }
                    },
                }

            def metadata(self):
                return {"transport": "fake"}

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            registry = _write_registry(root / "agents.yaml")
            config = _write_config(root / "config.yaml", registry, root / "state.sqlite")
            runtime = BAORuntime(str(config))
            runtime.a2a = _FakeA2A()

            # Around p=0.5 this proxy has near-zero expected reduction, so net gain is negative.
            should_escalate, gain = runtime._should_escalate(  # noqa: SLF001
                p_mal=0.5,
                combined_uncertainty=math.log(2.0),
                next_agent_cost=1.0,
                next_agent_id="lstm_autoencoder",
            )
            self.assertFalse(should_escalate)
            self.assertLess(gain, 0.0)

            # Near the reject/accept boundary, the same cost can produce positive expected net gain.
            should_escalate, gain = runtime._should_escalate(  # noqa: SLF001
                p_mal=0.08,
                combined_uncertainty=math.log(2.0),
                next_agent_cost=0.1,
                next_agent_id="lstm_autoencoder",
            )
            self.assertTrue(should_escalate)
            self.assertGreater(gain, 0.0)

    def test_empirical_voi_reduction_uses_likelihood_bins(self) -> None:
        reduction = expected_cost_reduction_from_likelihood_model(
            p_mal=0.08,
            costs=DecisionCosts(c_fn=25.0, c_fp=2.0, c_h=2.0),
            likelihood_model={
                "bin_edges": [0.0, 0.5, 1.0],
                "p_obs_given_attack_bins": [0.2, 0.8],
                "p_obs_given_clean_bins": [0.8, 0.2],
            },
            reliability_weight=1.0,
            rho=1.0,
        )
        self.assertGreater(reduction, 0.0)


if __name__ == "__main__":
    unittest.main()
