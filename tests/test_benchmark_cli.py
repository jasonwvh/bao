from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

import main as benchmark_cli
from orchestrator.a2a import A2AClient


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
            {
                "id": "wgan_gp",
                "enabled": True,
                "endpoint": "http://localhost:8084",
                "transport": "http-json",
                "timeout_ms": 1000,
                "cost": 5.0,
                "capabilities": ["flow_tabular"],
                "health_path": "/a2a/health",
                "infer_path": "/a2a/infer",
                "capabilities_path": "/a2a/capabilities",
            },
        ],
    }
    path.write_text(yaml.dump(payload))
    return path


def _write_config(path: Path, registry_path: Path) -> Path:
    payload = {
        "orchestration": {
            "seed": 7,
            "agent_registry_path": str(registry_path),
            "agent_sequence": ["ocsvm", "lstm_autoencoder", "wgan_gp"],
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
            "agent_weights": {"ocsvm": 0.6, "lstm_autoencoder": 1.4, "wgan_gp": 1.0},
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
            "max_agents": 3,
        },
        "voi": {"enabled": True, "rho": 0.7, "mode": "expected_cost_reduction", "min_net_gain": 0.0},
        "metrics": {"warnings_enabled": True},
        "benchmark": {"reset_state": True, "write_manifest": True},
        "a2a": {"retries": 0},
        "state": {"sqlite_path": "./unused.sqlite"},
    }
    path.write_text(yaml.dump(payload))
    return path


def _write_dataset(path: Path) -> Path:
    path.write_text(
        "flow_id,label,attack_cat,dur,spkts,dpkts,sbytes,dbytes,proto,service,state\n"
        "f1,0,Normal,1,1,2,10,20,tcp,http,FIN\n"
        "f2,1,DoS,2,3,4,20,30,tcp,http,INT\n"
        "f3,0,Normal,1,1,1,10,10,udp,dns,CON\n"
    )
    return path


def _fake_infer(self, handle, payload):
    probs = {
        "ocsvm": 0.52,
        "lstm_autoencoder": 0.79,
        "wgan_gp": 0.73,
    }
    p = float(probs.get(handle.agent_id, 0.5))
    ep = 1.0 - min(abs(2.0 * p - 1.0), 1.0)
    return {
        "agent_id": handle.agent_id,
        "proba": [1.0 - p, p],
        "prediction": {"label": "malicious" if p >= 0.5 else "benign", "probability": p},
        "uncertainty": {"epistemic": ep, "aleatoric": 0.0, "total_entropy": 0.0},
        "likelihoods": {
            "p_obs_given_attack": max(1e-6, p),
            "p_obs_given_clean": max(1e-6, 1.0 - p),
        },
        "cost": float(handle.cost),
    }


def _fake_capabilities(self, handle):
    return {
        "agent_id": handle.agent_id,
        "capabilities": ["flow_tabular"],
        "cost": float(handle.cost),
        "metadata": {
            "threshold_probability": 0.5,
            "likelihood_model": {
                "bin_edges": [0.0, 0.5, 1.0],
                "p_obs_given_attack_bins": [0.2, 0.8],
                "p_obs_given_clean_bins": [0.8, 0.2],
            },
        },
    }


class BenchmarkCliTests(unittest.TestCase):
    def test_bao_writes_consistent_artifacts_without_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = _write_dataset(root / "dataset.csv")
            registry = _write_registry(root / "agents.yaml")
            config = _write_config(root / "config.yaml", registry)
            out_root = root / "runs"

            argv = [
                "main.py",
                "--mode",
                "bao",
                "--dataset",
                str(dataset),
                "--config",
                str(config),
                "--output-dir",
                str(out_root),
                "--run-id",
                "run_bao",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(A2AClient, "infer", new=_fake_infer),
                patch.object(A2AClient, "capabilities", new=_fake_capabilities),
            ):
                benchmark_cli.main()

            run_dir = out_root / "run_bao"
            self.assertTrue((run_dir / "benchmark.json").exists())
            self.assertTrue((run_dir / "replay_results.json").exists())
            self.assertTrue((run_dir / "run_manifest.json").exists())
            self.assertTrue((run_dir / "state.sqlite").exists())
            self.assertFalse(any(run_dir.glob("*.jsonl")))
            self.assertFalse((run_dir / "flows.jsonl").exists())

    def test_all_mode_replay_has_approach_key(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = _write_dataset(root / "dataset.csv")
            registry = _write_registry(root / "agents.yaml")
            config = _write_config(root / "config.yaml", registry)
            out_root = root / "runs"

            argv = [
                "main.py",
                "--mode",
                "all",
                "--dataset",
                str(dataset),
                "--config",
                str(config),
                "--output-dir",
                str(out_root),
                "--run-id",
                "run_all",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(A2AClient, "infer", new=_fake_infer),
                patch.object(A2AClient, "capabilities", new=_fake_capabilities),
            ):
                benchmark_cli.main()

            replay = json.loads((out_root / "run_all" / "replay_results.json").read_text())
            self.assertGreater(len(replay), 0)
            self.assertTrue(all("approach" in row for row in replay))
            self.assertTrue(all("family" in row for row in replay))
            self.assertTrue(all("metadata" in row for row in replay))
            benchmark = json.loads((out_root / "run_all" / "benchmark.json").read_text())
            self.assertIn("comparison", benchmark)
            self.assertIn("warnings", benchmark)
            self.assertIn("thresholded_single_agent", benchmark["results"])
            self.assertIn("cost_aware_single_agent", benchmark["results"])
            self.assertIn("thresholded_ensemble", benchmark["results"])
            self.assertIn("cost_aware_ensemble", benchmark["results"])
            self.assertIn("bao", benchmark["results"])
            bao = benchmark["results"]["bao"]
            self.assertIn("ece", bao)
            self.assertIn("brier", bao)
            self.assertIn("group_metrics", bao)
            self.assertIn("attack_cat_recall_gap", bao)
            self.assertIn("deltas_vs_best_thresholded_single_agent", bao)
            self.assertIn("average_probability", benchmark["results"]["thresholded_ensemble"])
            self.assertIn("majority_vote", benchmark["results"]["thresholded_ensemble"])

    def test_ensemble_mode_writes_static_ensemble_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = _write_dataset(root / "dataset.csv")
            registry = _write_registry(root / "agents.yaml")
            config = _write_config(root / "config.yaml", registry)
            out_root = root / "runs"

            argv = [
                "main.py",
                "--mode",
                "ensemble",
                "--ensemble-method",
                "average_probability",
                "--baseline-family",
                "thresholded_single_agent",
                "--dataset",
                str(dataset),
                "--config",
                str(config),
                "--output-dir",
                str(out_root),
                "--run-id",
                "run_ensemble",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(A2AClient, "infer", new=_fake_infer),
                patch.object(A2AClient, "capabilities", new=_fake_capabilities),
            ):
                benchmark_cli.main()

            benchmark = json.loads((out_root / "run_ensemble" / "benchmark.json").read_text())
            replay = json.loads((out_root / "run_ensemble" / "replay_results.json").read_text())
            self.assertEqual(benchmark["approach"], "average_probability")
            self.assertEqual(benchmark["family"], "thresholded_ensemble")
            self.assertIn("ece", benchmark)
            self.assertIn("brier", benchmark)
            self.assertTrue(all("agent_probabilities" in row for row in replay))

    def test_each_run_gets_new_sqlite_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = _write_dataset(root / "dataset.csv")
            registry = _write_registry(root / "agents.yaml")
            config = _write_config(root / "config.yaml", registry)
            out_root = root / "runs"

            argv1 = [
                "main.py",
                "--mode",
                "bao",
                "--dataset",
                str(dataset),
                "--config",
                str(config),
                "--output-dir",
                str(out_root),
                "--run-id",
                "run_1",
            ]
            argv2 = [
                "main.py",
                "--mode",
                "bao",
                "--dataset",
                str(dataset),
                "--config",
                str(config),
                "--output-dir",
                str(out_root),
                "--run-id",
                "run_2",
            ]

            with (
                patch.object(A2AClient, "infer", new=_fake_infer),
                patch.object(A2AClient, "capabilities", new=_fake_capabilities),
            ):
                with patch.object(sys, "argv", argv1):
                    benchmark_cli.main()
                with patch.object(sys, "argv", argv2):
                    benchmark_cli.main()

            manifest_1 = json.loads((out_root / "run_1" / "run_manifest.json").read_text())
            manifest_2 = json.loads((out_root / "run_2" / "run_manifest.json").read_text())
            self.assertNotEqual(manifest_1["sqlite_path"], manifest_2["sqlite_path"])
            self.assertTrue(Path(manifest_1["sqlite_path"]).exists())
            self.assertTrue(Path(manifest_2["sqlite_path"]).exists())


if __name__ == "__main__":
    unittest.main()
