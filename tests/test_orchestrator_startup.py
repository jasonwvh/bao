from __future__ import annotations

import json
from pathlib import Path

import pytest

from bao.integrated_system import IntegratedBAOSystem


def _cfg(tmp_path: Path, registry_path: Path):
    cfg = {
        "thresholds": {"p_accept": 0.3, "p_reject": 0.7, "uncertainty": 0.6},
        "costs": {"c_fn": 100.0, "c_fp": 1.0, "c_h": 5.0, "agent_costs": {"agent_a": 1.0, "agent_b": 5.0}},
        "orchestration": {"max_iterations": 2, "learning_rate": 0.25, "seed": 7, "agent_registry_path": str(registry_path)},
        "a2a": {"retries": 0, "a2a_version": "v1"},
        "state": {"sqlite_path": str(tmp_path / "state.sqlite")},
        "voi": {"use_surrogate": True, "allow_exact": False},
        "drift": {"window": 4, "threshold": 0.0001},
        "logging": {"jsonl_path": str(tmp_path / "flows.jsonl"), "enable_mlflow": False},
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return p


def test_startup_succeeds_with_healthy_defaults(tmp_path: Path, a2a_stack):
    bao = IntegratedBAOSystem(_cfg(tmp_path, a2a_stack["registry_path"]))
    assert "agent_a" in bao.default_agents


def test_startup_fails_when_required_defaults_unhealthy(tmp_path: Path):
    reg = tmp_path / "agents.yaml"
    reg.write_text(
        """
version: v1
agents:
  - id: agent_a
    enabled: true
    endpoint: http://127.0.0.1:65530
    transport: http-json
    timeout_ms: 100
    cost: 1.0
    capabilities: [flow_tabular]
    health_path: /a2a/health
    infer_path: /a2a/infer
routing:
  default_agents: [agent_a]
  require_healthy: true
  max_parallel: 1
  fallback_strategy: skip_unhealthy
""".strip()
    )

    with pytest.raises(RuntimeError):
        IntegratedBAOSystem(_cfg(tmp_path, reg))
