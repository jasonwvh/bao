from __future__ import annotations

from pathlib import Path

import pytest

from bao.control.registry import RegistryValidationError, load_registry, to_runtime_handles


def test_registry_load_and_runtime_handles():
    reg = load_registry("/Users/jasonwvh/Documents/projects/bao/config/agents.yaml")
    handles = to_runtime_handles(reg)

    assert reg["version"] == "v1"
    assert "agent_a" in handles
    assert "agent_b" in handles
    assert "agent_c" in handles
    assert "agent_d" in handles
    assert handles["agent_a"].transport == "http-json"


def test_registry_rejects_local_endpoint(tmp_path: Path):
    p = tmp_path / "bad_agents.yaml"
    p.write_text(
        """
version: v1
agents:
  - id: agent_a
    enabled: true
    endpoint: local://agent_a
    transport: http-json
    timeout_ms: 100
    cost: 1.0
    capabilities: [x]
    health_path: /a2a/health
    infer_path: /a2a/infer
routing:
  default_agents: [agent_a]
  require_healthy: true
  max_parallel: 1
  fallback_strategy: skip_unhealthy
""".strip()
    )
    with pytest.raises(RegistryValidationError):
        load_registry(p)
