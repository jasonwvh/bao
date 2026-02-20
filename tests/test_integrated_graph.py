from __future__ import annotations

import asyncio
import json
from pathlib import Path

from bao.integrated_system import IntegratedBAOSystem


def _config(tmp_path: Path, registry_path: Path) -> Path:
    cfg = {
        "thresholds": {"p_accept": 0.3, "p_reject": 0.7, "uncertainty": 0.6},
        "costs": {"c_fn": 100.0, "c_fp": 1.0, "c_h": 5.0, "agent_costs": {"agent_a": 1.0, "agent_b": 5.0}},
        "orchestration": {
            "max_iterations": 2,
            "learning_rate": 0.25,
            "seed": 7,
            "agent_registry_path": str(registry_path),
        },
        "a2a": {"retries": 0, "a2a_version": "v1"},
        "state": {"sqlite_path": str(tmp_path / "state.sqlite")},
        "voi": {"use_surrogate": True, "allow_exact": False},
        "drift": {"window": 4, "threshold": 0.0001},
        "logging": {"jsonl_path": str(tmp_path / "flows.jsonl"), "enable_mlflow": False},
    }
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    return path


def test_graph_contains_check_uncertainty_and_state_keys_initialized(tmp_path: Path, a2a_stack):
    bao = IntegratedBAOSystem(_config(tmp_path, a2a_stack["registry_path"]))
    assert "check_uncertainty" in bao.graph_nodes

    result = asyncio.run(
        bao.process_flow(
            flow_features={"packet_count": 20, "byte_count": 5000, "flow_duration": 1.0, "dst_port": 80},
            flow_id="f1",
            timestamp=1.0,
            true_label=0,
        )
    )

    for key in [
        "agents_available",
        "max_iterations",
        "inference_time_ms",
        "confidence",
        "decision",
        "compromise_prob",
    ]:
        assert key in result


def test_threshold_immutability_across_flows(tmp_path: Path, a2a_stack):
    bao = IntegratedBAOSystem(_config(tmp_path, a2a_stack["registry_path"]))
    baseline = bao.thresholds.uncertainty

    asyncio.run(
        bao.process_flow(
            flow_features={"packet_count": 180, "byte_count": 90000, "flow_duration": 10.0, "dst_port": 22},
            flow_id="f2",
            timestamp=2.0,
            true_label=1,
        )
    )
    asyncio.run(
        bao.process_flow(
            flow_features={"packet_count": 22, "byte_count": 6000, "flow_duration": 1.2, "dst_port": 80},
            flow_id="f3",
            timestamp=3.0,
            true_label=0,
        )
    )
    assert bao.thresholds.uncertainty == baseline
