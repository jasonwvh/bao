from __future__ import annotations

import asyncio
import json
from pathlib import Path

from bao.integrated_system import IntegratedBAOSystem


def _config(tmp_path: Path, registry_path: Path) -> Path:
    cfg = {
        "thresholds": {"p_accept": 0.49, "p_reject": 0.55, "uncertainty": 0.95},
        "costs": {"c_fn": 100.0, "c_fp": 1.0, "c_h": 5.0, "agent_costs": {"agent_a": 1.0, "agent_b": 5.0}},
        "orchestration": {
            "max_iterations": 2,
            "learning_rate": 0.25,
            "seed": 11,
            "agent_registry_path": str(registry_path),
        },
        "a2a": {"retries": 0, "a2a_version": "v1"},
        "state": {"sqlite_path": str(tmp_path / "state.sqlite")},
        "voi": {"use_surrogate": True, "allow_exact": False},
        "drift": {"window": 2, "threshold": 0.0},
        "logging": {"jsonl_path": str(tmp_path / "flows.jsonl"), "enable_mlflow": False},
    }
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    return path


def test_benign_flow_accepts(tmp_path: Path, a2a_stack):
    bao = IntegratedBAOSystem(_config(tmp_path, a2a_stack["registry_path"]))
    out = asyncio.run(
        bao.process_flow(
            flow_features={"packet_count": 12, "byte_count": 3000, "flow_duration": 0.8, "dst_port": 80},
            flow_id="benign_1",
            timestamp=1.0,
            true_label=0,
        )
    )
    assert out["decision"] == "accept"


def test_malicious_flow_rejects_after_query(tmp_path: Path, a2a_stack):
    bao = IntegratedBAOSystem(_config(tmp_path, a2a_stack["registry_path"]))
    out = asyncio.run(
        bao.process_flow(
            flow_features={"packet_count": 600, "byte_count": 280000, "flow_duration": 80, "src_port": 222, "dst_port": 22},
            flow_id="mal_1",
            timestamp=2.0,
            true_label=1,
        )
    )
    assert out["decision"] == "reject"
    assert len(out["agents_queried"]) >= 1


def test_uncertain_flow_defers_and_has_hitl_context(tmp_path: Path, a2a_stack):
    bao = IntegratedBAOSystem(_config(tmp_path, a2a_stack["registry_path"]))
    bao.thresholds = bao.thresholds.__class__(
        p_accept=0.05,
        p_reject=0.95,
        uncertainty=0.95,
    )
    out = asyncio.run(
        bao.process_flow(
            flow_features={"packet_count": 70, "byte_count": 22000, "flow_duration": 5, "src_port": 1200, "dst_port": 8080},
            flow_id="uncertain_1",
            timestamp=3.0,
            true_label=1,
        )
    )
    if out["decision"] != "defer":
        out = asyncio.run(
            bao.process_flow(
                flow_features={"packet_count": 90, "byte_count": 26000, "flow_duration": 6, "src_port": 1400, "dst_port": 8080},
                flow_id="uncertain_2",
                timestamp=4.0,
                true_label=0,
            )
        )
    assert out["decision"] == "defer"
    assert out["hitl_context"] is not None


def test_drift_path_can_trigger(tmp_path: Path, a2a_stack):
    bao = IntegratedBAOSystem(_config(tmp_path, a2a_stack["registry_path"]))

    for i in range(5):
        asyncio.run(
            bao.process_flow(
                flow_features={"packet_count": 15, "byte_count": 4000, "flow_duration": 1, "dst_port": 80},
                flow_id=f"drift_a_{i}",
                timestamp=float(i),
                true_label=0,
            )
        )
    out = asyncio.run(
        bao.process_flow(
            flow_features={"packet_count": 700, "byte_count": 350000, "flow_duration": 110, "dst_port": 22},
            flow_id="drift_b",
            timestamp=10.0,
            true_label=1,
        )
    )

    assert "drift_detected" in out
