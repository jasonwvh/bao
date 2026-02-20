from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_replay_cli_runs_and_writes_artifacts(tmp_path: Path, a2a_stack):
    ds = tmp_path / "replay.csv"
    with open(ds, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["flow_id", "packet_count", "byte_count", "flow_duration", "src_port", "dst_port", "label"],
        )
        w.writeheader()
        for i in range(12):
            w.writerow(
                {
                    "flow_id": f"f{i}",
                    "packet_count": 20 + i * 10,
                    "byte_count": 5000 + i * 1000,
                    "flow_duration": 1 + i * 0.5,
                    "src_port": 1000 + i,
                    "dst_port": 22 if i % 3 == 0 else 80,
                    "label": 1 if i % 4 == 0 else 0,
                }
            )

    cfg = json.loads(Path("/Users/jasonwvh/Documents/projects/bao/config/bao_config.json").read_text())
    cfg["orchestration"]["agent_registry_path"] = str(a2a_stack["registry_path"])
    cfg["state"]["sqlite_path"] = str(tmp_path / "state.sqlite")
    cfg["logging"]["jsonl_path"] = str(tmp_path / "flows.jsonl")
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    out_dir = tmp_path / "out"

    cmd = [
        sys.executable,
        "/Users/jasonwvh/Documents/projects/bao/scripts/run_replay.py",
        "--dataset",
        str(ds),
        "--config",
        str(cfg_path),
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    assert (out_dir / "replay_results.jsonl").exists()
    assert (out_dir / "summary.json").exists()
