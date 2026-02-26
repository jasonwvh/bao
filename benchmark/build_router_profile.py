#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator.config import file_sha256, load_orchestrator_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build adaptive-router profile from per-agent replay outputs")
    p.add_argument("--config", default="config/orchestrator_config.utility.yaml", help="Path to orchestrator config YAML")
    p.add_argument("--input-root", default="artifacts/replay/matrix", help="Root containing replay_results_<agent>.json files")
    p.add_argument("--output-path", default="artifacts/replay/router_profile.json", help="Output profile JSON")
    p.add_argument("--agents", default=None, help="Comma-separated agent ids (defaults to orchestration.agent_sequence)")
    p.add_argument("--bin-count", type=int, default=None, help="Probability bins (defaults to routing.bin_count)")
    return p.parse_args()


def _find_replay_file(root: Path, agent_id: str) -> Path:
    candidates = [
        root / agent_id / f"replay_results_{agent_id}.json",
        root / f"replay_results_{agent_id}.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = sorted(root.glob(f"**/replay_results_{agent_id}.json"))
    if not matches:
        raise FileNotFoundError(f"missing replay results for agent {agent_id!r} under {root}")
    return matches[0]


def _load_rows(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"expected list payload in {path}")
    return payload


def _bin_index(p: float, bin_count: int) -> int:
    x = max(0.0, min(1.0, float(p)))
    return min(bin_count - 1, int(x * bin_count))


def main() -> None:
    args = parse_args()
    cfg = load_orchestrator_config(args.config)
    input_root = Path(args.input_root).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.agents is None:
        agents = list(cfg.orchestration.agent_sequence)
    else:
        agents = [a.strip() for a in str(args.agents).split(",") if a.strip()]
    if not agents:
        raise ValueError("at least one agent must be provided")
    bin_count = max(2, int(args.bin_count if args.bin_count is not None else cfg.routing.bin_count))

    replay_paths: Dict[str, Path] = {aid: _find_replay_file(input_root, aid) for aid in agents}
    rows_by_agent = {aid: _load_rows(path) for aid, path in replay_paths.items()}
    by_flow = {aid: {str(r["flow_id"]): r for r in rows} for aid, rows in rows_by_agent.items()}

    common_flow_ids = set(next(iter(by_flow.values())).keys())
    for data in by_flow.values():
        common_flow_ids &= set(data.keys())
    if not common_flow_ids:
        raise RuntimeError("no common flow_ids across agent replay files")
    common = sorted(common_flow_ids)

    global_stats: Dict[str, Dict[str, float | int]] = {}
    for aid in agents:
        rows = by_flow[aid]
        n = len(common)
        correct = 0
        prob_sum = 0.0
        for fid in common:
            item = rows[fid]
            prob = float(item["probability"])
            pred = int(item["prediction"])
            y = int(item["true_label"])
            correct += int(pred == y)
            prob_sum += prob
        global_stats[aid] = {
            "count": n,
            "accuracy": (correct / n) if n else 0.0,
            "mean_probability": (prob_sum / n) if n else 0.5,
        }

    pairwise: Dict[str, Dict[str, Dict]] = {}
    for src in agents:
        pairwise[src] = {}
        src_rows = by_flow[src]
        for tgt in agents:
            if src == tgt:
                continue
            tgt_rows = by_flow[tgt]
            bins = [{"count": 0, "target_prob_sum": 0.0, "target_correct_sum": 0} for _ in range(bin_count)]
            for fid in common:
                p_src = float(src_rows[fid]["probability"])
                idx = _bin_index(p_src, bin_count)
                t = tgt_rows[fid]
                bins[idx]["count"] += 1
                bins[idx]["target_prob_sum"] += float(t["probability"])
                bins[idx]["target_correct_sum"] += int(int(t["prediction"]) == int(t["true_label"]))

            out_bins = []
            for i, b in enumerate(bins):
                lo = i / bin_count
                hi = (i + 1) / bin_count
                count = int(b["count"])
                out_bins.append(
                    {
                        "lo": lo,
                        "hi": hi,
                        "count": count,
                        "mean_target_probability": (b["target_prob_sum"] / count) if count else None,
                        "target_accuracy": (b["target_correct_sum"] / count) if count else None,
                    }
                )
            pairwise[src][tgt] = {
                "bins": out_bins,
                "fallback_mean_target_probability": float(global_stats[tgt]["mean_probability"]),
                "fallback_target_accuracy": float(global_stats[tgt]["accuracy"]),
            }

    payload = {
        "version": "v1",
        "created_at_unix": time.time(),
        "bin_count": bin_count,
        "agents": agents,
        "num_common_flows": len(common),
        "source_files": {aid: str(path) for aid, path in replay_paths.items()},
        "source_file_sha256": {aid: file_sha256(path) for aid, path in replay_paths.items()},
        "global": global_stats,
        "pairwise": pairwise,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Router profile written: {output_path}")
    print(f"Common flows: {len(common)}")


if __name__ == "__main__":
    main()
