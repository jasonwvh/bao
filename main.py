#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator.data.replay import load_replay_dataset
from orchestrator.integrated_system import IntegratedBAOSystem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run orchestrator replay pipeline")
    p.add_argument("--dataset", default="data/UNSW_NB15_testing-set.csv", help="CSV or parquet with label column")
    p.add_argument("--config", default="config/orchestrator_config.yaml", help="Path to orchestrator config YAML")
    p.add_argument("--max-flows", type=int, default=0, help="Max flows to process (0=all)")
    p.add_argument("--output-dir", default="artifacts/replay", help="Output directory")
    p.add_argument("--seed", type=int, default=7, help="Seed")
    return p.parse_args()


async def _run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_replay_dataset(args.dataset, max_rows=(args.max_flows or None))
    if not rows:
        raise RuntimeError("No rows loaded from dataset")

    config = yaml.safe_load(Path(args.config).read_text())
    config.setdefault("orchestration", {})["seed"] = args.seed
    if "agent_registry_path" in config.get("orchestration", {}):
        reg_path = Path(config["orchestration"]["agent_registry_path"])
        if not reg_path.is_absolute():
            reg_path = (Path(args.config).resolve().parent / reg_path).resolve()
        config["orchestration"]["agent_registry_path"] = str(reg_path)

    runtime_config_path = output_dir / "effective_orchestrator_config.yaml"
    runtime_config_path.write_text(yaml.dump(config, default_flow_style=False))

    system = IntegratedBAOSystem(config_path=runtime_config_path)

    results_path = output_dir / "replay_results.jsonl"
    if results_path.exists():
        results_path.unlink()

    for row in rows:
        res = await system.process_flow(
            flow_features=row["flow_features"],
            flow_id=row["flow_id"],
            timestamp=row.get("timestamp") or time.time(),
            true_label=row.get("true_label"),
        )
        compact = {
            "flow_id": row["flow_id"],
            "decision": res.get("decision"),
            "compromise_prob": res.get("compromise_prob"),
            "epistemic_uncertainty": res.get("epistemic_uncertainty"),
            "cumulative_cost": res.get("cumulative_cost"),
            "agents_queried": res.get("agents_queried"),
        }
        with open(results_path, "a") as f:
            f.write(json.dumps(compact) + "\n")

    summary = system.get_system_statistics()
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Processed {summary['flows_processed']} flows")
    print(f"Replay output: {results_path}")
    print(f"Summary: {summary_path}")


def main() -> None:
    args = parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
