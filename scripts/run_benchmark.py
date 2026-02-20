#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bao.benchmark.runner import run_benchmarks
from bao.benchmark.types import BenchmarkConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run BAO baseline benchmark suite")
    p.add_argument("--dataset", required=True, help="Path to CSV with label column")
    p.add_argument("--seeds", default="1,2,3", help="Comma-separated seeds")
    p.add_argument(
        "--policies",
        default="",
        help=(
            "Optional comma-separated policies from: "
            "bao,baseline_query_all,baseline_fixed_cascade,"
            "baseline_confidence_escalation,baseline_single_cheapest"
        ),
    )
    p.add_argument("--output-dir", default="artifacts/benchmark", help="Output directory")
    p.add_argument("--config", default="", help="Optional JSON config override path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    policies = [x.strip() for x in args.policies.split(",") if x.strip()] or None

    cfg = BenchmarkConfig()
    if args.config:
        loaded = json.loads(Path(args.config).read_text())
        cfg = BenchmarkConfig(**loaded)

    out = run_benchmarks(
        dataset_path=args.dataset,
        seeds=seeds,
        policy_names=policies,
        output_dir=args.output_dir,
        config=cfg,
    )

    print(f"Benchmark complete. Artifacts: {out['output_dir']}")
    print(f"Summary: {Path(out['output_dir']) / 'summary.md'}")


if __name__ == "__main__":
    main()
