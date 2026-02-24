#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run benchmark matrix for agents and BAO")
    p.add_argument("--dataset", default="data/UNSW_NB15_testing-set.csv", help="Benchmark dataset path")
    p.add_argument("--config", default="config/orchestrator_config.yaml", help="Orchestrator config path")
    p.add_argument("--output-root", default="artifacts/replay/matrix", help="Output root directory")
    p.add_argument("--max-flows", type=int, default=0, help="Limit number of flows (0=all)")
    p.add_argument("--prediction-source", choices=["decision", "probability"], default="probability")
    p.add_argument("--write-manifest", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _run(cmd: List[str]) -> None:
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    max_flows_args: List[str] = []
    if args.max_flows and int(args.max_flows) > 0:
        max_flows_args = ["--max-flows", str(int(args.max_flows))]

    manifest_flag = "--write-manifest" if bool(args.write_manifest) else "--no-write-manifest"

    _run(
        [
            sys.executable,
            "agents/ocsvm/benchmark.py",
            "--dataset",
            args.dataset,
            "--output-dir",
            str(out_root / "ocsvm"),
            "--prediction-source",
            args.prediction_source,
            manifest_flag,
            *max_flows_args,
        ]
    )
    _run(
        [
            sys.executable,
            "agents/lstm_autoencoder/benchmark.py",
            "--dataset",
            args.dataset,
            "--output-dir",
            str(out_root / "lstm_autoencoder"),
            "--prediction-source",
            args.prediction_source,
            manifest_flag,
            *max_flows_args,
        ]
    )
    _run(
        [
            sys.executable,
            "agents/wgan_gp/benchmark.py",
            "--dataset",
            args.dataset,
            "--output-dir",
            str(out_root / "wgan_gp"),
            "--prediction-source",
            args.prediction_source,
            manifest_flag,
            *max_flows_args,
        ]
    )
    _run(
        [
            sys.executable,
            "main.py",
            "--dataset",
            args.dataset,
            "--config",
            args.config,
            "--output-dir",
            str(out_root / "bao"),
            "--prediction-source",
            args.prediction_source,
            manifest_flag,
            *max_flows_args,
        ]
    )

    print(f"Benchmark matrix complete: {out_root}")


if __name__ == "__main__":
    main()

