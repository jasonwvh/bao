#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.metrics import compute_metrics
from orchestrator.config import load_orchestrator_config
from orchestrator.control.registry import load_registry, to_runtime_handles
from orchestrator.decision import DecisionCosts, realized_action_cost, select_expected_cost_action


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run benchmark matrix for agents and BAO")
    p.add_argument("--dataset", default="data/UNSW_NB15_testing-set.csv", help="Benchmark dataset path")
    p.add_argument("--config", default="config/orchestrator_config.yaml", help="Orchestrator config path")
    p.add_argument("--output-root", default="artifacts/replay/matrix", help="Output root directory")
    p.add_argument("--max-flows", type=int, default=0, help="Limit number of flows (0=all)")
    p.add_argument("--prediction-source", choices=["decision", "probability"], default="decision")
    p.add_argument("--utility-evaluation", choices=["cost_action_parity", "legacy_mixed"], default=None)
    p.add_argument("--write-manifest", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--build-profile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--calibrate-costs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compare-engines", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--router-profile", default=None, help="Router profile output path")
    p.add_argument("--calibration-json", default=None, help="Cost calibration output JSON")
    p.add_argument("--calibrated-config", default=None, help="Calibrated orchestrator config output path")
    return p.parse_args()


def _run(cmd: List[str]) -> None:
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _agent_metrics_with_costs(replay_path: Path, *, per_call_cost: float, costs: DecisionCosts, approach: str) -> dict:
    rows = _load_json(replay_path)
    predictions = [int(r["prediction"]) for r in rows]
    labels = [int(r["true_label"]) for r in rows]
    probabilities = [float(r["probability"]) for r in rows]
    query_costs = [float(per_call_cost)] * len(rows)
    action_decisions = [select_expected_cost_action(float(r["probability"]), costs)[0] for r in rows]
    action_costs = [
        realized_action_cost(
            decision=action_decisions[i],
            prediction=int(r["prediction"]),
            true_label=int(r["true_label"]),
            costs=costs,
        )
        for i, r in enumerate(rows)
    ]
    return compute_metrics(
        predictions=predictions,
        labels=labels,
        probabilities=probabilities,
        query_costs=query_costs,
        action_costs=action_costs,
        action_decisions=action_decisions,
        utility_evaluation="cost_action_parity",
        approach=approach,
    )


def main() -> None:
    args = parse_args()
    orch_cfg = load_orchestrator_config(args.config)
    registry = load_registry(orch_cfg.orchestration.agent_registry_path)
    handles = to_runtime_handles(registry)
    agent_query_cost = {aid: float(handle.cost) for aid, handle in handles.items()}
    base_decision_costs = DecisionCosts(
        c_fn=float(orch_cfg.decision.c_fn),
        c_fp=float(orch_cfg.decision.c_fp),
        c_h=float(orch_cfg.decision.c_h),
    )
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    max_flows_args: List[str] = []
    if args.max_flows and int(args.max_flows) > 0:
        max_flows_args = ["--max-flows", str(int(args.max_flows))]
    utility_eval = str(args.utility_evaluation or orch_cfg.benchmark.utility_evaluation)
    utility_eval_args = ["--utility-evaluation", utility_eval]

    manifest_flag = "--write-manifest" if bool(args.write_manifest) else "--no-write-manifest"

    _run(
        [
            sys.executable,
            "agents/ocsvm/benchmark.py",
            "--dataset",
            args.dataset,
            "--config",
            args.config,
            "--output-dir",
            str(out_root / "ocsvm"),
            "--prediction-source",
            args.prediction_source,
            *utility_eval_args,
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
            "--config",
            args.config,
            "--output-dir",
            str(out_root / "lstm_autoencoder"),
            "--prediction-source",
            args.prediction_source,
            *utility_eval_args,
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
            "--config",
            args.config,
            "--output-dir",
            str(out_root / "wgan_gp"),
            "--prediction-source",
            args.prediction_source,
            *utility_eval_args,
            manifest_flag,
            *max_flows_args,
        ]
    )

    profile_path = Path(args.router_profile) if args.router_profile else (out_root / "router_profile.json")
    if args.build_profile:
        _run(
            [
                sys.executable,
                "benchmark/build_router_profile.py",
                "--input-root",
                str(out_root),
                "--output-path",
                str(profile_path),
                "--config",
                args.config,
            ]
        )

    calibrated_config = Path(args.config)
    calibration_json: Path | None = None
    if args.calibrate_costs:
        calibration_json = Path(args.calibration_json) if args.calibration_json else (out_root / "cost_calibration.json")
        calibrated_config = Path(args.calibrated_config) if args.calibrated_config else (out_root / "orchestrator_calibrated.yaml")
        _run(
            [
                sys.executable,
                "benchmark/calibrate_decision_costs.py",
                "--input-root",
                str(out_root),
                "--profile-path",
                str(profile_path),
                "--base-config",
                str(args.config),
                "--output-json",
                str(calibration_json),
                "--output-config",
                str(calibrated_config),
            ]
        )

    _run(
        [
            sys.executable,
            "main.py",
            "--dataset",
            args.dataset,
            "--config",
            str(calibrated_config),
            "--output-dir",
            str(out_root / "bao"),
            "--prediction-source",
            args.prediction_source,
            *utility_eval_args,
            manifest_flag,
            *max_flows_args,
        ]
    )

    if args.compare_engines:
        _run(
            [
                sys.executable,
                "benchmark/compare_engines.py",
                "--dataset",
                args.dataset,
                "--config",
                str(calibrated_config),
                "--max-flows",
                str(int(args.max_flows) if int(args.max_flows or 0) > 0 else 2000),
                "--output-json",
                str(out_root / "engine_compare.json"),
            ]
        )

    summary = {
        "ocsvm": _load_json(out_root / "ocsvm" / "benchmark_ocsvm.json"),
        "lstm_autoencoder": _load_json(out_root / "lstm_autoencoder" / "benchmark_lstm_autoencoder.json"),
        "wgan_gp": _load_json(out_root / "wgan_gp" / "benchmark_wgan_gp.json"),
        "bao": _load_json(out_root / "bao" / "benchmark_bao.json"),
    }
    if calibration_json is not None and calibration_json.exists():
        cal = _load_json(calibration_json)
        costs = DecisionCosts(
            c_fn=float(cal.get("c_fn", base_decision_costs.c_fn)),
            c_fp=float(cal.get("c_fp", base_decision_costs.c_fp)),
            c_h=float(cal.get("c_h", base_decision_costs.c_h)),
        )
        summary["costs_used_for_recalibration"] = {"c_fn": costs.c_fn, "c_fp": costs.c_fp, "c_h": costs.c_h}
        summary["ocsvm_recalibrated_costs"] = _agent_metrics_with_costs(
            out_root / "ocsvm" / "replay_results_ocsvm.json",
            per_call_cost=float(agent_query_cost.get("ocsvm", 0.0)),
            costs=costs,
            approach="ocsvm_recalibrated",
        )
        summary["lstm_autoencoder_recalibrated_costs"] = _agent_metrics_with_costs(
            out_root / "lstm_autoencoder" / "replay_results_lstm_autoencoder.json",
            per_call_cost=float(agent_query_cost.get("lstm_autoencoder", 0.0)),
            costs=costs,
            approach="lstm_autoencoder_recalibrated",
        )
        summary["wgan_gp_recalibrated_costs"] = _agent_metrics_with_costs(
            out_root / "wgan_gp" / "replay_results_wgan_gp.json",
            per_call_cost=float(agent_query_cost.get("wgan_gp", 0.0)),
            costs=costs,
            approach="wgan_gp_recalibrated",
        )
    (out_root / "utility_report.json").write_text(json.dumps(summary, indent=2))

    print(f"Benchmark matrix complete: {out_root}")


if __name__ == "__main__":
    main()
