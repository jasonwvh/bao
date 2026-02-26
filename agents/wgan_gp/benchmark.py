#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.runner import (
    BenchmarkAccumulator,
    build_benchmark_manifest,
    dataset_composition,
    infer_prediction,
    label_to_decision,
    write_json,
)
from agents.common.streaming import derive_stream_id
from orchestrator.config import PREDICTION_SOURCES, UTILITY_EVALUATIONS, load_orchestrator_config
from orchestrator.control.registry import load_registry, to_runtime_handles
from orchestrator.data.replay import load_replay_dataset
from orchestrator.data_plane.a2a_client import A2AClient
from orchestrator.decision import DecisionCosts, select_expected_cost_action


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark WGAN-GP agent")
    p.add_argument("--dataset", default="data/UNSW_NB15_testing-set.csv", help="CSV or parquet dataset")
    p.add_argument("--max-flows", type=int, default=0, help="Max flows to process (0=all)")
    p.add_argument("--output-dir", default="artifacts/replay", help="Output directory")
    p.add_argument("--config", default="config/orchestrator_config.utility.yaml", help="Path to orchestrator config YAML")
    p.add_argument("--registry", default=None, help="Optional path to A2A agent registry YAML override")
    p.add_argument("--cost", type=float, default=None, help="Override cost per inference")
    p.add_argument("--prediction-source", choices=sorted(PREDICTION_SOURCES), default="probability")
    p.add_argument("--utility-evaluation", choices=sorted(UTILITY_EVALUATIONS), default=None)
    p.add_argument("--write-manifest", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _build_payload(row: dict) -> dict:
    stream_id = derive_stream_id(
        flow_features=dict(row.get("flow_features") or {}),
        flow_id=str(row.get("flow_id") or ""),
    )
    return {
        "request_id": str(uuid.uuid4()),
        "flow_id": row["flow_id"],
        "timestamp": row.get("timestamp") or time.time(),
        "flow_features": row["flow_features"],
        "context": {
            "belief": {"p_mal": 0.5, "uncertainty": 0.69314718056},
            "requested_capabilities": [],
            "elicit_likelihood": True,
            "seed": 7,
            "stream_id": stream_id,
        },
    }


def main() -> None:
    args = parse_args()
    cfg = load_orchestrator_config(args.config)
    registry_path = args.registry if args.registry else str(cfg.orchestration.agent_registry_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = load_registry(registry_path)
    handles = to_runtime_handles(registry)
    handle = handles.get("wgan_gp")
    if handle is None:
        raise RuntimeError("wgan_gp not found or disabled in registry")

    a2a = A2AClient(retries=0)
    health = a2a.health(handle)
    if str(health.get("status", "")).lower() != "ok":
        raise RuntimeError(f"wgan_gp service unhealthy: {health}")

    per_call_cost = float(args.cost) if args.cost is not None else float(handle.cost)

    rows = load_replay_dataset(args.dataset, max_rows=(args.max_flows or None))
    if not rows:
        raise RuntimeError("No rows loaded from dataset")

    utility_evaluation = (
        str(args.utility_evaluation).strip().lower()
        if args.utility_evaluation is not None
        else str(cfg.benchmark.utility_evaluation).strip().lower()
    )
    acc = BenchmarkAccumulator(
        prediction_source=args.prediction_source,
        utility_evaluation=utility_evaluation,
        decision_costs=DecisionCosts(c_fn=cfg.decision.c_fn, c_fp=cfg.decision.c_fp, c_h=cfg.decision.c_h),
    )
    replay_rows = []

    for row in rows:
        true_label = row.get("true_label")
        if true_label is None:
            continue

        output = a2a.infer(handle, _build_payload(row))
        p_mal = max(1e-6, min(1.0 - 1e-6, float((output.get("proba") or [0.5, 0.5])[1])))

        label_hint = (output.get("prediction") or {}).get("label")
        decision = label_to_decision(label_hint)
        if utility_evaluation == "cost_action_parity":
            action_decision, _ = select_expected_cost_action(p_mal, acc.decision_costs)
        else:
            action_decision = decision
        acc.add_sample(
            true_label=int(true_label),
            probability=p_mal,
            cost=per_call_cost,
            decision=decision,
            action_decision=action_decision,
            label_hint=label_hint,
        )
        pred = infer_prediction(
            prediction_source=args.prediction_source,
            probability=p_mal,
            decision=decision,
            label_hint=label_hint,
        )
        replay_rows.append(
            {
                "flow_id": row["flow_id"],
                "true_label": int(true_label),
                "prediction": int(pred),
                "probability": float(p_mal),
                "agent_id": "wgan_gp",
            }
        )

    metrics = acc.compute(approach="wgan_gp")
    metrics["utility_evaluation"] = utility_evaluation
    replay_path = output_dir / "replay_results.json"
    replay_agent_path = output_dir / "replay_results_wgan_gp.json"
    replay_payload = json.dumps(replay_rows, indent=2)
    replay_path.write_text(replay_payload)
    replay_agent_path.write_text(replay_payload)

    benchmark_path = output_dir / "benchmark_wgan_gp.json"
    write_json(benchmark_path, metrics)

    if args.write_manifest:
        manifest = build_benchmark_manifest(
            repo_root=REPO_ROOT,
            dataset_path=Path(args.dataset).resolve(),
            config_path=None,
            approach="wgan_gp",
            agents_used=["wgan_gp"],
            extra={
                "prediction_source": args.prediction_source,
                "utility_evaluation": utility_evaluation,
                "dataset_composition": dataset_composition(rows),
                "registry_path": str(Path(registry_path).resolve()),
            },
        )
        write_json(output_dir / "benchmark_wgan_gp_manifest.json", manifest)

    print(f"Processed {metrics['flows_processed']} flows")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Total cost: {metrics['total_cost']:.4f}")
    print(f"Avg cost/flow: {metrics['avg_cost_per_flow']:.4f}")
    print(f"Output: {benchmark_path}")
    print(f"Replay output: {replay_path}")
    print(f"Replay output (agent): {replay_agent_path}")


if __name__ == "__main__":
    main()
