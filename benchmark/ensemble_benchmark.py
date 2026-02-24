#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.runner import (
    BenchmarkAccumulator,
    build_benchmark_manifest,
    dataset_composition,
    write_json,
)
from orchestrator.config import PREDICTION_SOURCES
from orchestrator.control.registry import load_registry, to_runtime_handles
from orchestrator.data.replay import load_replay_dataset
from orchestrator.data_plane.a2a_client import A2AClient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Ensemble (OCSVM + LSTM-AE + WGAN-GP)")
    p.add_argument("--dataset", default="data/UNSW_NB15_testing-set.csv", help="CSV or parquet dataset")
    p.add_argument("--max-flows", type=int, default=0, help="Max flows to process (0=all)")
    p.add_argument("--output-dir", default="artifacts/replay", help="Output directory")
    p.add_argument("--registry", default="config/agents.yaml", help="Path to A2A agent registry YAML")
    p.add_argument("--ocsvm-cost", type=float, default=None, help="Override OCSVM cost per inference")
    p.add_argument("--lstm-ae-cost", type=float, default=None, help="Override LSTM-AE cost per inference")
    p.add_argument("--wgan-cost", type=float, default=None, help="Override WGAN-GP cost per inference")
    p.add_argument("--prediction-source", choices=sorted(PREDICTION_SOURCES), default="probability")
    p.add_argument("--write-manifest", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _build_payload(row: dict) -> dict:
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
        },
    }


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = load_registry(args.registry)
    handles = to_runtime_handles(registry)

    required = ("ocsvm", "lstm_autoencoder", "wgan_gp")
    for aid in required:
        if aid not in handles:
            raise RuntimeError(f"{aid} not found or disabled in registry")

    a2a = A2AClient(retries=0)
    for aid in required:
        health = a2a.health(handles[aid])
        if str(health.get("status", "")).lower() != "ok":
            raise RuntimeError(f"{aid} service unhealthy: {health}")

    ocsvm_cost = float(args.ocsvm_cost) if args.ocsvm_cost is not None else float(handles["ocsvm"].cost)
    lstm_cost = float(args.lstm_ae_cost) if args.lstm_ae_cost is not None else float(handles["lstm_autoencoder"].cost)
    wgan_cost = float(args.wgan_cost) if args.wgan_cost is not None else float(handles["wgan_gp"].cost)

    rows = load_replay_dataset(args.dataset, max_rows=(args.max_flows or None))
    if not rows:
        raise RuntimeError("No rows loaded from dataset")

    acc = BenchmarkAccumulator(prediction_source=args.prediction_source)
    ensemble_cost = float(ocsvm_cost + lstm_cost + wgan_cost)

    for row in rows:
        true_label = row.get("true_label")
        if true_label is None:
            continue

        payload = _build_payload(row)
        if_output = a2a.infer(handles["ocsvm"], payload)
        ae_output = a2a.infer(handles["lstm_autoencoder"], payload)
        llm_output = a2a.infer(handles["wgan_gp"], payload)

        p_ensemble = (
            float(if_output["proba"][1])
            + float(ae_output["proba"][1])
            + float(llm_output["proba"][1])
        ) / 3.0

        acc.add_sample(
            true_label=int(true_label),
            probability=float(p_ensemble),
            cost=ensemble_cost,
        )

    metrics = acc.compute(approach="ensemble")

    benchmark_path = output_dir / "benchmark_ensemble.json"
    write_json(benchmark_path, metrics)

    if args.write_manifest:
        manifest = build_benchmark_manifest(
            repo_root=REPO_ROOT,
            dataset_path=Path(args.dataset).resolve(),
            config_path=None,
            approach="ensemble",
            agents_used=["ocsvm", "lstm_autoencoder", "wgan_gp"],
            extra={
                "prediction_source": args.prediction_source,
                "dataset_composition": dataset_composition(rows),
                "registry_path": str(Path(args.registry).resolve()),
            },
        )
        write_json(output_dir / "benchmark_ensemble_manifest.json", manifest)

    print(f"Processed {metrics['flows_processed']} flows")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Total cost: {metrics['total_cost']:.4f}")
    print(f"Avg cost/flow: {metrics['avg_cost_per_flow']:.4f}")
    print(f"Output: {benchmark_path}")


if __name__ == "__main__":
    main()
