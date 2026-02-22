#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.metrics import compute_metrics
from orchestrator.data.replay import load_replay_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ensemble_benchmark")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Ensemble (IF + AE) with simple average fusion")
    p.add_argument("--dataset", default="data/UNSW_NB15_testing-set.csv", help="CSV or parquet dataset")
    p.add_argument("--max-flows", type=int, default=0, help="Max flows to process (0=all)")
    p.add_argument("--output-dir", default="artifacts/replay", help="Output directory")
    p.add_argument("--if-model-path", default=None, help="Path to Isolation Forest model")
    p.add_argument("--ae-model-path", default=None, help="Path to Autoencoder model")
    p.add_argument("--if-cost", type=float, default=1.0, help="Isolation Forest cost per inference")
    p.add_argument("--ae-cost", type=float, default=2.5, help="Autoencoder cost per inference")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if_model_path = args.if_model_path
    if if_model_path is None:
        if_model_path = os.path.join(REPO_ROOT, "agents", "isolation_forest", "models", "isolation_forest.pkl")

    ae_model_path = args.ae_model_path
    if ae_model_path is None:
        ae_model_path = os.path.join(REPO_ROOT, "agents", "autoencoder", "models", "autoencoder.pt")

    from agents.autoencoder.service import Autoencoder
    from agents.isolation_forest.service import IsolationForestAgent

    if_agent = IsolationForestAgent(model_path=if_model_path, cost=args.if_cost)
    ae_agent = Autoencoder(model_path=ae_model_path, cost=args.ae_cost)

    rows = load_replay_dataset(args.dataset, max_rows=(args.max_flows or None))
    if not rows:
        raise RuntimeError("No rows loaded from dataset")

    predictions: list[int] = []
    labels: list[int] = []
    probabilities: list[float] = []
    costs: list[float] = []

    ensemble_cost = args.if_cost + args.ae_cost

    for idx, row in enumerate(rows):
        true_label = row.get("true_label")
        if true_label is None:
            continue

        if_output = if_agent.predict_with_uncertainty(row["flow_features"])
        ae_output = ae_agent.predict_with_uncertainty(row["flow_features"])

        p_if = float(if_output["proba"][1])
        p_ae = float(ae_output["proba"][1])

        p_ensemble = (p_if + p_ae) / 2.0
        pred = 1 if p_ensemble >= 0.5 else 0

        labels.append(int(true_label))
        probabilities.append(p_ensemble)
        predictions.append(pred)
        costs.append(ensemble_cost)

        logger.info(
            "[%d] flow_id=%s true=%d pred=%d p_if=%.4f p_ae=%.4f p_ens=%.4f cost=%.2f",
            idx + 1, row["flow_id"], int(true_label), pred, p_if, p_ae, p_ensemble, ensemble_cost
        )

    if not labels:
        raise RuntimeError("No labeled rows found in dataset")

    metrics = compute_metrics(
        predictions=predictions,
        labels=labels,
        probabilities=probabilities,
        costs=costs,
        approach="ensemble",
    )

    benchmark_path = output_dir / "benchmark_ensemble.json"
    benchmark_path.write_text(json.dumps(metrics, indent=2))

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
