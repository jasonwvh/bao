#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.metrics import compute_metrics
from orchestrator.data.replay import load_replay_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ae_benchmark")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark LSTM Autoencoder agent")
    p.add_argument("--dataset", default="data/UNSW_NB15_testing-set.csv", help="CSV or parquet dataset")
    p.add_argument("--max-flows", type=int, default=0, help="Max flows to process (0=all)")
    p.add_argument("--output-dir", default="artifacts/replay", help="Output directory")
    p.add_argument("--model-path", default=None, help="Path to model file")
    p.add_argument("--cost", type=float, default=1.0, help="Cost per inference")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_path
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "models", "lstm_autoencoder.pt")

    from agents.lstm_autoencoder.service import LSTMAutoencoderAgent

    agent = LSTMAutoencoderAgent(model_path=model_path, cost=args.cost)

    rows = load_replay_dataset(args.dataset, max_rows=(args.max_flows or None))
    if not rows:
        raise RuntimeError("No rows loaded from dataset")

    predictions: list[int] = []
    labels: list[int] = []
    probabilities: list[float] = []
    costs: list[float] = []

    for idx, row in enumerate(rows):
        true_label = row.get("true_label")
        if true_label is None:
            continue

        output = agent.predict_with_uncertainty(row["flow_features"], flow_id=row.get("flow_id"))
        p_mal = float(output["proba"][1])
        pred = 1 if p_mal >= 0.5 else 0

        labels.append(int(true_label))
        probabilities.append(p_mal)
        predictions.append(pred)
        costs.append(agent.cost)

        logger.info(
            "[%d] flow_id=%s true=%d pred=%d p_mal=%.4f cost=%.2f",
            idx + 1, row["flow_id"], int(true_label), pred, p_mal, agent.cost
        )

    if not labels:
        raise RuntimeError("No labeled rows found in dataset")

    metrics = compute_metrics(
        predictions=predictions,
        labels=labels,
        probabilities=probabilities,
        costs=costs,
        approach="lstm_autoencoder",
    )

    benchmark_path = output_dir / "benchmark_lstm_autoencoder.json"
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
