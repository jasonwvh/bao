#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.metrics import compute_metrics
from orchestrator.data.replay import load_replay_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Ensemble (OCSVM + LSTM-AE + WGAN-GP)")
    p.add_argument("--dataset", default="data/UNSW_NB15_testing-set.csv", help="CSV or parquet dataset")
    p.add_argument("--max-flows", type=int, default=0, help="Max flows to process (0=all)")
    p.add_argument("--output-dir", default="artifacts/replay", help="Output directory")
    p.add_argument("--ocsvm-model-path", default=None, help="Path to OCSVM model")
    p.add_argument("--lstm-ae-model-path", default=None, help="Path to LSTM-AE model")
    p.add_argument("--wgan-model-path", default=None, help="Path to WGAN-GP model")
    p.add_argument("--ocsvm-cost", type=float, default=1.0, help="OCSVM cost per inference")
    p.add_argument("--lstm-ae-cost", type=float, default=2.5, help="LSTM-AE cost per inference")
    p.add_argument("--wgan-cost", type=float, default=8.0, help="WGAN-GP cost per inference")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if_model_path = args.ocsvm_model_path
    if if_model_path is None:
        if_model_path = os.path.join(REPO_ROOT, "agents", "ocsvm", "models", "ocsvm.pkl")

    ae_model_path = args.lstm_ae_model_path
    if ae_model_path is None:
        ae_model_path = os.path.join(REPO_ROOT, "agents", "lstm_autoencoder", "models", "lstm_autoencoder.pt")

    llm_model_path = args.wgan_model_path
    if llm_model_path is None:
        llm_model_path = os.path.join(REPO_ROOT, "agents", "wgan_gp", "models", "wgan_gp.pt")

    from agents.lstm_autoencoder.service import LSTMAutoencoderAgent
    from agents.ocsvm.service import OCSVMAgent
    from agents.wgan_gp.service import WGANGPAgent

    if_agent = OCSVMAgent(model_path=if_model_path, cost=args.ocsvm_cost)
    ae_agent = LSTMAutoencoderAgent(model_path=ae_model_path, cost=args.lstm_ae_cost)
    llm_agent = WGANGPAgent(model_path=llm_model_path, cost=args.wgan_cost)

    rows = load_replay_dataset(args.dataset, max_rows=(args.max_flows or None))
    if not rows:
        raise RuntimeError("No rows loaded from dataset")

    predictions: list[int] = []
    labels: list[int] = []
    probabilities: list[float] = []
    costs: list[float] = []

    ensemble_cost = args.ocsvm_cost + args.lstm_ae_cost + args.wgan_cost

    for row in rows:
        true_label = row.get("true_label")
        if true_label is None:
            continue

        if_output = if_agent.predict_with_uncertainty(row["flow_features"])
        ae_output = ae_agent.predict_with_uncertainty(row["flow_features"], flow_id=row.get("flow_id"))
        llm_output = llm_agent.predict_with_uncertainty(row["flow_features"])

        p_if = float(if_output["proba"][1])
        p_ae = float(ae_output["proba"][1])
        p_llm = float(llm_output["proba"][1])

        p_ensemble = (p_if + p_ae + p_llm) / 3.0
        pred = 1 if p_ensemble >= 0.5 else 0

        labels.append(int(true_label))
        probabilities.append(p_ensemble)
        predictions.append(pred)
        costs.append(ensemble_cost)

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
