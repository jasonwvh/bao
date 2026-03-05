#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.common.streaming import derive_stream_id
from orchestrator.a2a import A2AClient, A2AClientError, load_registry
from orchestrator.config import OrchestratorConfig, load_config
from orchestrator.data import DataAdapter, load_replay_dataset
from orchestrator.decisioning import DecisionCosts, probability_to_prediction, realized_action_cost, select_decision
from orchestrator.runtime import BAORuntime
from orchestrator.state import SQLiteState


AGENT_CHOICES = ["ocsvm", "lstm_autoencoder", "wgan_gp"]
MODE_CHOICES = ["bao", "agent", "all"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lean BAO benchmark entrypoint")
    p.add_argument("--mode", choices=MODE_CHOICES, default="bao")
    p.add_argument("--agent", choices=AGENT_CHOICES, default=None)
    p.add_argument("--dataset", default="data/UNSW_NB15_testing-set.csv")
    p.add_argument("--config", default="config/orchestrator_config.utility.yaml")
    p.add_argument("--output-dir", default="artifacts/runs")
    p.add_argument("--max-flows", type=int, default=0)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def _sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _compute_auc(labels: List[int], probs: List[float]) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        score = float(roc_auc_score(labels, probs))
        if math.isnan(score):
            return 0.5
        return score
    except Exception:
        return 0.5


def _evaluation_prediction(*, probability: float, decision: str, true_label: int) -> int:
    d = str(decision).strip().lower()
    if d == "defer":
        # Evaluation policy: defer is counted as correct because final judgment is human.
        return int(true_label)
    return probability_to_prediction(float(probability))


def _allocate_run_dir(output_root: Path, requested_run_id: Optional[str]) -> tuple[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    base = str(requested_run_id or f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}")
    run_id = base
    idx = 1
    while (output_root / run_id).exists():
        run_id = f"{base}_{idx:02d}"
        idx += 1
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


@dataclass
class MetricsAccumulator:
    costs: DecisionCosts
    labels: List[int] = field(default_factory=list)
    predictions: List[int] = field(default_factory=list)
    probabilities: List[float] = field(default_factory=list)
    query_costs: List[float] = field(default_factory=list)
    action_costs: List[float] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)

    def add(self, *, true_label: int, probability: float, decision: str, query_cost: float) -> None:
        p = float(probability)
        y = int(true_label)
        d = str(decision).strip().lower()
        pred = _evaluation_prediction(probability=p, decision=d, true_label=y)
        a_cost = realized_action_cost(decision=d, true_label=y, costs=self.costs)

        self.labels.append(y)
        self.predictions.append(pred)
        self.probabilities.append(p)
        self.query_costs.append(float(query_cost))
        self.action_costs.append(float(a_cost))
        self.decisions.append(d)

    def compute(self, approach: str, summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        tp = sum(1 for p, y in zip(self.predictions, self.labels) if p == 1 and y == 1)
        tn = sum(1 for p, y in zip(self.predictions, self.labels) if p == 0 and y == 0)
        fp = sum(1 for p, y in zip(self.predictions, self.labels) if p == 1 and y == 0)
        fn = sum(1 for p, y in zip(self.predictions, self.labels) if p == 0 and y == 1)

        total = max(1, len(self.labels))
        accuracy = (tp + tn) / float(total)
        precision = tp / float(max(1, tp + fp))
        recall = tp / float(max(1, tp + fn))
        f1 = (2.0 * precision * recall) / float(max(1e-9, precision + recall))
        auc = _compute_auc(self.labels, self.probabilities)

        query_total = float(sum(self.query_costs))
        action_total = float(sum(self.action_costs))
        utility_total = query_total + action_total

        defer_count = sum(1 for d in self.decisions if d == "defer")

        out = {
            "approach": approach,
            "flows_processed": int(len(self.labels)),
            "accuracy": round(float(accuracy), 6),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "auc": round(float(auc), 6),
            "avg_cost_per_flow": round(query_total / float(total), 4),
            "query_cost_total": round(query_total, 4),
            "action_cost_total": round(action_total, 4),
            "utility_cost_total": round(utility_total, 4),
            "utility_cost_per_flow": round(utility_total / float(total), 4),
            "defer_count": int(defer_count),
            "defer_rate": round(float(defer_count) / float(total), 6),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }
        if isinstance(summary, dict):
            for key in ("agent_utilization", "routing_selection_counts", "routing"):
                if key in summary:
                    out[key] = summary[key]
        return out


def _build_payload(
    row: Dict[str, Any],
    p_mal: float = 0.5,
    uncertainty: float = 0.69314718056,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    flow_features = dict(row["flow_features"])
    stream_id = derive_stream_id(flow_features=flow_features, flow_id=row["flow_id"])
    return {
        "request_id": str(uuid.uuid4()),
        "flow_id": row["flow_id"],
        "timestamp": row.get("timestamp") or time.time(),
        "flow_features": flow_features,
        "context": {
            "belief": {"p_mal": float(p_mal), "uncertainty": float(uncertainty)},
            "requested_capabilities": [],
            "seed": 7,
            "stream_id": stream_id,
            "session_id": str(session_id) if session_id else None,
            "elicit_likelihood": True,
        },
    }


def _run_agent_baseline(
    *,
    agent_id: str,
    rows: List[Dict[str, Any]],
    cfg: OrchestratorConfig,
) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    handles = load_registry(cfg.orchestration.agent_registry_path)
    handle = handles.get(agent_id)
    if handle is None:
        raise RuntimeError(f"Agent not enabled in registry: {agent_id}")

    data_adapter = DataAdapter(schema_path=cfg.preprocessing.schema_path)
    a2a = A2AClient(retries=cfg.a2a.retries)
    costs = DecisionCosts(c_fn=cfg.decision.c_fn, c_fp=cfg.decision.c_fp, c_h=cfg.decision.c_h)
    acc = MetricsAccumulator(costs=costs)
    warnings: Dict[str, Dict[str, Any]] = {}
    session_id = f"baseline-{agent_id}-{uuid.uuid4().hex[:8]}"

    replay_rows: List[Dict[str, Any]] = []
    for row in rows:
        features = data_adapter.transform(dict(row["flow_features"]))
        payload = _build_payload(
            {
                "flow_id": row["flow_id"],
                "timestamp": row.get("timestamp"),
                "flow_features": features,
            },
            session_id=session_id,
        )
        query_cost = float(handle.cost)
        error_msg: Optional[str] = None
        try:
            out = a2a.infer(handle, payload)
            p = float((out.get("proba") or [0.5, 0.5])[1])
        except A2AClientError as exc:
            p = 0.5
            query_cost = 0.0
            error_msg = str(exc)
            key = f"{agent_id}_transport_failure"
            current = warnings.get(key)
            if current is None:
                warnings[key] = {
                    "code": key,
                    "message": f"{agent_id} infer transport failure",
                    "count": 1,
                }
            else:
                current["count"] = int(current["count"]) + 1
        decision, _ = select_decision(p, costs)

        acc.add(
            true_label=int(row["true_label"]),
            probability=p,
            decision=decision,
            query_cost=query_cost,
        )

        replay_row = {
            "approach": agent_id,
            "flow_id": row["flow_id"],
            "true_label": int(row["true_label"]),
            "decision": decision,
            "probability": float(p),
            "prediction": _evaluation_prediction(
                probability=float(p),
                decision=decision,
                true_label=int(row["true_label"]),
            ),
            "query_cost": query_cost,
        }
        if error_msg:
            replay_row["agent_error"] = error_msg
        replay_rows.append(replay_row)

    return replay_rows, acc.compute(approach=agent_id), sorted(warnings.values(), key=lambda x: str(x["code"]))


def _run_bao(*, rows: List[Dict[str, Any]], runtime: BAORuntime) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    costs = DecisionCosts(c_fn=runtime.config.decision.c_fn, c_fp=runtime.config.decision.c_fp, c_h=runtime.config.decision.c_h)
    acc = MetricsAccumulator(costs=costs)

    replay_rows: List[Dict[str, Any]] = []
    for row in rows:
        res = runtime.process_flow(
            flow_features=row["flow_features"],
            flow_id=row["flow_id"],
            timestamp=row.get("timestamp") or time.time(),
            true_label=int(row["true_label"]),
        )

        p = float(res["compromise_prob"])
        decision = str(res["decision"])
        acc.add(
            true_label=int(row["true_label"]),
            probability=p,
            decision=decision,
            query_cost=float(res["cumulative_cost"]),
        )

        replay_rows.append(
            {
                "approach": "bao",
                "flow_id": row["flow_id"],
                "true_label": int(row["true_label"]),
                "decision": decision,
                "probability": p,
                "prediction": _evaluation_prediction(
                    probability=p,
                    decision=decision,
                    true_label=int(row["true_label"]),
                ),
                "query_cost": float(res["cumulative_cost"]),
                "epistemic_uncertainty": float(res["epistemic_uncertainty"]),
                "combined_uncertainty": float(res["combined_uncertainty"]),
                "expected_net_gain": float(res.get("expected_net_gain", float("nan"))),
                "agents_queried": list(res["agents_queried"]),
            }
        )

    summary = runtime.get_summary()
    metrics = acc.compute(approach="bao", summary=summary)
    return replay_rows, metrics, summary


def _build_comparison_block(results: Dict[str, Any]) -> Dict[str, Any]:
    bao = dict(results.get("bao") or {})
    ocsvm = dict(results.get("ocsvm") or {})
    lstm = dict(results.get("lstm_autoencoder") or {})
    if not bao or not ocsvm or not lstm:
        return {}

    bao_acc = float(bao.get("accuracy", 0.0))
    ocsvm_acc = float(ocsvm.get("accuracy", 0.0))
    lstm_util = float(lstm.get("utility_cost_total", 0.0))
    bao_util = float(bao.get("utility_cost_total", 0.0))

    return {
        "targets": {
            "bao_accuracy_gt_ocsvm": bool(bao_acc > ocsvm_acc),
            "bao_utility_lt_lstm": bool(bao_util < lstm_util),
        },
        "deltas": {
            "accuracy_bao_minus_ocsvm": round(bao_acc - ocsvm_acc, 6),
            "utility_bao_minus_lstm": round(bao_util - lstm_util, 4),
        },
    }


def main() -> None:
    args = parse_args()
    if args.mode == "agent" and not args.agent:
        raise RuntimeError("--agent is required when --mode agent")

    output_root = Path(args.output_dir).resolve()
    run_id, run_dir = _allocate_run_dir(output_root, args.run_id)

    sqlite_path = run_dir / "state.sqlite"
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    # Ensure sqlite exists for every mode (bao/agent/all).
    SQLiteState(sqlite_path)

    rows = load_replay_dataset(args.dataset, max_rows=(args.max_flows if args.max_flows > 0 else None))

    replay_results: List[Dict[str, Any]] = []
    benchmark_payload: Dict[str, Any]
    run_warnings: Dict[str, Dict[str, Any]] = {}

    def merge_warnings(items: List[Dict[str, Any]]) -> None:
        for item in items:
            code = str(item.get("code", "")).strip() or "warning"
            count = int(item.get("count", 1))
            if code not in run_warnings:
                run_warnings[code] = {"code": code, "message": str(item.get("message", "")), "count": count}
            else:
                run_warnings[code]["count"] = int(run_warnings[code]["count"]) + count

    if args.mode == "bao":
        runtime = BAORuntime(cfg, state_sqlite_path=sqlite_path)
        replay_rows, metrics, summary = _run_bao(rows=rows, runtime=runtime)
        replay_results.extend(replay_rows)
        merge_warnings(list(summary.get("warnings") or []))
        benchmark_payload = metrics
        benchmark_payload["summary"] = summary

    elif args.mode == "agent":
        replay_rows, metrics, warnings = _run_agent_baseline(agent_id=str(args.agent), rows=rows, cfg=cfg)
        replay_results.extend(replay_rows)
        merge_warnings(warnings)
        benchmark_payload = metrics

    else:  # mode == all
        results: Dict[str, Any] = {}

        for aid in AGENT_CHOICES:
            replay_rows, metrics, warnings = _run_agent_baseline(agent_id=aid, rows=rows, cfg=cfg)
            replay_results.extend(replay_rows)
            results[aid] = metrics
            merge_warnings(warnings)

        runtime = BAORuntime(cfg, state_sqlite_path=sqlite_path)
        bao_replay, bao_metrics, bao_summary = _run_bao(rows=rows, runtime=runtime)
        replay_results.extend(bao_replay)
        results["bao"] = bao_metrics
        merge_warnings(list(bao_summary.get("warnings") or []))

        benchmark_payload = {
            "mode": "all",
            "flows_processed": len(rows),
            "results": results,
            "bao_summary": bao_summary,
            "comparison": _build_comparison_block(results),
        }

    benchmark_payload["warnings"] = sorted(run_warnings.values(), key=lambda x: str(x["code"]))

    replay_path = run_dir / "replay_results.json"
    benchmark_path = run_dir / "benchmark.json"
    manifest_path = run_dir / "run_manifest.json"

    replay_path.write_text(json.dumps(replay_results, indent=2))
    benchmark_path.write_text(json.dumps(benchmark_payload, indent=2))

    manifest = {
        "run_id": run_id,
        "created_at_unix": time.time(),
        "mode": args.mode,
        "agent": args.agent,
        "dataset_path": str(Path(args.dataset).resolve()),
        "dataset_sha256": _sha256(Path(args.dataset).resolve()),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "sqlite_path": str(sqlite_path),
        "sqlite_sha256": _sha256(sqlite_path),
        "artifacts": {
            "benchmark": str(benchmark_path),
            "replay_results": str(replay_path),
            "manifest": str(manifest_path),
            "sqlite": str(sqlite_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Run directory: {run_dir}")
    print(f"Benchmark: {benchmark_path}")
    print(f"Replay: {replay_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
