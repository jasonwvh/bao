#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.common.streaming import derive_stream_id
from orchestrator.a2a import A2AClient, A2AClientError, calibrate_handle_costs, load_registry
from orchestrator.benchmarking import (
    MetricsAccumulator,
    attach_reference_deltas,
    build_comparison_block,
    build_metric_reference,
    evaluation_prediction,
    threshold_decision,
)
from orchestrator.config import OrchestratorConfig, load_config
from orchestrator.data import DataAdapter, load_replay_dataset
from orchestrator.decisioning import DecisionCosts, select_decision
from orchestrator.runtime import BAORuntime
from orchestrator.state import SQLiteState


AGENT_CHOICES = ["ocsvm", "lstm_autoencoder", "wgan_gp"]
MODE_CHOICES = ["bao", "agent", "all"]
BASELINE_FAMILIES = ["thresholded_single_agent", "cost_aware_single_agent"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lean BAO benchmark entrypoint")
    p.add_argument("--mode", choices=MODE_CHOICES, default="bao")
    p.add_argument("--agent", choices=AGENT_CHOICES, default=None)
    p.add_argument("--baseline-family", choices=BASELINE_FAMILIES, default="cost_aware_single_agent")
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


def _load_calibrated_handles(cfg: OrchestratorConfig):
    handles = load_registry(cfg.orchestration.agent_registry_path)
    return calibrate_handle_costs(
        handles,
        human_review_cost=cfg.decision.c_h,
        false_positive_cost=cfg.decision.c_fp,
        max_fraction_of_action_cost=cfg.query.detector_cost_fraction,
    )


def _select_baseline_decision(*, family: str, probability: float, costs: DecisionCosts) -> str:
    if family == "thresholded_single_agent":
        return threshold_decision(probability)
    decision, _ = select_decision(probability, costs)
    return decision


def _run_agent_baseline(
    *,
    agent_id: str,
    rows: List[Dict[str, Any]],
    cfg: OrchestratorConfig,
    family: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    handles = _load_calibrated_handles(cfg)
    handle = handles.get(agent_id)
    if handle is None:
        raise RuntimeError(f"Agent not enabled in registry: {agent_id}")

    data_adapter = DataAdapter(schema_path=cfg.preprocessing.schema_path)
    a2a = A2AClient(retries=cfg.a2a.retries)
    costs = DecisionCosts(c_fn=cfg.decision.c_fn, c_fp=cfg.decision.c_fp, c_h=cfg.decision.c_h)
    acc = MetricsAccumulator(costs=costs)
    warnings: Dict[str, Dict[str, Any]] = {}
    session_id = f"baseline-{family}-{agent_id}-{uuid.uuid4().hex[:8]}"

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

        decision = _select_baseline_decision(family=family, probability=p, costs=costs)
        acc.add(
            true_label=int(row["true_label"]),
            probability=p,
            decision=decision,
            query_cost=query_cost,
            metadata=dict(row.get("metadata") or {}),
        )

        replay_row = {
            "family": family,
            "approach": agent_id,
            "flow_id": row["flow_id"],
            "true_label": int(row["true_label"]),
            "decision": decision,
            "probability": float(p),
            "prediction": evaluation_prediction(
                probability=float(p),
                decision=decision,
                true_label=int(row["true_label"]),
            ),
            "query_cost": query_cost,
            "metadata": dict(row.get("metadata") or {}),
        }
        if error_msg:
            replay_row["agent_error"] = error_msg
        replay_rows.append(replay_row)

    return (
        replay_rows,
        acc.compute(approach=agent_id, family=family),
        sorted(warnings.values(), key=lambda x: str(x["code"])),
    )


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
        metadata = dict(row.get("metadata") or {})
        acc.add(
            true_label=int(row["true_label"]),
            probability=p,
            decision=decision,
            query_cost=float(res["cumulative_cost"]),
            metadata=metadata,
        )

        replay_rows.append(
            {
                "family": "bao",
                "approach": "bao",
                "flow_id": row["flow_id"],
                "true_label": int(row["true_label"]),
                "decision": decision,
                "probability": p,
                "prediction": evaluation_prediction(
                    probability=p,
                    decision=decision,
                    true_label=int(row["true_label"]),
                ),
                "query_cost": float(res["cumulative_cost"]),
                "epistemic_uncertainty": float(res["epistemic_uncertainty"]),
                "combined_uncertainty": float(res["combined_uncertainty"]),
                "expected_net_gain": float(res.get("expected_net_gain", float("nan"))),
                "agents_queried": list(res["agents_queried"]),
                "metadata": metadata,
            }
        )

    summary = runtime.get_summary()
    metrics = acc.compute(approach="bao", family="bao", summary=summary)
    return replay_rows, metrics, summary


def _decorate_results_with_reference(
    *,
    bao_metrics: Dict[str, Any],
    thresholded_results: Dict[str, Dict[str, Any]],
    cost_aware_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    reference = build_metric_reference(thresholded_results)
    if reference:
        attach_reference_deltas(bao_metrics, reference)
        for result_set in (thresholded_results, cost_aware_results):
            for metrics in result_set.values():
                attach_reference_deltas(metrics, reference)
    return build_comparison_block(bao_metrics=bao_metrics, thresholded_results=thresholded_results)


def main() -> None:
    args = parse_args()
    if args.mode == "agent" and not args.agent:
        raise RuntimeError("--agent is required when --mode agent")

    output_root = Path(args.output_dir).resolve()
    run_id, run_dir = _allocate_run_dir(output_root, args.run_id)

    sqlite_path = run_dir / "state.sqlite"
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

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
        replay_rows, metrics, warnings = _run_agent_baseline(
            agent_id=str(args.agent),
            rows=rows,
            cfg=cfg,
            family=str(args.baseline_family),
        )
        replay_results.extend(replay_rows)
        merge_warnings(warnings)
        benchmark_payload = metrics

    else:
        thresholded_results: Dict[str, Dict[str, Any]] = {}
        cost_aware_results: Dict[str, Dict[str, Any]] = {}

        for aid in AGENT_CHOICES:
            for family, target in (
                ("thresholded_single_agent", thresholded_results),
                ("cost_aware_single_agent", cost_aware_results),
            ):
                replay_rows, metrics, warnings = _run_agent_baseline(agent_id=aid, rows=rows, cfg=cfg, family=family)
                replay_results.extend(replay_rows)
                target[aid] = metrics
                merge_warnings(warnings)

        runtime = BAORuntime(cfg, state_sqlite_path=sqlite_path)
        bao_replay, bao_metrics, bao_summary = _run_bao(rows=rows, runtime=runtime)
        replay_results.extend(bao_replay)
        merge_warnings(list(bao_summary.get("warnings") or []))
        comparison = _decorate_results_with_reference(
            bao_metrics=bao_metrics,
            thresholded_results=thresholded_results,
            cost_aware_results=cost_aware_results,
        )

        benchmark_payload = {
            "mode": "all",
            "flows_processed": len(rows),
            "results": {
                "thresholded_single_agent": thresholded_results,
                "cost_aware_single_agent": cost_aware_results,
                "bao": bao_metrics,
            },
            "bao_summary": bao_summary,
            "comparison": comparison,
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
        "baseline_family": args.baseline_family,
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
