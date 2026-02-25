#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator.config import load_orchestrator_config
from orchestrator.data.replay import load_replay_dataset
from orchestrator.integrated_system import IntegratedBAOSystem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare deterministic and LangGraph engine performance")
    p.add_argument("--dataset", default="data/UNSW_NB15_testing-set.csv")
    p.add_argument("--config", default="config/orchestrator_config.yaml")
    p.add_argument("--max-flows", type=int, default=2000)
    p.add_argument("--output-json", default="artifacts/replay/engine_compare.json")
    p.add_argument(
        "--guardrail-overhead",
        type=float,
        default=None,
        help="Allowed relative overhead (default from routing.langgraph_perf_guardrail_overhead)",
    )
    return p.parse_args()


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round(0.95 * (len(s) - 1)))))
    return float(s[idx])


def _build_runtime_config(
    *,
    base_raw: Dict[str, Any],
    base_cfg: Any,
    engine: str,
    temp_dir: Path,
) -> Path:
    cfg = dict(base_raw)

    orch = dict(cfg.get("orchestration", {}) or {})
    orch["engine"] = engine
    orch["agent_registry_path"] = str(base_cfg.orchestration.agent_registry_path)
    cfg["orchestration"] = orch

    state = dict(cfg.get("state", {}) or {})
    state["sqlite_path"] = str((temp_dir / f"state_{engine}.sqlite").resolve())
    cfg["state"] = state

    logging_cfg = dict(cfg.get("logging", {}) or {})
    logging_cfg["jsonl_path"] = str((temp_dir / f"flows_{engine}.jsonl").resolve())
    cfg["logging"] = logging_cfg

    routing = dict(cfg.get("routing", {}) or {})
    if base_cfg.routing.profile_path is not None:
        routing["profile_path"] = str(base_cfg.routing.profile_path)
    cfg["routing"] = routing

    pre = dict(cfg.get("preprocessing", {}) or {})
    if base_cfg.preprocessing.schema_path is not None:
        pre["schema_path"] = str(base_cfg.preprocessing.schema_path)
    cfg["preprocessing"] = pre

    benchmark = dict(cfg.get("benchmark", {}) or {})
    benchmark["reset_state"] = True
    cfg["benchmark"] = benchmark

    out = temp_dir / f"runtime_{engine}.yaml"
    out.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    return out


async def _run_engine(system: IntegratedBAOSystem, rows: List[Dict[str, Any]]) -> Dict[str, float]:
    latencies_ms: List[float] = []
    t0 = time.perf_counter()
    for row in rows:
        start = time.perf_counter()
        await system.process_flow(
            flow_features=row["flow_features"],
            flow_id=row["flow_id"],
            timestamp=row.get("timestamp") or 0.0,
            true_label=row.get("true_label"),
        )
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    elapsed = max(1e-9, time.perf_counter() - t0)
    return {
        "rows": float(len(rows)),
        "elapsed_s": float(elapsed),
        "throughput_flows_per_s": float(len(rows)) / float(elapsed),
        "latency_p95_ms": _p95(latencies_ms),
        "latency_mean_ms": float(sum(latencies_ms) / max(1, len(latencies_ms))),
    }


async def _main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    raw = yaml.safe_load(config_path.read_text()) or {}
    cfg = load_orchestrator_config(config_path)
    guardrail = (
        float(args.guardrail_overhead)
        if args.guardrail_overhead is not None
        else float(cfg.routing.langgraph_perf_guardrail_overhead)
    )
    rows = load_replay_dataset(args.dataset, max_rows=(args.max_flows or None))
    if not rows:
        raise RuntimeError("No rows loaded from dataset")

    with tempfile.TemporaryDirectory(prefix="bao_engines_") as td:
        tmp = Path(td)
        det_cfg = _build_runtime_config(base_raw=raw, base_cfg=cfg, engine="deterministic", temp_dir=tmp)
        lg_cfg = _build_runtime_config(base_raw=raw, base_cfg=cfg, engine="langgraph", temp_dir=tmp)

        det = IntegratedBAOSystem(det_cfg)
        lg = IntegratedBAOSystem(lg_cfg)

        det_stats = await _run_engine(det, rows)
        lg_stats = await _run_engine(lg, rows)

    throughput_overhead = 0.0
    latency_overhead = 0.0
    if det_stats["throughput_flows_per_s"] > 0:
        throughput_overhead = max(
            0.0,
            (det_stats["throughput_flows_per_s"] - lg_stats["throughput_flows_per_s"])
            / det_stats["throughput_flows_per_s"],
        )
    if det_stats["latency_p95_ms"] > 0:
        latency_overhead = max(
            0.0,
            (lg_stats["latency_p95_ms"] - det_stats["latency_p95_ms"]) / det_stats["latency_p95_ms"],
        )

    report = {
        "dataset": str(Path(args.dataset).resolve()),
        "rows": len(rows),
        "guardrail_overhead": guardrail,
        "deterministic": det_stats,
        "langgraph": lg_stats,
        "throughput_overhead": throughput_overhead,
        "latency_p95_overhead": latency_overhead,
        "passes_guardrail": bool(
            throughput_overhead <= float(guardrail) and latency_overhead <= float(guardrail)
        ),
    }

    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"Engine comparison report: {output}")
    print(
        f"passes_guardrail={report['passes_guardrail']} "
        f"throughput_overhead={throughput_overhead:.4f} latency_p95_overhead={latency_overhead:.4f}"
    )


if __name__ == "__main__":
    asyncio.run(_main())
