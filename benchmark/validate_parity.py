#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
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
    p = argparse.ArgumentParser(description="Validate parity between deterministic and LangGraph runtimes")
    p.add_argument("--dataset", default="data/UNSW_NB15_testing-set.csv")
    p.add_argument("--config", default="config/orchestrator_config.yaml")
    p.add_argument("--max-flows", type=int, default=0)
    p.add_argument("--output-json", default="artifacts/replay/parity_report.json")
    p.add_argument("--tolerance", type=float, default=None, help="Posterior tolerance (default: routing.parity_tolerance)")
    return p.parse_args()


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


async def _run(system: IntegratedBAOSystem, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        res = await system.process_flow(
            flow_features=row["flow_features"],
            flow_id=row["flow_id"],
            timestamp=row.get("timestamp") or 0.0,
            true_label=row.get("true_label"),
        )
        out.append(
            {
                "flow_id": row["flow_id"],
                "decision": res.get("decision"),
                "action_decision": res.get("action_decision"),
                "compromise_prob": float(res.get("compromise_prob", 0.5)),
                "cumulative_cost": float(res.get("cumulative_cost", 0.0)),
                "agents_queried": list(res.get("agents_queried") or []),
            }
        )
    return out


def _compare(det_rows: List[Dict[str, Any]], lg_rows: List[Dict[str, Any]], tolerance: float) -> Dict[str, Any]:
    mismatches: List[Dict[str, Any]] = []
    for i, (det, lg) in enumerate(zip(det_rows, lg_rows)):
        equal = (
            det["decision"] == lg["decision"]
            and det["action_decision"] == lg["action_decision"]
            and det["agents_queried"] == lg["agents_queried"]
            and abs(float(det["cumulative_cost"]) - float(lg["cumulative_cost"])) <= tolerance
            and abs(float(det["compromise_prob"]) - float(lg["compromise_prob"])) <= tolerance
        )
        if equal:
            continue
        mismatches.append(
            {
                "index": i,
                "flow_id": det["flow_id"],
                "deterministic": det,
                "langgraph": lg,
            }
        )
        if len(mismatches) >= 20:
            break

    return {
        "total_compared": min(len(det_rows), len(lg_rows)),
        "mismatch_count": len(mismatches),
        "matches": len(mismatches) == 0,
        "mismatches_sample": mismatches,
    }


async def _main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    raw = yaml.safe_load(config_path.read_text()) or {}
    cfg = load_orchestrator_config(config_path)
    tolerance = float(args.tolerance) if args.tolerance is not None else float(cfg.routing.parity_tolerance)
    rows = load_replay_dataset(args.dataset, max_rows=(args.max_flows or None))
    if not rows:
        raise RuntimeError("No rows loaded from dataset")

    with tempfile.TemporaryDirectory(prefix="bao_parity_") as td:
        tmp = Path(td)
        det_cfg = _build_runtime_config(base_raw=raw, base_cfg=cfg, engine="deterministic", temp_dir=tmp)
        lg_cfg = _build_runtime_config(base_raw=raw, base_cfg=cfg, engine="langgraph", temp_dir=tmp)

        det = IntegratedBAOSystem(det_cfg)
        lg = IntegratedBAOSystem(lg_cfg)

        det_rows, lg_rows = await asyncio.gather(_run(det, rows), _run(lg, rows))
        report = _compare(det_rows, lg_rows, tolerance=tolerance)
        report.update(
            {
                "dataset": str(Path(args.dataset).resolve()),
                "rows": len(rows),
                "tolerance": tolerance,
                "deterministic_config": str(det_cfg),
                "langgraph_config": str(lg_cfg),
            }
        )

    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"Parity report: {output}")
    print(f"matches={report['matches']} mismatch_count={report['mismatch_count']}")


if __name__ == "__main__":
    asyncio.run(_main())
