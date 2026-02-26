#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.runner import (
    BenchmarkAccumulator,
    attach_routing_observability,
    build_benchmark_manifest,
    dataset_composition,
    reset_sqlite_state,
    write_json,
)
from orchestrator.config import (
    FIRST_AGENT_STRATEGIES,
    FUSION_METHODS,
    ORCHESTRATION_ENGINES,
    PREDICTION_SOURCES,
    QUERY_POLICIES,
    UTILITY_EVALUATIONS,
    load_orchestrator_config,
)
from orchestrator.data.replay import load_replay_dataset
from orchestrator.decision import DecisionCosts
from orchestrator.integrated_system import IntegratedBAOSystem


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run orchestrator replay pipeline")
    p.add_argument("--dataset", default="data/UNSW_NB15_testing-set.csv", help="CSV or parquet with label column")
    p.add_argument("--config", default="config/orchestrator_config.utility.yaml", help="Path to orchestrator config YAML")
    p.add_argument("--max-flows", type=int, default=0, help="Max flows to process (0=all)")
    p.add_argument("--output-dir", default="artifacts/replay", help="Output directory")
    p.add_argument("--seed", type=int, default=7, help="Seed")
    p.add_argument("--max-agents", type=int, default=None, help="Override query.max_agents")
    p.add_argument("--query-policy", choices=sorted(QUERY_POLICIES), default=None, help="Override query.policy")
    p.add_argument("--first-agent", default=None, help="Override query.first_agent")
    p.add_argument("--min-expected-gain", type=float, default=None, help="Override query.min_expected_gain")
    p.add_argument(
        "--agent-sequence",
        default=None,
        help="Comma-separated agent sequence override (e.g. lstm_autoencoder,ocsvm)",
    )
    p.add_argument("--fusion-method", choices=sorted(FUSION_METHODS), default=None, help="Override fusion.method")
    p.add_argument("--router-profile", default=None, help="Override routing.profile_path")
    p.add_argument(
        "--update-mode",
        choices=["posterior_first", "likelihood_strict"],
        default=None,
        help="Override orchestration update mode",
    )
    p.add_argument("--engine", choices=sorted(ORCHESTRATION_ENGINES), default=None, help="Override orchestration.engine")
    p.add_argument(
        "--first-agent-strategy",
        choices=sorted(FIRST_AGENT_STRATEGIES),
        default=None,
        help="Override orchestration.first_agent_strategy",
    )
    p.add_argument("--cost-calibration-json", default=None, help="Optional JSON with calibrated c_fn/c_fp/c_h")
    p.add_argument("--prediction-source", choices=sorted(PREDICTION_SOURCES), default=None)
    p.add_argument("--utility-evaluation", choices=sorted(UTILITY_EVALUATIONS), default=None)
    p.add_argument("--diagnostic-dataset", default=None, help="Optional secondary replay dataset")
    p.add_argument("--reset-state", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--write-manifest", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument(
        "--auto-recalibrate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Auto-generate calibrated BAO config/profile before running benchmark",
    )
    return p.parse_args()


def _apply_overrides(raw_config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(raw_config)

    orch = dict(cfg.get("orchestration", {}) or {})
    orch["seed"] = int(args.seed)
    if args.update_mode is not None:
        orch["update_mode"] = str(args.update_mode)
    if args.engine is not None:
        orch["engine"] = str(args.engine)
    if args.first_agent_strategy is not None:
        orch["first_agent_strategy"] = str(args.first_agent_strategy)
    if args.agent_sequence is not None:
        orch["agent_sequence"] = [x.strip() for x in str(args.agent_sequence).split(",") if x.strip()]
    cfg["orchestration"] = orch

    fusion = dict(cfg.get("fusion", {}) or {})
    if args.fusion_method is not None:
        fusion["method"] = str(args.fusion_method)
    cfg["fusion"] = fusion

    decision = dict(cfg.get("decision", {}) or {})
    if args.cost_calibration_json is not None:
        payload = json.loads(Path(args.cost_calibration_json).read_text())
        costs = dict(decision.get("costs", {}) or {})
        if "c_fn" in payload:
            costs["c_fn"] = float(payload["c_fn"])
        if "c_fp" in payload:
            costs["c_fp"] = float(payload["c_fp"])
        if "c_h" in payload:
            costs["c_h"] = float(payload["c_h"])
        decision["costs"] = costs
    cfg["decision"] = decision

    benchmark = dict(cfg.get("benchmark", {}) or {})
    if args.prediction_source is not None:
        benchmark["prediction_source"] = str(args.prediction_source)
    if args.utility_evaluation is not None:
        benchmark["utility_evaluation"] = str(args.utility_evaluation)
    if args.reset_state is not None:
        benchmark["reset_state"] = bool(args.reset_state)
    if args.write_manifest is not None:
        benchmark["write_manifest"] = bool(args.write_manifest)
    cfg["benchmark"] = benchmark

    query = dict(cfg.get("query", {}) or {})
    if args.max_agents is not None:
        query["max_agents"] = int(args.max_agents)
    if args.query_policy is not None:
        query["policy"] = str(args.query_policy)
    if args.first_agent is not None:
        query["first_agent"] = str(args.first_agent)
    if args.min_expected_gain is not None:
        query["min_expected_gain"] = float(args.min_expected_gain)
    cfg["query"] = query

    routing = dict(cfg.get("routing", {}) or {})
    if args.router_profile is not None:
        routing["profile_path"] = str(Path(args.router_profile).expanduser().resolve())
    cfg["routing"] = routing

    return cfg


def _absolutize_config_paths(config: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    cfg = dict(config)

    orch = dict(cfg.get("orchestration", {}) or {})
    if "agent_registry_path" in orch:
        p = Path(str(orch["agent_registry_path"]))
        if not p.is_absolute():
            orch["agent_registry_path"] = str((base_dir / p).resolve())
    cfg["orchestration"] = orch

    state = dict(cfg.get("state", {}) or {})
    if "sqlite_path" in state:
        p = Path(str(state["sqlite_path"]))
        if not p.is_absolute():
            state["sqlite_path"] = str((base_dir / p).resolve())
    cfg["state"] = state

    logging_cfg = dict(cfg.get("logging", {}) or {})
    if "jsonl_path" in logging_cfg:
        p = Path(str(logging_cfg["jsonl_path"]))
        if not p.is_absolute():
            logging_cfg["jsonl_path"] = str((base_dir / p).resolve())
    cfg["logging"] = logging_cfg

    pre = dict(cfg.get("preprocessing", {}) or {})
    if "schema_path" in pre and pre["schema_path"] not in (None, ""):
        p = Path(str(pre["schema_path"]))
        if not p.is_absolute():
            pre["schema_path"] = str((base_dir / p).resolve())
    cfg["preprocessing"] = pre

    routing = dict(cfg.get("routing", {}) or {})
    if "profile_path" in routing and routing["profile_path"] not in (None, ""):
        p = Path(str(routing["profile_path"]))
        if not p.is_absolute():
            routing["profile_path"] = str((base_dir / p).resolve())
    cfg["routing"] = routing

    return cfg


async def _run_dataset(
    *,
    system: IntegratedBAOSystem,
    rows: List[Dict[str, Any]],
    results_path: Path,
    prediction_source: str,
    utility_evaluation: str,
    decision_costs: DecisionCosts,
    approach: str,
) -> Dict[str, Any]:
    if results_path.exists():
        results_path.unlink()

    acc = BenchmarkAccumulator(
        prediction_source=prediction_source,
        utility_evaluation=utility_evaluation,
        decision_costs=decision_costs,
    )

    for row in rows:
        res = await system.process_flow(
            flow_features=row["flow_features"],
            flow_id=row["flow_id"],
            timestamp=row.get("timestamp") or time.time(),
            true_label=row.get("true_label"),
        )

        compact = {
            "flow_id": row["flow_id"],
            "decision": res.get("decision"),
            "action_decision": res.get("action_decision"),
            "compromise_prob": res.get("compromise_prob"),
            "epistemic_uncertainty": res.get("epistemic_uncertainty"),
            "combined_uncertainty": res.get("combined_uncertainty"),
            "cumulative_cost": res.get("cumulative_cost"),
            "agents_queried": res.get("agents_queried"),
        }
        with results_path.open("a") as f:
            f.write(json.dumps(compact) + "\n")

        true_label = row.get("true_label")
        if true_label is None:
            continue

        acc.add_sample(
            true_label=int(true_label),
            probability=float(res.get("compromise_prob", 0.5)),
            cost=float(res.get("cumulative_cost", 0.0)),
            decision=res.get("decision"),
            action_decision=res.get("action_decision"),
        )

    return acc.compute(approach=approach)


def _run_cmd(cmd: List[str]) -> None:
    logging.getLogger("main").info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def _build_calibrated_config(
    *,
    dataset: str,
    base_config_path: Path,
    output_dir: Path,
    prediction_source: str,
    utility_evaluation: str,
    write_manifest: bool,
    max_flows: int,
) -> Path:
    calibration_root = output_dir / "recalibration"
    calibration_root.mkdir(parents=True, exist_ok=True)

    max_flows_args: List[str] = []
    if int(max_flows or 0) > 0:
        max_flows_args = ["--max-flows", str(int(max_flows))]
    manifest_flag = "--write-manifest" if bool(write_manifest) else "--no-write-manifest"

    for aid, script in (
        ("ocsvm", "agents/ocsvm/benchmark.py"),
        ("lstm_autoencoder", "agents/lstm_autoencoder/benchmark.py"),
        ("wgan_gp", "agents/wgan_gp/benchmark.py"),
    ):
        _run_cmd(
            [
                sys.executable,
                script,
                "--dataset",
                dataset,
                "--config",
                str(base_config_path),
                "--output-dir",
                str(calibration_root / aid),
                "--prediction-source",
                prediction_source,
                "--utility-evaluation",
                utility_evaluation,
                manifest_flag,
                *max_flows_args,
            ]
        )

    router_profile_path = output_dir / "router_profile.json"
    _run_cmd(
        [
            sys.executable,
            "benchmark/build_router_profile.py",
            "--config",
            str(base_config_path),
            "--input-root",
            str(calibration_root),
            "--output-path",
            str(router_profile_path),
        ]
    )

    calibration_json = output_dir / "cost_calibration.json"
    calibrated_cfg_path = output_dir / "effective_orchestrator_config_calibrated.yaml"
    _run_cmd(
        [
            sys.executable,
            "benchmark/calibrate_decision_costs.py",
            "--base-config",
            str(base_config_path),
            "--input-root",
            str(calibration_root),
            "--profile-path",
            str(router_profile_path),
            "--output-json",
            str(calibration_json),
            "--output-config",
            str(calibrated_cfg_path),
        ]
    )

    return calibrated_cfg_path


async def _run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config).resolve()
    raw_config = yaml.safe_load(config_path.read_text()) or {}
    config = _apply_overrides(raw_config, args)
    config = _absolutize_config_paths(config, config_path.parent)

    runtime_config_path = output_dir / "effective_orchestrator_config.yaml"
    runtime_config_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))

    cfg = load_orchestrator_config(runtime_config_path)
    auto_recalibrate = bool(cfg.decision.cost_calibration.enabled) if args.auto_recalibrate is None else bool(args.auto_recalibrate)
    if auto_recalibrate:
        runtime_config_path = _build_calibrated_config(
            dataset=args.dataset,
            base_config_path=runtime_config_path,
            output_dir=output_dir,
            prediction_source=cfg.benchmark.prediction_source,
            utility_evaluation=cfg.benchmark.utility_evaluation,
            write_manifest=cfg.benchmark.write_manifest,
            max_flows=int(args.max_flows or 0),
        )
        cfg = load_orchestrator_config(runtime_config_path)

    if cfg.benchmark.reset_state:
        reset_sqlite_state(cfg.state.sqlite_path)

    rows = load_replay_dataset(args.dataset, max_rows=(args.max_flows or None))
    if not rows:
        raise RuntimeError("No rows loaded from dataset")

    system = IntegratedBAOSystem(config_path=runtime_config_path)

    results_path = output_dir / "replay_results.jsonl"
    benchmark_metrics = await _run_dataset(
        system=system,
        rows=rows,
        results_path=results_path,
        prediction_source=cfg.benchmark.prediction_source,
        utility_evaluation=cfg.benchmark.utility_evaluation,
        decision_costs=DecisionCosts(c_fn=cfg.decision.c_fn, c_fp=cfg.decision.c_fp, c_h=cfg.decision.c_h),
        approach="bao",
    )

    summary = system.get_system_statistics()
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    benchmark_metrics = attach_routing_observability(benchmark_metrics, summary)

    benchmark_path = output_dir / "benchmark_bao.json"
    write_json(benchmark_path, benchmark_metrics)

    if cfg.benchmark.write_manifest:
        manifest = build_benchmark_manifest(
            repo_root=REPO_ROOT,
            dataset_path=Path(args.dataset).resolve(),
            config_path=runtime_config_path,
            approach="bao",
            agents_used=list(system.agent_sequence),
            extra={
                "prediction_source": cfg.benchmark.prediction_source,
                "utility_evaluation": cfg.benchmark.utility_evaluation,
                "dataset_composition": dataset_composition(rows),
                "summary": summary,
            },
        )
        write_json(output_dir / "benchmark_bao_manifest.json", manifest)

    print(f"Benchmark metrics: {benchmark_path}")
    print(f"Processed {summary['flows_processed']} flows")
    print(f"Replay output: {results_path}")
    print(f"Summary: {summary_path}")

    if args.diagnostic_dataset:
        diagnostic_rows = load_replay_dataset(args.diagnostic_dataset, max_rows=(args.max_flows or None))
        if diagnostic_rows:
            if cfg.benchmark.reset_state:
                reset_sqlite_state(cfg.state.sqlite_path)
            diagnostic_system = IntegratedBAOSystem(config_path=runtime_config_path)
            diagnostic_results = output_dir / "replay_results_diagnostic.jsonl"
            diagnostic_metrics = await _run_dataset(
                system=diagnostic_system,
                rows=diagnostic_rows,
                results_path=diagnostic_results,
                prediction_source=cfg.benchmark.prediction_source,
                utility_evaluation=cfg.benchmark.utility_evaluation,
                decision_costs=DecisionCosts(c_fn=cfg.decision.c_fn, c_fp=cfg.decision.c_fp, c_h=cfg.decision.c_h),
                approach="bao_diagnostic",
            )
            write_json(output_dir / "benchmark_bao_diagnostic.json", diagnostic_metrics)
            if cfg.benchmark.write_manifest:
                diagnostic_manifest = build_benchmark_manifest(
                    repo_root=REPO_ROOT,
                    dataset_path=Path(args.diagnostic_dataset).resolve(),
                    config_path=runtime_config_path,
                    approach="bao_diagnostic",
                    agents_used=list(diagnostic_system.agent_sequence),
                    extra={
                        "prediction_source": cfg.benchmark.prediction_source,
                        "utility_evaluation": cfg.benchmark.utility_evaluation,
                        "dataset_composition": dataset_composition(diagnostic_rows),
                        "summary": diagnostic_system.get_system_statistics(),
                    },
                )
                write_json(output_dir / "benchmark_bao_diagnostic_manifest.json", diagnostic_manifest)
            print(f"Diagnostic benchmark: {output_dir / 'benchmark_bao_diagnostic.json'}")


def main() -> None:
    args = parse_args()
    setup_logging()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
