#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator.decision import DecisionCosts, min_expected_action_cost, realized_action_cost
from orchestrator.config import load_orchestrator_config
from orchestrator.control.registry import load_registry, to_runtime_handles
from orchestrator.router import AdaptiveRouter
from orchestrator.types import AgentRuntimeHandle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate decision costs with an accuracy floor")
    p.add_argument("--input-root", default="artifacts/replay/matrix", help="Root containing replay_results_<agent>.json")
    p.add_argument("--profile-path", required=True, help="Router profile JSON")
    p.add_argument("--base-config", default="config/orchestrator_config.yaml", help="Base orchestrator config")
    p.add_argument("--output-json", default="artifacts/replay/cost_calibration.json", help="Calibration summary output")
    p.add_argument("--output-config", default="artifacts/replay/orchestrator_calibrated.yaml", help="Calibrated config output")
    p.add_argument("--agents", default=None, help="Comma-separated agent sequence (defaults to config)")
    p.add_argument("--first-agent", default=None, help="First queried agent (defaults to query.first_agent)")
    p.add_argument("--max-agents", type=int, default=None, help="Max agents queried per flow (defaults to config)")
    p.add_argument("--min-expected-gain", type=float, default=None, help="Adaptive router gain threshold (defaults to config)")
    p.add_argument(
        "--max-agents-grid",
        default=None,
        help="Optional comma-separated max_agents candidates (default uses --max-agents only)",
    )
    p.add_argument(
        "--min-expected-gain-grid",
        default=None,
        help="Comma-separated min_expected_gain candidates (defaults to config cost_calibration)",
    )
    p.add_argument("--fusion-method", choices=["handoff_latest", "utility_select"], default=None)
    p.add_argument("--accuracy-floor-delta", type=float, default=None, help="Allowed drop vs strongest single agent")
    p.add_argument("--c-fn-grid", default=None, help="Grid values for c_fn (comma-separated)")
    p.add_argument("--c-fp-grid", default=None, help="Grid values for c_fp (comma-separated)")
    p.add_argument("--c-h-grid", default=None, help="Grid values for c_h (comma-separated)")
    return p.parse_args()


def _parse_grid(spec: str) -> List[float]:
    return [float(x.strip()) for x in str(spec).split(",") if x.strip()]


def _list_or_default_float(cli_spec: str | None, cfg_values: List[float], fallback: List[float]) -> List[float]:
    if cli_spec not in (None, ""):
        vals = _parse_grid(str(cli_spec))
        return vals if vals else list(fallback)
    if cfg_values:
        return [float(x) for x in cfg_values]
    return list(fallback)


def _list_or_default_int(cli_spec: str | None, cfg_values: List[int], fallback: List[int]) -> List[int]:
    if cli_spec not in (None, ""):
        vals = [int(x.strip()) for x in str(cli_spec).split(",") if x.strip()]
        return vals if vals else list(fallback)
    if cfg_values:
        return [int(x) for x in cfg_values]
    return list(fallback)


def _find_replay_file(root: Path, agent_id: str) -> Path:
    candidates = [
        root / agent_id / f"replay_results_{agent_id}.json",
        root / f"replay_results_{agent_id}.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = sorted(root.glob(f"**/replay_results_{agent_id}.json"))
    if not matches:
        raise FileNotFoundError(f"missing replay file for {agent_id!r} under {root}")
    return matches[0]


class _StaticReliability:
    def __init__(self, reliability: Dict[str, float]):
        self.reliability = reliability

    def get_global_reliability(self, agent_id: str) -> float:
        return float(self.reliability.get(agent_id, 0.5))


@dataclass
class EvalResult:
    accuracy: float
    utility_cost_per_flow: float
    query_cost_per_flow: float
    action_cost_per_flow: float


@dataclass
class Candidate:
    costs: DecisionCosts
    max_agents: int
    min_expected_gain: float
    result: EvalResult


def _evaluate_costs(
    *,
    costs: DecisionCosts,
    sequence: List[str],
    first_agent: str,
    max_agents: int,
    min_expected_gain: float,
    fusion_method: str,
    rows_by_agent: Dict[str, Dict[str, Dict]],
    common_flow_ids: List[str],
    agent_costs: Dict[str, float],
    reliability: Dict[str, float],
    profile_path: Path,
) -> EvalResult:
    router = AdaptiveRouter(
        decision_costs=costs,
        profile_path=profile_path,
        min_samples_per_bin=1,
    )
    belief = _StaticReliability(reliability)

    total_correct = 0
    total_query_cost = 0.0
    total_action_cost = 0.0

    ordered = [first_agent] + [a for a in sequence if a != first_agent]
    for fid in common_flow_ids:
        available = [a for a in ordered if a in rows_by_agent]
        if not available:
            continue

        queried: List[str] = []
        queried_probs: Dict[str, float] = {}

        first = available[0]
        r0 = rows_by_agent[first][fid]
        current_p = float(r0["probability"])
        queried.append(first)
        queried_probs[first] = current_p
        query_cost = float(agent_costs[first])
        last_agent = first

        while len(queried) < min(max_agents, len(available)):
            remaining = [a for a in available if a not in queried]
            if not remaining:
                break
            next_agent, _scores = router.select_next_agent(
                current_probability=current_p,
                source_agent=last_agent,
                candidate_agents=remaining,
                agent_handles={
                    aid: AgentRuntimeHandle(
                        agent_id=aid,
                        endpoint="http://localhost",
                        transport="http-json",
                        timeout_ms=1000,
                        cost=float(agent_costs[aid]),
                        capabilities=[],
                        health_path="/a2a/health",
                        infer_path="/a2a/infer",
                        capabilities_path="/a2a/capabilities",
                        meta={},
                    )
                    for aid in remaining
                },
                belief_manager=belief,
                min_expected_gain=min_expected_gain,
            )
            if next_agent is None:
                break

            queried.append(next_agent)
            row = rows_by_agent[next_agent][fid]
            queried_probs[next_agent] = float(row["probability"])
            query_cost += float(agent_costs[next_agent])
            last_agent = next_agent

            if fusion_method == "handoff_latest":
                current_p = queried_probs[next_agent]
            else:
                best_agent = next_agent
                best_proxy = float("inf")
                best_prob = queried_probs[next_agent]
                for aid, prob in queried_probs.items():
                    rel = max(1e-6, float(reliability.get(aid, 0.5)))
                    proxy = min_expected_action_cost(prob, costs) / rel
                    if proxy < best_proxy:
                        best_proxy = proxy
                        best_agent = aid
                        best_prob = prob
                current_p = best_prob
                last_agent = best_agent

        y = int(rows_by_agent[first][fid]["true_label"])
        pred = 1 if current_p >= 0.5 else 0
        total_correct += int(pred == y)
        total_query_cost += query_cost
        total_action_cost += realized_action_cost(
            decision=None,
            prediction=pred,
            true_label=y,
            costs=costs,
        )

    n = max(1, len(common_flow_ids))
    return EvalResult(
        accuracy=total_correct / n,
        query_cost_per_flow=total_query_cost / n,
        action_cost_per_flow=total_action_cost / n,
        utility_cost_per_flow=(total_query_cost + total_action_cost) / n,
    )


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    profile_path = Path(args.profile_path).resolve()
    base_config_path = Path(args.base_config).resolve()
    cfg_model = load_orchestrator_config(base_config_path)
    output_json = Path(args.output_json).resolve()
    output_cfg = Path(args.output_config).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_cfg.parent.mkdir(parents=True, exist_ok=True)

    registry = load_registry(cfg_model.orchestration.agent_registry_path)
    handles = to_runtime_handles(registry)

    if args.agents in (None, ""):
        sequence = list(cfg_model.orchestration.agent_sequence) or list(handles.keys())
    else:
        sequence = [a.strip() for a in str(args.agents).split(",") if a.strip()]

    first_agent = str(args.first_agent if args.first_agent is not None else (cfg_model.query.first_agent or sequence[0])).strip()
    if first_agent not in sequence:
        sequence = [first_agent] + sequence

    max_agents_default = int(args.max_agents) if args.max_agents is not None else int(cfg_model.query.max_agents)
    min_expected_gain_default = (
        float(args.min_expected_gain) if args.min_expected_gain is not None else float(cfg_model.query.min_expected_gain)
    )
    fusion_method = str(
        args.fusion_method
        if args.fusion_method is not None
        else (cfg_model.decision.cost_calibration.fusion_method or cfg_model.fusion.method)
    ).strip().lower()
    accuracy_floor_delta = (
        float(args.accuracy_floor_delta)
        if args.accuracy_floor_delta is not None
        else float(cfg_model.decision.accuracy_floor_delta)
    )

    rows_by_agent: Dict[str, Dict[str, Dict]] = {}
    agent_costs: Dict[str, float] = {aid: float(handle.cost) for aid, handle in handles.items()}
    for aid in sequence:
        agent_costs.setdefault(aid, 0.0)
    reliability: Dict[str, float] = {}

    for aid in sequence:
        replay_path = _find_replay_file(input_root, aid)
        rows = json.loads(replay_path.read_text())
        by_flow = {str(r["flow_id"]): r for r in rows}
        rows_by_agent[aid] = by_flow
        correct = sum(1 for r in rows if int(r["prediction"]) == int(r["true_label"]))
        reliability[aid] = (correct / len(rows)) if rows else 0.5

    common_flow_ids = set(next(iter(rows_by_agent.values())).keys())
    for rows in rows_by_agent.values():
        common_flow_ids &= set(rows.keys())
    common = sorted(common_flow_ids)
    if not common:
        raise RuntimeError("No common flows in replay files for calibration")

    single_acc = {
        aid: sum(1 for fid in common if int(rows_by_agent[aid][fid]["prediction"]) == int(rows_by_agent[aid][fid]["true_label"])) / len(common)
        for aid in sequence
    }
    strongest_single = max(single_acc.values())
    floor = strongest_single - accuracy_floor_delta

    fn_grid = _list_or_default_float(
        args.c_fn_grid,
        cfg_model.decision.cost_calibration.c_fn_grid,
        [float(cfg_model.decision.c_fn)],
    )
    fp_grid = _list_or_default_float(
        args.c_fp_grid,
        cfg_model.decision.cost_calibration.c_fp_grid,
        [float(cfg_model.decision.c_fp)],
    )
    h_grid = _list_or_default_float(
        args.c_h_grid,
        cfg_model.decision.cost_calibration.c_h_grid,
        [float(cfg_model.decision.c_h)],
    )

    max_agents_grid = _list_or_default_int(
        args.max_agents_grid,
        cfg_model.decision.cost_calibration.max_agents_grid,
        [max_agents_default],
    )
    max_agents_grid = [max(1, int(v)) for v in max_agents_grid]

    min_gain_grid = _list_or_default_float(
        args.min_expected_gain_grid,
        cfg_model.decision.cost_calibration.min_expected_gain_grid,
        [min_expected_gain_default],
    )
    if min_expected_gain_default not in min_gain_grid:
        min_gain_grid.append(min_expected_gain_default)

    candidates: List[Candidate] = []
    for c_fn in fn_grid:
        for c_fp in fp_grid:
            for c_h in h_grid:
                for max_agents in max_agents_grid:
                    for min_gain in min_gain_grid:
                        costs = DecisionCosts(c_fn=float(c_fn), c_fp=float(c_fp), c_h=float(c_h))
                        result = _evaluate_costs(
                            costs=costs,
                            sequence=sequence,
                            first_agent=first_agent,
                            max_agents=max_agents,
                            min_expected_gain=min_gain,
                            fusion_method=fusion_method,
                            rows_by_agent=rows_by_agent,
                            common_flow_ids=common,
                            agent_costs=agent_costs,
                            reliability=reliability,
                            profile_path=profile_path,
                        )
                        candidates.append(
                            Candidate(
                                costs=costs,
                                max_agents=max_agents,
                                min_expected_gain=min_gain,
                                result=result,
                            )
                        )

    valid = [c for c in candidates if c.result.accuracy >= floor]
    if valid:
        best = min(valid, key=lambda x: x.result.utility_cost_per_flow)
        selection_reason = "min_utility_with_accuracy_floor"
    else:
        best = max(candidates, key=lambda x: x.result.accuracy)
        selection_reason = "fallback_max_accuracy_no_floor_match"

    best_costs = best.costs
    best_result = best.result

    payload = {
        "selection_reason": selection_reason,
        "accuracy_floor": floor,
        "strongest_single_accuracy": strongest_single,
        "single_agent_accuracy": single_acc,
        "num_common_flows": len(common),
        "c_fn": best_costs.c_fn,
        "c_fp": best_costs.c_fp,
        "c_h": best_costs.c_h,
        "selected_max_agents": best.max_agents,
        "selected_min_expected_gain": best.min_expected_gain,
        "result": {
            "accuracy": best_result.accuracy,
            "query_cost_per_flow": best_result.query_cost_per_flow,
            "action_cost_per_flow": best_result.action_cost_per_flow,
            "utility_cost_per_flow": best_result.utility_cost_per_flow,
        },
    }
    output_json.write_text(json.dumps(payload, indent=2))

    cfg = yaml.safe_load(base_config_path.read_text()) or {}
    base_dir = base_config_path.parent
    decision = dict(cfg.get("decision", {}) or {})
    decision["costs"] = {"c_fn": best_costs.c_fn, "c_fp": best_costs.c_fp, "c_h": best_costs.c_h}
    decision["accuracy_floor_delta"] = float(accuracy_floor_delta)
    cost_cal = dict(decision.get("cost_calibration", {}) or {})
    cost_cal["enabled"] = True
    cost_cal["mode"] = "validation_derived"
    cost_cal["fusion_method"] = fusion_method
    decision["cost_calibration"] = cost_cal
    cfg["decision"] = decision

    query = dict(cfg.get("query", {}) or {})
    query["policy"] = "adaptive_router"
    query["first_agent"] = first_agent
    query["max_agents"] = int(best.max_agents)
    query["min_expected_gain"] = float(best.min_expected_gain)
    cfg["query"] = query

    fusion = dict(cfg.get("fusion", {}) or {})
    fusion["method"] = fusion_method
    cfg["fusion"] = fusion

    routing = dict(cfg.get("routing", {}) or {})
    routing["profile_path"] = str(profile_path)
    cfg["routing"] = routing

    orch = dict(cfg.get("orchestration", {}) or {})
    orch["agent_sequence"] = sequence
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

    preprocessing = dict(cfg.get("preprocessing", {}) or {})
    if "schema_path" in preprocessing and preprocessing["schema_path"] not in (None, ""):
        p = Path(str(preprocessing["schema_path"]))
        if not p.is_absolute():
            preprocessing["schema_path"] = str((base_dir / p).resolve())
    cfg["preprocessing"] = preprocessing

    output_cfg.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    print(f"Calibration JSON: {output_json}")
    print(f"Calibrated config: {output_cfg}")


if __name__ == "__main__":
    main()
