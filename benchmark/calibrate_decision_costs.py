#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator.core import UtilizationTarget, build_utilization_adjustments, compute_utilization_rates
from orchestrator.decision import DecisionCosts, min_expected_action_cost, realized_action_cost, select_expected_cost_action
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
    p.add_argument(
        "--uncertainty-threshold-grid",
        default=None,
        help="Comma-separated adaptive uncertainty thresholds (nats) for stop gating",
    )
    p.add_argument(
        "--defer-uncertainty-threshold-grid",
        default=None,
        help="Comma-separated defer uncertainty thresholds (nats)",
    )
    p.add_argument(
        "--defer-margin-grid",
        default=None,
        help="Comma-separated |p-0.5| defer margin candidates",
    )
    p.add_argument("--fusion-method", choices=["handoff_latest", "utility_select"], default=None)
    p.add_argument("--accuracy-floor-delta", type=float, default=None, help="Allowed drop vs strongest single agent")
    p.add_argument("--accuracy-margin-over-ocsvm", type=float, default=0.02)
    p.add_argument("--utility-ratio-vs-lstm", type=float, default=0.90)
    p.add_argument("--defer-rate-min", type=float, default=0.03)
    p.add_argument("--defer-rate-max", type=float, default=0.10)
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


def _entropy_nats(p: float) -> float:
    x = max(1e-9, min(1.0 - 1e-9, float(p)))
    return -(x * math.log(x) + (1.0 - x) * math.log(1.0 - x))


def _single_agent_utility_per_flow(
    *,
    rows: Dict[str, Dict],
    flow_ids: List[str],
    costs: DecisionCosts,
    query_cost: float,
) -> float:
    total = 0.0
    for fid in flow_ids:
        item = rows[fid]
        p = float(item["probability"])
        y = int(item["true_label"])
        pred = 1 if p >= 0.5 else 0
        action, _ = select_expected_cost_action(p, costs)
        total += float(query_cost)
        total += realized_action_cost(decision=action, prediction=pred, true_label=y, costs=costs)
    return total / float(max(1, len(flow_ids)))


@dataclass
class EvalResult:
    accuracy: float
    utility_cost_per_flow: float
    query_cost_per_flow: float
    action_cost_per_flow: float
    defer_count: int
    defer_rate: float
    utilization: Dict[str, float]
    utilization_weighted_violation: float
    utilization_violation_detail: Dict[str, Dict[str, float]]


@dataclass
class Candidate:
    costs: DecisionCosts
    max_agents: int
    min_expected_gain: float
    uncertainty_threshold: float
    defer_uncertainty_threshold: float
    defer_margin: float
    result: EvalResult


def _evaluate_costs(
    *,
    costs: DecisionCosts,
    sequence: List[str],
    first_agent: str,
    max_agents: int,
    min_expected_gain: float,
    uncertainty_threshold: float,
    defer_uncertainty_threshold: float,
    defer_margin: float,
    defer_require_all_agents_exhausted: bool,
    fusion_method: str,
    rows_by_agent: Dict[str, Dict[str, Dict]],
    common_flow_ids: List[str],
    agent_costs: Dict[str, float],
    reliability: Dict[str, float],
    profile_path: Path,
    utilization_targets: Dict[str, UtilizationTarget],
    utilization_warmup_flows: int,
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
    total_defer = 0
    agent_query_counts: Dict[str, int] = {aid: 0 for aid in sequence}

    ordered = [first_agent] + [a for a in sequence if a != first_agent]
    for flow_idx, fid in enumerate(common_flow_ids):
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
        agent_query_counts[first] = int(agent_query_counts.get(first, 0)) + 1

        while len(queried) < min(max_agents, len(available)):
            if _entropy_nats(current_p) <= float(uncertainty_threshold):
                break
            remaining = [a for a in available if a not in queried]
            if not remaining:
                break
            util_adjustments = build_utilization_adjustments(
                candidate_agents=remaining,
                agent_calls=agent_query_counts,
                flows_processed=int(flow_idx),
                targets=utilization_targets,
                warmup_flows=int(utilization_warmup_flows),
            )
            next_agent, _scores, _mode = router.select_next_agent(
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
                utilization_adjustments=util_adjustments,
                exploration_enabled=False,
                force_under_target_topup=False,
            )
            if next_agent is None:
                break

            queried.append(next_agent)
            row = rows_by_agent[next_agent][fid]
            queried_probs[next_agent] = float(row["probability"])
            query_cost += float(agent_costs[next_agent])
            last_agent = next_agent
            agent_query_counts[next_agent] = int(agent_query_counts.get(next_agent, 0)) + 1

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
        action_decision, _ = select_expected_cost_action(current_p, costs)
        queried_exhausted = len(queried) >= min(max_agents, len(available))
        if (
            (not bool(defer_require_all_agents_exhausted) or queried_exhausted)
            and _entropy_nats(current_p) >= float(defer_uncertainty_threshold)
            and abs(float(current_p) - 0.5) <= float(defer_margin)
        ):
            action_decision = "defer"
            total_defer += 1
        total_action_cost += realized_action_cost(
            decision=action_decision,
            prediction=pred,
            true_label=y,
            costs=costs,
        )

    n = max(1, len(common_flow_ids))
    utilization = compute_utilization_rates(agent_calls=agent_query_counts, flows_processed=n)
    violation_detail: Dict[str, Dict[str, float]] = {}
    weighted_violation = 0.0
    for aid, target in utilization_targets.items():
        rate = float(utilization.get(aid, 0.0))
        under = max(0.0, float(target.min_rate) - rate)
        over = max(0.0, rate - float(target.max_rate))
        weighted = (under * float(target.bonus_under)) + (over * float(target.penalty_over))
        weighted_violation += weighted
        violation_detail[aid] = {
            "rate": rate,
            "min_rate": float(target.min_rate),
            "max_rate": float(target.max_rate),
            "under": under,
            "over": over,
            "weighted": weighted,
        }

    return EvalResult(
        accuracy=total_correct / n,
        query_cost_per_flow=total_query_cost / n,
        action_cost_per_flow=total_action_cost / n,
        utility_cost_per_flow=(total_query_cost + total_action_cost) / n,
        defer_count=int(total_defer),
        defer_rate=float(total_defer) / float(n),
        utilization=utilization,
        utilization_weighted_violation=weighted_violation,
        utilization_violation_detail=violation_detail,
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

    if args.first_agent is not None:
        first_agent = str(args.first_agent).strip()
    elif cfg_model.orchestration.first_agent_strategy == "dynamic_cheapest":
        first_agent = min(
            sequence,
            key=lambda aid: float(handles.get(aid).cost) if aid in handles else float("inf"),
        )
    else:
        first_agent = str(cfg_model.query.first_agent or sequence[0]).strip()
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
    utilization_targets = {
            t.agent_id: UtilizationTarget(
                agent_id=t.agent_id,
                min_rate=float(t.min_rate),
                max_rate=float(t.max_rate),
                bonus_under=float(t.bonus_under),
                penalty_over=float(t.penalty_over),
            )
        for t in cfg_model.query.utilization_targets
    }
    utilization_warmup_flows = int(cfg_model.query.utilization_warmup_flows)

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
    ocsvm_accuracy = float(single_acc.get("ocsvm", single_acc.get(first_agent, strongest_single)))
    accuracy_margin_over_ocsvm = max(0.0, float(args.accuracy_margin_over_ocsvm))
    floor = ocsvm_accuracy + accuracy_margin_over_ocsvm
    utility_ratio_vs_lstm = max(0.0, float(args.utility_ratio_vs_lstm))
    defer_rate_min = max(0.0, float(args.defer_rate_min))
    defer_rate_max = max(defer_rate_min, float(args.defer_rate_max))

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

    uncertainty_threshold_grid = _list_or_default_float(
        args.uncertainty_threshold_grid,
        cfg_model.decision.cost_calibration.uncertainty_threshold_grid,
        [float(cfg_model.query.uncertainty_threshold)],
    )
    uncertainty_threshold_grid = [max(0.0, min(0.69314718056, float(x))) for x in uncertainty_threshold_grid]
    if float(cfg_model.query.uncertainty_threshold) not in uncertainty_threshold_grid:
        uncertainty_threshold_grid.append(float(cfg_model.query.uncertainty_threshold))

    defer_uncertainty_threshold_grid = _list_or_default_float(
        args.defer_uncertainty_threshold_grid,
        cfg_model.decision.cost_calibration.defer_uncertainty_threshold_grid,
        [float(cfg_model.decision.defer_policy.uncertainty_threshold)],
    )
    defer_uncertainty_threshold_grid = [max(0.0, min(0.69314718056, float(x))) for x in defer_uncertainty_threshold_grid]
    if float(cfg_model.decision.defer_policy.uncertainty_threshold) not in defer_uncertainty_threshold_grid:
        defer_uncertainty_threshold_grid.append(float(cfg_model.decision.defer_policy.uncertainty_threshold))

    defer_margin_grid = _list_or_default_float(
        args.defer_margin_grid,
        cfg_model.decision.cost_calibration.defer_margin_grid,
        [float(cfg_model.decision.defer_policy.margin_from_half)],
    )
    defer_margin_grid = [max(0.0, min(0.5, float(x))) for x in defer_margin_grid]
    if float(cfg_model.decision.defer_policy.margin_from_half) not in defer_margin_grid:
        defer_margin_grid.append(float(cfg_model.decision.defer_policy.margin_from_half))

    lstm_rows = rows_by_agent.get("lstm_autoencoder")
    if lstm_rows is None:
        raise RuntimeError("lstm_autoencoder replay is required for Pareto calibration")

    utility_ceiling_by_cost: Dict[tuple[float, float, float], float] = {}
    candidates: List[Candidate] = []
    for c_fn in fn_grid:
        for c_fp in fp_grid:
            for c_h in h_grid:
                costs = DecisionCosts(c_fn=float(c_fn), c_fp=float(c_fp), c_h=float(c_h))
                cost_key = (costs.c_fn, costs.c_fp, costs.c_h)
                if cost_key not in utility_ceiling_by_cost:
                    lstm_utility = _single_agent_utility_per_flow(
                        rows=lstm_rows,
                        flow_ids=common,
                        costs=costs,
                        query_cost=float(agent_costs.get("lstm_autoencoder", 0.0)),
                    )
                    utility_ceiling_by_cost[cost_key] = float(lstm_utility) * utility_ratio_vs_lstm
                for max_agents in max_agents_grid:
                    for min_gain in min_gain_grid:
                        for uncertainty_threshold in uncertainty_threshold_grid:
                            for defer_uncertainty_threshold in defer_uncertainty_threshold_grid:
                                for defer_margin in defer_margin_grid:
                                    result = _evaluate_costs(
                                        costs=costs,
                                        sequence=sequence,
                                        first_agent=first_agent,
                                        max_agents=max_agents,
                                        min_expected_gain=min_gain,
                                        uncertainty_threshold=uncertainty_threshold,
                                        defer_uncertainty_threshold=defer_uncertainty_threshold,
                                        defer_margin=defer_margin,
                                        defer_require_all_agents_exhausted=bool(
                                            cfg_model.decision.defer_policy.require_all_agents_exhausted
                                        ),
                                        fusion_method=fusion_method,
                                        rows_by_agent=rows_by_agent,
                                        common_flow_ids=common,
                                        agent_costs=agent_costs,
                                        reliability=reliability,
                                        profile_path=profile_path,
                                        utilization_targets=utilization_targets,
                                        utilization_warmup_flows=utilization_warmup_flows,
                                    )
                                    candidates.append(
                                        Candidate(
                                            costs=costs,
                                            max_agents=max_agents,
                                            min_expected_gain=min_gain,
                                            uncertainty_threshold=float(uncertainty_threshold),
                                            defer_uncertainty_threshold=float(defer_uncertainty_threshold),
                                            defer_margin=float(defer_margin),
                                            result=result,
                                        )
                                    )

    valid = [
        c
        for c in candidates
        if (
            c.result.accuracy >= floor
            and c.result.utility_cost_per_flow <= utility_ceiling_by_cost[(c.costs.c_fn, c.costs.c_fp, c.costs.c_h)]
            and defer_rate_min <= c.result.defer_rate <= defer_rate_max
        )
    ]
    if valid:
        best = min(
            valid,
            key=lambda x: (
                x.result.utility_cost_per_flow + x.result.utilization_weighted_violation,
                x.result.utility_cost_per_flow,
                -x.result.accuracy,
                x.result.utilization_weighted_violation,
            ),
        )
        selection_reason = "pareto_margin_constraints_satisfied"
    else:
        best = min(
            candidates,
            key=lambda x: (
                # Prioritize minimal total normalized constraint violation.
                max(0.0, floor - x.result.accuracy)
                + max(
                    0.0,
                    x.result.utility_cost_per_flow
                    - utility_ceiling_by_cost[(x.costs.c_fn, x.costs.c_fp, x.costs.c_h)],
                )
                + max(0.0, defer_rate_min - x.result.defer_rate)
                + max(0.0, x.result.defer_rate - defer_rate_max),
                x.result.utility_cost_per_flow,
                -x.result.accuracy,
            ),
        )
        selection_reason = "fallback_min_constraint_violation"

    best_costs = best.costs
    best_result = best.result

    payload = {
        "selection_reason": selection_reason,
        "accuracy_floor": floor,
        "accuracy_margin_over_ocsvm": accuracy_margin_over_ocsvm,
        "utility_ratio_vs_lstm": utility_ratio_vs_lstm,
        "defer_rate_min": defer_rate_min,
        "defer_rate_max": defer_rate_max,
        "strongest_single_accuracy": strongest_single,
        "ocsvm_accuracy": ocsvm_accuracy,
        "single_agent_accuracy": single_acc,
        "utility_ceiling_lstm_per_cost": {
            f"{k[0]}|{k[1]}|{k[2]}": v for k, v in utility_ceiling_by_cost.items()
        },
        "num_common_flows": len(common),
        "c_fn": best_costs.c_fn,
        "c_fp": best_costs.c_fp,
        "c_h": best_costs.c_h,
        "selected_max_agents": best.max_agents,
        "selected_min_expected_gain": best.min_expected_gain,
        "selected_uncertainty_threshold": best.uncertainty_threshold,
        "selected_defer_uncertainty_threshold": best.defer_uncertainty_threshold,
        "selected_defer_margin": best.defer_margin,
        "utilization_warmup_flows": utilization_warmup_flows,
        "utilization_targets": {
            aid: {
                "min_rate": float(t.min_rate),
                "max_rate": float(t.max_rate),
                "bonus_under": float(t.bonus_under),
                "penalty_over": float(t.penalty_over),
            }
            for aid, t in utilization_targets.items()
        },
        "result": {
            "accuracy": best_result.accuracy,
            "query_cost_per_flow": best_result.query_cost_per_flow,
            "action_cost_per_flow": best_result.action_cost_per_flow,
            "utility_cost_per_flow": best_result.utility_cost_per_flow,
            "defer_count": best_result.defer_count,
            "defer_rate": best_result.defer_rate,
            "utilization": best_result.utilization,
            "utilization_weighted_violation": best_result.utilization_weighted_violation,
            "utilization_violation_detail": best_result.utilization_violation_detail,
        },
    }
    output_json.write_text(json.dumps(payload, indent=2))

    cfg = yaml.safe_load(base_config_path.read_text()) or {}
    base_dir = base_config_path.parent
    decision = dict(cfg.get("decision", {}) or {})
    decision["costs"] = {"c_fn": best_costs.c_fn, "c_fp": best_costs.c_fp, "c_h": best_costs.c_h}
    decision["accuracy_floor_delta"] = float(accuracy_floor_delta)
    defer_policy_cfg = dict(decision.get("defer_policy", {}) or {})
    defer_policy_cfg["enabled"] = True
    defer_policy_cfg["uncertainty_threshold"] = float(best.defer_uncertainty_threshold)
    defer_policy_cfg["margin_from_half"] = float(best.defer_margin)
    defer_policy_cfg["require_all_agents_exhausted"] = bool(cfg_model.decision.defer_policy.require_all_agents_exhausted)
    decision["defer_policy"] = defer_policy_cfg
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
    query["uncertainty_threshold"] = float(best.uncertainty_threshold)
    query["exploration_uncertainty_threshold"] = float(best.uncertainty_threshold)
    query["apply_uncertainty_gate_in_adaptive"] = True
    query["force_under_target_topup"] = False
    query["exploration_enabled"] = False
    query["exploration_base_rate"] = 0.0
    query["exploration_max_rate"] = 0.0
    query["utilization_warmup_flows"] = int(utilization_warmup_flows)
    query["utilization_targets"] = [
        {
            "agent_id": aid,
            "min_rate": float(t.min_rate),
            "max_rate": float(t.max_rate),
            "bonus_under": float(t.bonus_under),
            "penalty_over": float(t.penalty_over),
        }
        for aid, t in utilization_targets.items()
    ]
    cfg["query"] = query

    fusion = dict(cfg.get("fusion", {}) or {})
    fusion["method"] = fusion_method
    cfg["fusion"] = fusion

    benchmark = dict(cfg.get("benchmark", {}) or {})
    benchmark["utility_evaluation"] = "cost_action_parity"
    cfg["benchmark"] = benchmark

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
