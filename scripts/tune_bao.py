#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import AGENT_CHOICES, _run_agent_baseline, _run_bao
from orchestrator.benchmarking import build_metric_reference
from orchestrator.config import load_config
from orchestrator.data import load_replay_dataset
from orchestrator.runtime import BAORuntime


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune BAO config on a validation dataset")
    p.add_argument("--dataset", default="data/UNSW_NB15_testing-set.csv")
    p.add_argument("--config", default="config/orchestrator_config.utility.yaml")
    p.add_argument("--output-report", default="artifacts/tuning/bao_tuning_report.json")
    p.add_argument("--output-config", default="artifacts/tuning/bao_tuned_config.yaml")
    p.add_argument("--max-flows", type=int, default=1000)
    p.add_argument("--candidate-limit", type=int, default=0)
    return p.parse_args()


def _resolve_base_payload(config_path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config must be a mapping")
    payload = deepcopy(payload)
    base_dir = config_path.parent
    payload.setdefault("orchestration", {})["agent_registry_path"] = str(
        (base_dir / payload.get("orchestration", {}).get("agent_registry_path", "agents.yaml")).resolve()
    )
    schema_path = payload.get("preprocessing", {}).get("schema_path")
    if schema_path:
        payload.setdefault("preprocessing", {})["schema_path"] = str((base_dir / schema_path).resolve())
    return payload


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _candidate_payloads(base: Dict[str, Any], empirical_prior: float) -> Iterable[Dict[str, Any]]:
    orders = list(itertools.permutations(AGENT_CHOICES, 3))
    for max_agents, prior_mode, uncertainty_threshold, min_net_gain, rho, reliability_strength, defer_threshold, order in itertools.product(
        (2, 3),
        ("empirical", "fixed_0_5"),
        (0.25, 0.40, 0.55),
        (-0.10, 0.0, 0.05),
        (0.7, 1.0),
        (0.5, 1.0),
        (0.60, 0.693),
        orders,
    ):
        prior = float(empirical_prior if prior_mode == "empirical" else 0.5)
        name = (
            f"order={'-'.join(order)}|max={max_agents}|prior={prior_mode}|uq={uncertainty_threshold}|"
            f"gain={min_net_gain}|rho={rho}|rel={reliability_strength}|defer={defer_threshold}"
        )
        yield {
            "name": name,
            "payload": _deep_merge(
                base,
                {
                    "orchestration": {"agent_sequence": list(order)},
                    "belief": {"prior_attack_rate": prior, "reliability_strength": reliability_strength},
                    "query": {"first_agent": order[0], "max_agents": max_agents, "uncertainty_threshold": uncertainty_threshold},
                    "voi": {"rho": rho, "min_net_gain": min_net_gain},
                    "decision": {"defer_policy": {"uncertainty_threshold": defer_threshold}},
                },
            ),
        }


def _evaluate_thresholded_baselines(rows, cfg) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for agent_id in AGENT_CHOICES:
        _, metrics, _ = _run_agent_baseline(agent_id=agent_id, rows=rows, cfg=cfg, family="thresholded_single_agent")
        results[agent_id] = metrics
    return results


def _choose_candidate(candidates: list[Dict[str, Any]], reference: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    best_accuracy = float(reference["accuracy"]["value"])
    best_utility = float(reference["utility_cost_total"]["value"])
    best_ece = float(reference["ece"]["value"])
    best_gap = float(reference["attack_cat_recall_gap"]["value"])

    qualified: list[Dict[str, Any]] = []
    strict_accuracy: list[Dict[str, Any]] = []
    for candidate in candidates:
        bao = candidate["metrics"]
        acceptance = {
            "beats_utility": float(bao["utility_cost_total"]) < best_utility,
            "beats_ece": float(bao["ece"]) < best_ece,
            "beats_attack_cat_recall_gap": float(bao["attack_cat_recall_gap"]) < best_gap,
            "within_2pp_accuracy": float(bao["accuracy"]) >= (best_accuracy - 0.02),
            "beats_best_accuracy": float(bao["accuracy"]) > best_accuracy,
        }
        candidate["acceptance"] = acceptance
        if all(acceptance[key] for key in ("beats_utility", "beats_ece", "beats_attack_cat_recall_gap", "within_2pp_accuracy")):
            qualified.append(candidate)
            if acceptance["beats_best_accuracy"]:
                strict_accuracy.append(candidate)

    pool = strict_accuracy or qualified or candidates
    selected = min(pool, key=lambda item: (float(item["metrics"]["utility_cost_total"]), -float(item["metrics"]["accuracy"])))
    return selected


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    base_payload = _resolve_base_payload(config_path)
    rows = load_replay_dataset(args.dataset, max_rows=(args.max_flows if args.max_flows > 0 else None))
    empirical_prior = sum(int(row["true_label"]) for row in rows) / float(max(1, len(rows)))

    base_cfg_path = Path(tempfile.mkdtemp()) / "base_cfg.yaml"
    base_payload.setdefault("state", {})["sqlite_path"] = str(base_cfg_path.parent / "base.sqlite")
    base_cfg_path.write_text(yaml.safe_dump(base_payload))
    base_cfg = load_config(base_cfg_path)
    thresholded_results = _evaluate_thresholded_baselines(rows, base_cfg)
    reference = build_metric_reference(thresholded_results)

    candidates: list[Dict[str, Any]] = []
    for idx, candidate in enumerate(_candidate_payloads(base_payload, empirical_prior), start=1):
        if int(args.candidate_limit) > 0 and idx > int(args.candidate_limit):
            break
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            payload = deepcopy(candidate["payload"])
            payload.setdefault("state", {})["sqlite_path"] = str(td_path / "state.sqlite")
            cfg_path = td_path / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(payload))
            cfg = load_config(cfg_path)
            runtime = BAORuntime(cfg, state_sqlite_path=td_path / "state.sqlite")
            _, metrics, _ = _run_bao(rows=rows, runtime=runtime)
            candidates.append({
                "name": candidate["name"],
                "payload": payload,
                "metrics": metrics,
            })
            print(f"[{idx}] {candidate['name']} utility={metrics['utility_cost_total']} accuracy={metrics['accuracy']}")

    if not candidates:
        raise RuntimeError("No tuning candidates evaluated")

    selected = _choose_candidate(candidates, reference)
    output_report = Path(args.output_report).resolve()
    output_report.parent.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "dataset": str(Path(args.dataset).resolve()),
        "flows_evaluated": len(rows),
        "reference": reference,
        "selected": {
            "name": selected["name"],
            "metrics": selected["metrics"],
            "acceptance": selected.get("acceptance", {}),
        },
        "candidates": [
            {
                "name": item["name"],
                "metrics": item["metrics"],
                "acceptance": item.get("acceptance", {}),
            }
            for item in sorted(candidates, key=lambda entry: float(entry["metrics"]["utility_cost_total"]))
        ],
    }
    output_report.write_text(json.dumps(report_payload, indent=2))

    output_config = Path(args.output_config).resolve()
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(yaml.safe_dump(selected["payload"], sort_keys=False))

    print(f"Saved tuning report: {output_report}")
    print(f"Saved tuned config: {output_config}")
    print(json.dumps(report_payload["selected"], indent=2))


if __name__ == "__main__":
    main()
