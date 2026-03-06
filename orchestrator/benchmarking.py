from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from agents.common.calibration import brier_score, expected_calibration_error
from orchestrator.decisioning import DecisionCosts, probability_to_prediction, realized_action_cost


LOWER_IS_BETTER_METRICS = {"utility_cost_total", "ece", "brier", "attack_cat_recall_gap"}
HIGHER_IS_BETTER_METRICS = {"accuracy", "precision", "recall", "f1", "auc"}


def evaluation_prediction(*, probability: float, decision: str, true_label: int) -> int:
    d = str(decision).strip().lower()
    if d == "defer":
        return int(true_label)
    return probability_to_prediction(float(probability))


def threshold_decision(probability: float) -> str:
    return "reject" if float(probability) >= 0.5 else "accept"


def compute_auc(labels: List[int], probs: List[float]) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        score = float(roc_auc_score(labels, probs))
        if math.isnan(score):
            return 0.5
        return score
    except Exception:
        return 0.5


def _round(value: Any, digits: int = 6) -> float:
    return round(float(value), digits)


def compute_attack_cat_recall_gap(
    *,
    labels: List[int],
    predictions: List[int],
    metadata_rows: List[Dict[str, Any]],
    min_group_size: int = 50,
) -> Dict[str, Any]:
    grouped: Dict[str, List[int]] = {}
    for y, pred, meta in zip(labels, predictions, metadata_rows):
        if int(y) != 1:
            continue
        group = str((meta or {}).get("attack_cat", "")).strip()
        if not group or group.lower() == "normal":
            continue
        grouped.setdefault(group, []).append(int(pred))

    included: Dict[str, Dict[str, Any]] = {}
    excluded: Dict[str, Dict[str, Any]] = {}
    recalls: List[float] = []
    for group, vals in sorted(grouped.items()):
        support = int(len(vals))
        recall = float(np.mean(np.asarray(vals, dtype=np.float64))) if support > 0 else 0.0
        payload = {
            "support": support,
            "recall": _round(recall),
        }
        if support >= int(min_group_size):
            included[group] = payload
            recalls.append(recall)
        else:
            excluded[group] = payload

    gap = max(recalls) - min(recalls) if len(recalls) >= 2 else 0.0
    return {
        "field": "attack_cat",
        "metric": "recall_gap",
        "min_group_size": int(min_group_size),
        "value": _round(gap),
        "included_groups": included,
        "excluded_groups": excluded,
    }


@dataclass
class MetricsAccumulator:
    costs: DecisionCosts
    group_min_size: int = 50
    labels: List[int] = field(default_factory=list)
    predictions: List[int] = field(default_factory=list)
    probabilities: List[float] = field(default_factory=list)
    query_costs: List[float] = field(default_factory=list)
    action_costs: List[float] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    metadata_rows: List[Dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        *,
        true_label: int,
        probability: float,
        decision: str,
        query_cost: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        p = float(probability)
        y = int(true_label)
        d = str(decision).strip().lower()
        pred = evaluation_prediction(probability=p, decision=d, true_label=y)
        a_cost = realized_action_cost(decision=d, true_label=y, costs=self.costs)

        self.labels.append(y)
        self.predictions.append(pred)
        self.probabilities.append(p)
        self.query_costs.append(float(query_cost))
        self.action_costs.append(float(a_cost))
        self.decisions.append(d)
        self.metadata_rows.append(dict(metadata or {}))

    def compute(self, *, approach: str, family: str, summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        tp = sum(1 for p, y in zip(self.predictions, self.labels) if p == 1 and y == 1)
        tn = sum(1 for p, y in zip(self.predictions, self.labels) if p == 0 and y == 0)
        fp = sum(1 for p, y in zip(self.predictions, self.labels) if p == 1 and y == 0)
        fn = sum(1 for p, y in zip(self.predictions, self.labels) if p == 0 and y == 1)

        total = max(1, len(self.labels))
        accuracy = (tp + tn) / float(total)
        precision = tp / float(max(1, tp + fp))
        recall = tp / float(max(1, tp + fn))
        f1 = (2.0 * precision * recall) / float(max(1e-9, precision + recall))
        auc = compute_auc(self.labels, self.probabilities)
        ece = expected_calibration_error(np.asarray(self.probabilities, dtype=np.float64), np.asarray(self.labels, dtype=np.int64), bins=15)
        brier = brier_score(np.asarray(self.probabilities, dtype=np.float64), np.asarray(self.labels, dtype=np.int64))

        query_total = float(sum(self.query_costs))
        action_total = float(sum(self.action_costs))
        utility_total = query_total + action_total
        defer_count = sum(1 for d in self.decisions if d == "defer")
        group_metrics = compute_attack_cat_recall_gap(
            labels=self.labels,
            predictions=self.predictions,
            metadata_rows=self.metadata_rows,
            min_group_size=self.group_min_size,
        )

        out: Dict[str, Any] = {
            "approach": approach,
            "family": family,
            "flows_processed": int(len(self.labels)),
            "accuracy": _round(accuracy),
            "precision": _round(precision),
            "recall": _round(recall),
            "f1": _round(f1),
            "auc": _round(auc),
            "ece": _round(ece),
            "brier": _round(brier),
            "avg_cost_per_flow": round(query_total / float(total), 4),
            "query_cost_total": round(query_total, 4),
            "action_cost_total": round(action_total, 4),
            "utility_cost_total": round(utility_total, 4),
            "utility_cost_per_flow": round(utility_total / float(total), 4),
            "defer_count": int(defer_count),
            "defer_rate": _round(float(defer_count) / float(total)),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "attack_cat_recall_gap": float(group_metrics["value"]),
            "group_metrics": group_metrics,
        }
        if isinstance(summary, dict):
            for key in ("agent_utilization", "routing_selection_counts", "routing", "agent_costs"):
                if key in summary:
                    out[key] = summary[key]
        return out


def compute_percent_gain(value: float, baseline: float, *, lower_is_better: bool) -> Optional[float]:
    base = float(baseline)
    if not math.isfinite(base) or abs(base) <= 1e-12:
        return None
    current = float(value)
    if lower_is_better:
        return _round(((base - current) / base) * 100.0)
    return _round(((current - base) / base) * 100.0)


def build_metric_reference(thresholded_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    if not thresholded_results:
        return best

    for metric in sorted(LOWER_IS_BETTER_METRICS | HIGHER_IS_BETTER_METRICS):
        candidates = []
        for agent_id, metrics in thresholded_results.items():
            if metric not in metrics:
                continue
            value = float(metrics[metric])
            if not math.isfinite(value):
                continue
            candidates.append((agent_id, value))
        if not candidates:
            continue
        if metric in LOWER_IS_BETTER_METRICS:
            agent_id, value = min(candidates, key=lambda item: item[1])
        else:
            agent_id, value = max(candidates, key=lambda item: item[1])
        best[metric] = {"agent_id": agent_id, "value": _round(value)}
    return best


def attach_reference_deltas(metrics: Dict[str, Any], reference: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    deltas: Dict[str, Any] = {}
    for metric, ref in reference.items():
        if metric not in metrics:
            continue
        lower_is_better = metric in LOWER_IS_BETTER_METRICS
        gain = compute_percent_gain(float(metrics[metric]), float(ref["value"]), lower_is_better=lower_is_better)
        deltas[metric] = {
            "baseline_agent_id": str(ref["agent_id"]),
            "baseline_value": float(ref["value"]),
            "value": _round(float(metrics[metric])),
            "absolute_delta": _round(float(metrics[metric]) - float(ref["value"])),
            "percent_gain": gain,
            "higher_is_better": bool(not lower_is_better),
        }
    metrics["deltas_vs_best_thresholded_single_agent"] = deltas
    return metrics


def build_comparison_block(
    *,
    bao_metrics: Dict[str, Any],
    thresholded_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    reference = build_metric_reference(thresholded_results)
    if not reference:
        return {}

    headline: Dict[str, Any] = {}
    for metric, ref in reference.items():
        if metric not in bao_metrics:
            continue
        lower_is_better = metric in LOWER_IS_BETTER_METRICS
        bao_value = float(bao_metrics[metric])
        base_value = float(ref["value"])
        headline[metric] = {
            "baseline_family": "thresholded_single_agent",
            "baseline_agent_id": str(ref["agent_id"]),
            "baseline_value": base_value,
            "bao_value": _round(bao_value),
            "absolute_delta": _round(bao_value - base_value),
            "percent_gain": compute_percent_gain(bao_value, base_value, lower_is_better=lower_is_better),
            "higher_is_better": bool(not lower_is_better),
        }

    return {
        "reference_family": "thresholded_single_agent",
        "best_single_agent_by_metric": reference,
        "bao_vs_best_single_agent": headline,
    }
