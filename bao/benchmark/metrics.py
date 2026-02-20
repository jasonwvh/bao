from __future__ import annotations

import math
import random
from dataclasses import asdict
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .types import BenchmarkConfig, PolicyResult


def confusion_counts(labels: Iterable[int], decisions: Iterable[str]) -> Tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for y, d in zip(labels, decisions):
        if d == "defer":
            # Conservative metric mapping for deferred flows:
            # defer acts as benign pass for malicious (FN risk) and benign for benign.
            if y == 1:
                fn += 1
            else:
                tn += 1
            continue
        pred = 1 if d == "reject" else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_auc(labels: List[int], scores: List[float]) -> float:
    # Rank-based AUROC, dependency-free.
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    rank_sum = 0.0
    for idx, (_, y) in enumerate(pairs, start=1):
        if y == 1:
            rank_sum += idx

    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def compute_ece(labels: List[int], scores: List[float], bins: int = 10) -> float:
    if not labels:
        return 0.0

    arr_y = np.array(labels)
    arr_p = np.array(scores)

    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (arr_p >= lo) & (arr_p < hi if i < bins - 1 else arr_p <= hi)
        if not np.any(mask):
            continue
        conf = arr_p[mask].mean()
        acc = arr_y[mask].mean()
        ece += (mask.sum() / len(arr_y)) * abs(acc - conf)
    return float(ece)


def compute_policy_metrics(result: PolicyResult) -> Dict[str, float]:
    tp, fp, tn, fn = confusion_counts(result.labels, result.decisions)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": compute_auc(result.labels, result.scores),
        "ece": compute_ece(result.labels, result.scores),
        "brier": float(np.mean((np.array(result.scores) - np.array(result.labels)) ** 2)),
        "avg_cost": float(np.mean(result.costs)),
        "p50_latency_ms": float(np.percentile(result.latencies_ms, 50)),
        "p95_latency_ms": float(np.percentile(result.latencies_ms, 95)),
        "avg_agent_calls": float(np.mean(result.agent_calls)),
        "defer_rate": float(np.mean(result.defer_flags)),
        "queue_load_proxy": float(np.mean(result.defer_flags) * len(result.labels)),
    }
    return metrics


def aggregate_by_policy(results: List[PolicyResult]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for r in results:
        grouped.setdefault(r.policy_name, []).append(compute_policy_metrics(r))

    out: Dict[str, Dict[str, float]] = {}
    for policy, vals in grouped.items():
        keys = vals[0].keys()
        row: Dict[str, float] = {}
        for k in keys:
            numbers = [v[k] for v in vals]
            row[f"{k}_mean"] = mean(numbers)
            row[f"{k}_std"] = pstdev(numbers) if len(numbers) > 1 else 0.0
        out[policy] = row
    return out


def paired_bootstrap_ci(deltas: List[float], n_samples: int, rng_seed: int = 7, alpha: float = 0.05) -> Tuple[float, float]:
    if not deltas:
        return (0.0, 0.0)

    rng = random.Random(rng_seed)
    samples = []
    for _ in range(n_samples):
        draw = [deltas[rng.randrange(len(deltas))] for _ in range(len(deltas))]
        samples.append(sum(draw) / len(draw))
    lo = np.percentile(samples, 100 * (alpha / 2.0))
    hi = np.percentile(samples, 100 * (1.0 - alpha / 2.0))
    return float(lo), float(hi)


def compare_against_bao(results: List[PolicyResult], cfg: BenchmarkConfig) -> Dict[str, Dict[str, object]]:
    by_seed_policy: Dict[Tuple[int, str], Dict[str, float]] = {}
    for r in results:
        by_seed_policy[(r.seed, r.policy_name)] = compute_policy_metrics(r)

    seeds = sorted({r.seed for r in results})
    policies = sorted({r.policy_name for r in results if r.policy_name != "bao"})

    verdicts: Dict[str, Dict[str, object]] = {}
    for policy in policies:
        delta_f1 = []
        delta_cost = []
        delta_ece = []
        delta_defer = []
        for seed in seeds:
            bao_m = by_seed_policy.get((seed, "bao"))
            base_m = by_seed_policy.get((seed, policy))
            if bao_m is None or base_m is None:
                continue
            delta_f1.append(bao_m["f1"] - base_m["f1"])
            delta_cost.append(bao_m["avg_cost"] - base_m["avg_cost"])
            delta_ece.append(bao_m["ece"] - base_m["ece"])
            delta_defer.append(bao_m["defer_rate"] - base_m["defer_rate"])

        f1_ci = paired_bootstrap_ci(delta_f1, cfg.bootstrap_samples, rng_seed=17)
        cost_ci = paired_bootstrap_ci(delta_cost, cfg.bootstrap_samples, rng_seed=19)

        mean_delta_f1 = float(np.mean(delta_f1)) if delta_f1 else 0.0
        mean_delta_cost = float(np.mean(delta_cost)) if delta_cost else 0.0

        avg_base_cost = float(np.mean([by_seed_policy[(s, policy)]["avg_cost"] for s in seeds if (s, policy) in by_seed_policy]))

        primary = (mean_delta_f1 >= 0 and mean_delta_cost <= -0.2 * max(avg_base_cost, 1e-9))
        primary_alt = (abs(mean_delta_cost) <= 1e-6 and mean_delta_f1 >= 0.03)
        secondary = (np.mean(delta_defer) <= 0) or (np.mean(delta_ece) <= 0)
        ci_gate = (f1_ci[0] > 0 and cost_ci[1] < 0)

        verdicts[policy] = {
            "delta_f1_mean": mean_delta_f1,
            "delta_cost_mean": mean_delta_cost,
            "delta_f1_ci95": f1_ci,
            "delta_cost_ci95": cost_ci,
            "primary": bool(primary),
            "primary_alt": bool(primary_alt),
            "secondary": bool(secondary),
            "ci_gate": bool(ci_gate),
            "improved": bool((primary or primary_alt) and secondary and ci_gate),
        }

    return verdicts
