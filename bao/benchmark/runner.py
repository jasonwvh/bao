from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .agents import build_default_agents
from .calibration import fit_shared_calibrator
from .metrics import aggregate_by_policy, compare_against_bao, compute_policy_metrics
from .policies import build_policies
from .types import BenchmarkConfig, FlowRecord, PolicyResult


def load_labeled_csv(path: str) -> List[FlowRecord]:
    rows: List[FlowRecord] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            label = int(float(row["label"]))
            flow_id = row.get("flow_id") or f"flow_{idx:07d}"
            features = {
                k: float(v)
                for k, v in row.items()
                if k not in {"label", "flow_id"} and v is not None and str(v) != ""
            }
            rows.append(FlowRecord(flow_id=flow_id, label=label, features=features))
    return rows


def _split_calibration_eval(records: List[FlowRecord], frac: float) -> tuple[List[FlowRecord], List[FlowRecord]]:
    n = len(records)
    k = max(1, int(n * frac))
    k = min(k, n - 1) if n > 1 else 1
    return records[:k], records[k:]


def _save_results_csv(results: List[PolicyResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "policy",
        "precision",
        "recall",
        "f1",
        "auroc",
        "ece",
        "brier",
        "avg_cost",
        "p50_latency_ms",
        "p95_latency_ms",
        "avg_agent_calls",
        "defer_rate",
        "queue_load_proxy",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            m = compute_policy_metrics(r)
            writer.writerow(
                {
                    "seed": r.seed,
                    "policy": r.policy_name,
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "auroc": m["auroc"],
                    "ece": m["ece"],
                    "brier": m["brier"],
                    "avg_cost": m["avg_cost"],
                    "p50_latency_ms": m["p50_latency_ms"],
                    "p95_latency_ms": m["p95_latency_ms"],
                    "avg_agent_calls": m["avg_agent_calls"],
                    "defer_rate": m["defer_rate"],
                    "queue_load_proxy": m["queue_load_proxy"],
                }
            )


def _write_summary_md(
    out_path: Path,
    aggregate: Dict[str, Dict[str, float]],
    verdicts: Dict[str, Dict[str, object]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# BAO Benchmark Summary")
    lines.append("")
    lines.append("## Aggregate Metrics (mean ± std across seeds)")
    lines.append("")
    lines.append("| Policy | F1 | AUROC | Avg Cost | ECE | Defer Rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for policy, vals in sorted(aggregate.items()):
        lines.append(
            "| {p} | {f1:.4f} ± {f1s:.4f} | {auc:.4f} ± {aucs:.4f} | {cost:.4f} ± {costs:.4f} | {ece:.4f} ± {eces:.4f} | {defer:.4f} ± {defers:.4f} |".format(
                p=policy,
                f1=vals["f1_mean"],
                f1s=vals["f1_std"],
                auc=vals["auroc_mean"],
                aucs=vals["auroc_std"],
                cost=vals["avg_cost_mean"],
                costs=vals["avg_cost_std"],
                ece=vals["ece_mean"],
                eces=vals["ece_std"],
                defer=vals["defer_rate_mean"],
                defers=vals["defer_rate_std"],
            )
        )

    lines.append("")
    lines.append("## Improvement Verdicts (BAO vs Baselines)")
    lines.append("")
    lines.append("| Baseline | ΔF1 mean | ΔCost mean | ΔF1 CI95 | ΔCost CI95 | Primary | Secondary | CI Gate | Improved |")
    lines.append("|---|---:|---:|---|---|---|---|---|---|")
    for baseline, v in sorted(verdicts.items()):
        lines.append(
            "| {b} | {df1:.4f} | {dc:.4f} | [{f1l:.4f}, {f1h:.4f}] | [{cl:.4f}, {ch:.4f}] | {p} | {s} | {cg} | {imp} |".format(
                b=baseline,
                df1=v["delta_f1_mean"],
                dc=v["delta_cost_mean"],
                f1l=v["delta_f1_ci95"][0],
                f1h=v["delta_f1_ci95"][1],
                cl=v["delta_cost_ci95"][0],
                ch=v["delta_cost_ci95"][1],
                p="yes" if (v["primary"] or v["primary_alt"]) else "no",
                s="yes" if v["secondary"] else "no",
                cg="yes" if v["ci_gate"] else "no",
                imp="yes" if v["improved"] else "no",
            )
        )

    out_path.write_text("\n".join(lines))


def _write_plots(out_dir: Path, results: List[PolicyResult]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_per = []
    for r in results:
        m = compute_policy_metrics(r)
        m["policy"] = r.policy_name
        m["seed"] = r.seed
        metrics_per.append(m)

    policies = sorted({m["policy"] for m in metrics_per})
    means = {}
    for p in policies:
        subset = [m for m in metrics_per if m["policy"] == p]
        means[p] = {
            "f1": float(np.mean([m["f1"] for m in subset])),
            "cost": float(np.mean([m["avg_cost"] for m in subset])),
            "ece": float(np.mean([m["ece"] for m in subset])),
            "defer": float(np.mean([m["defer_rate"] for m in subset])),
            "recall": float(np.mean([m["recall"] for m in subset])),
        }

    # Cost vs F1 scatter
    plt.figure(figsize=(7, 4))
    for p in policies:
        plt.scatter(means[p]["cost"], means[p]["f1"], label=p)
    plt.xlabel("Average Cost")
    plt.ylabel("F1")
    plt.title("Cost vs F1")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "cost_vs_f1_scatter.png", dpi=160)
    plt.close()

    # Pareto frontier (cost vs F1)
    pts = sorted([(means[p]["cost"], means[p]["f1"], p) for p in policies], key=lambda x: x[0])
    frontier = []
    best_f1 = -1.0
    for c, f1, p in pts:
        if f1 > best_f1:
            frontier.append((c, f1, p))
            best_f1 = f1

    plt.figure(figsize=(7, 4))
    plt.scatter([x[0] for x in pts], [x[1] for x in pts], alpha=0.6)
    plt.plot([x[0] for x in frontier], [x[1] for x in frontier], marker="o")
    for c, f1, p in frontier:
        plt.text(c, f1, p, fontsize=8)
    plt.xlabel("Average Cost")
    plt.ylabel("F1")
    plt.title("Pareto Frontier")
    plt.tight_layout()
    plt.savefig(out_dir / "pareto_frontier.png", dpi=160)
    plt.close()

    # Calibration curves (aggregate, reliability diagram)
    plt.figure(figsize=(7, 4))
    bins = np.linspace(0, 1, 11)
    for p in policies:
        p_scores = []
        p_labels = []
        for r in results:
            if r.policy_name == p:
                p_scores.extend(r.scores)
                p_labels.extend(r.labels)
        scores = np.array(p_scores)
        labels = np.array(p_labels)
        xs = []
        ys = []
        for i in range(10):
            lo, hi = bins[i], bins[i + 1]
            mask = (scores >= lo) & (scores < hi if i < 9 else scores <= hi)
            if not np.any(mask):
                continue
            xs.append(scores[mask].mean())
            ys.append(labels[mask].mean())
        if xs:
            plt.plot(xs, ys, marker="o", label=p)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title("Calibration Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "calibration_curves.png", dpi=160)
    plt.close()

    # Deferral vs recall
    plt.figure(figsize=(7, 4))
    for p in policies:
        plt.scatter(means[p]["defer"], means[p]["recall"], label=p)
    plt.xlabel("Defer Rate")
    plt.ylabel("Recall")
    plt.title("Deferral vs Recall")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "deferral_vs_recall.png", dpi=160)
    plt.close()


def run_benchmarks(
    dataset_path: str,
    seeds: Sequence[int],
    policy_names: Optional[Sequence[str]] = None,
    output_dir: str = "artifacts/benchmark",
    config: Optional[BenchmarkConfig] = None,
) -> Dict[str, object]:
    cfg = config or BenchmarkConfig()
    rows = load_labeled_csv(dataset_path)
    if len(rows) < 5:
        raise ValueError("Dataset must contain at least 5 rows")

    calibration_rows, eval_rows = _split_calibration_eval(rows, cfg.calibration_fraction)

    all_results: List[PolicyResult] = []

    for seed in seeds:
        agents = build_default_agents()
        calibrator = fit_shared_calibrator(
            calibration_rows=[{"label": r.label, "features": r.features} for r in calibration_rows],
            agents=agents,
            seed=seed,
        )
        policies = build_policies(cfg, calibration_rows, agents, calibrator, seed)

        selected = list(policy_names) if policy_names else list(policies.keys())

        # All policies use identical replay order for this seed (temporal order preserved).
        for policy_name in selected:
            policy = policies[policy_name]
            labels = []
            scores = []
            decisions = []
            defers = []
            costs = []
            calls = []
            lats = []
            for row in eval_rows:
                out = policy.run_flow(row, agents, calibrator, seed)
                labels.append(row.label)
                scores.append(out["score"])
                decisions.append(out["decision"])
                defers.append(out["defer"])
                costs.append(out["cost"])
                calls.append(out["agent_calls"])
                lats.append(out["latency_ms"])

            all_results.append(
                PolicyResult(
                    policy_name=policy_name,
                    seed=seed,
                    labels=labels,
                    scores=scores,
                    decisions=decisions,
                    defer_flags=defers,
                    costs=costs,
                    agent_calls=calls,
                    latencies_ms=lats,
                )
            )

    aggregate = aggregate_by_policy(all_results)
    verdicts = compare_against_bao(all_results, cfg)

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    _save_results_csv(all_results, out_root / "results.csv")
    _write_summary_md(out_root / "summary.md", aggregate, verdicts)
    _write_plots(out_root / "plots", all_results)

    return {
        "results": all_results,
        "aggregate": aggregate,
        "verdicts": verdicts,
        "output_dir": str(out_root),
    }
