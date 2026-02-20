from __future__ import annotations

import csv
from pathlib import Path

from bao.benchmark.runner import run_benchmarks
from bao.benchmark.types import BenchmarkConfig


def _write_dataset(path: Path, n: int = 40) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["flow_id", "packet_count", "byte_count", "flow_duration", "src_port", "dst_port", "label"],
        )
        writer.writeheader()
        for i in range(n):
            writer.writerow(
                {
                    "flow_id": f"flow_{i}",
                    "packet_count": 10 + i * 3,
                    "byte_count": 1200 + i * 60,
                    "flow_duration": 1 + i * 0.4,
                    "src_port": 1000 + i,
                    "dst_port": 22 if i % 5 == 0 else 80,
                    "label": 1 if i % 4 == 0 else 0,
                }
            )


def test_runner_uses_identical_replay_order_and_has_cost_consistency(tmp_path: Path):
    ds = tmp_path / "data.csv"
    out_dir = tmp_path / "artifacts"
    _write_dataset(ds)

    res = run_benchmarks(
        dataset_path=str(ds),
        seeds=[1],
        output_dir=str(out_dir),
        config=BenchmarkConfig(),
    )

    results = res["results"]
    by_policy = {r.policy_name: r for r in results}

    # Identical replay order: labels are aligned exactly across policies.
    labels_ref = by_policy["bao"].labels
    for name, r in by_policy.items():
        assert r.labels == labels_ref, name

    # Cost consistency invariants.
    q_all_costs = by_policy["baseline_query_all"].costs
    assert all(abs(c - 6.0) < 1e-9 for c in q_all_costs)

    single_costs = by_policy["baseline_single_cheapest"].costs
    assert all(abs(c - 1.0) < 1e-9 for c in single_costs)


def test_benchmark_cli_artifacts_exist(tmp_path: Path):
    ds = tmp_path / "data.csv"
    out_dir = tmp_path / "bench_out"
    _write_dataset(ds, n=50)

    run_benchmarks(
        dataset_path=str(ds),
        seeds=[1, 2],
        output_dir=str(out_dir),
        config=BenchmarkConfig(bootstrap_samples=200),
    )

    assert (out_dir / "results.csv").exists()
    assert (out_dir / "summary.md").exists()
    # Plot files may be skipped only if matplotlib is unavailable.
    # Since pyproject requires it, assert key plot existence.
    assert (out_dir / "plots" / "cost_vs_f1_scatter.png").exists()


def test_regression_shared_config_source(tmp_path: Path):
    ds = tmp_path / "data.csv"
    out_dir = tmp_path / "artifacts"
    _write_dataset(ds)

    cfg = BenchmarkConfig(p_accept=0.25, p_reject=0.8)
    res = run_benchmarks(dataset_path=str(ds), seeds=[3], output_dir=str(out_dir), config=cfg)

    # Regression guard: BAO exists and every baseline exists under same run config.
    policies = {r.policy_name for r in res["results"]}
    assert {
        "bao",
        "baseline_query_all",
        "baseline_fixed_cascade",
        "baseline_confidence_escalation",
        "baseline_single_cheapest",
    }.issubset(policies)
