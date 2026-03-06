from __future__ import annotations

import unittest

from orchestrator.benchmarking import (
    MetricsAccumulator,
    attach_reference_deltas,
    build_comparison_block,
    build_metric_reference,
    compute_attack_cat_recall_gap,
)
from orchestrator.decisioning import DecisionCosts


class BenchmarkingTests(unittest.TestCase):
    def test_attack_cat_recall_gap_filters_small_groups(self) -> None:
        labels = [1] * 111
        predictions = ([1] * 60) + ([0] * 40) + ([1] * 10) + [0]
        metadata_rows = ([{"attack_cat": "DoS"}] * 60) + ([{"attack_cat": "Exploits"}] * 40) + ([{"attack_cat": "Worms"}] * 11)
        metrics = compute_attack_cat_recall_gap(
            labels=labels,
            predictions=predictions,
            metadata_rows=metadata_rows,
            min_group_size=20,
        )
        self.assertEqual(metrics["included_groups"]["DoS"]["support"], 60)
        self.assertIn("Worms", metrics["excluded_groups"])
        self.assertAlmostEqual(metrics["value"], 1.0, places=6)

    def test_metrics_accumulator_reports_held_out_calibration_metrics(self) -> None:
        acc = MetricsAccumulator(costs=DecisionCosts(c_fn=25.0, c_fp=2.0, c_h=2.0), group_min_size=2)
        rows = [
            (1, 0.9, "reject", 0.2, {"attack_cat": "DoS"}),
            (1, 0.8, "reject", 0.2, {"attack_cat": "DoS"}),
            (1, 0.2, "accept", 0.2, {"attack_cat": "Exploits"}),
            (1, 0.1, "accept", 0.2, {"attack_cat": "Exploits"}),
            (0, 0.7, "reject", 0.2, {"attack_cat": "Normal"}),
            (0, 0.1, "accept", 0.2, {"attack_cat": "Normal"}),
        ]
        for y, p, d, q, meta in rows:
            acc.add(true_label=y, probability=p, decision=d, query_cost=q, metadata=meta)
        metrics = acc.compute(approach="bao", family="bao")
        self.assertIn("ece", metrics)
        self.assertIn("brier", metrics)
        self.assertIn("group_metrics", metrics)
        self.assertGreater(metrics["ece"], 0.0)
        self.assertGreater(metrics["brier"], 0.0)
        self.assertAlmostEqual(metrics["attack_cat_recall_gap"], 1.0, places=6)

    def test_reference_deltas_and_comparison_block(self) -> None:
        thresholded = {
            "ocsvm": {
                "utility_cost_total": 100.0,
                "accuracy": 0.70,
                "ece": 0.25,
                "attack_cat_recall_gap": 0.30,
            },
            "lstm_autoencoder": {
                "utility_cost_total": 140.0,
                "accuracy": 0.82,
                "ece": 0.12,
                "attack_cat_recall_gap": 0.10,
            },
        }
        reference = build_metric_reference(thresholded)
        bao = {
            "utility_cost_total": 80.0,
            "accuracy": 0.84,
            "ece": 0.08,
            "attack_cat_recall_gap": 0.05,
        }
        attach_reference_deltas(bao, reference)
        comparison = build_comparison_block(bao_metrics=bao, thresholded_results=thresholded)
        self.assertEqual(reference["utility_cost_total"]["agent_id"], "ocsvm")
        self.assertEqual(reference["accuracy"]["agent_id"], "lstm_autoencoder")
        self.assertGreater(bao["deltas_vs_best_thresholded_single_agent"]["utility_cost_total"]["percent_gain"], 0.0)
        self.assertGreater(comparison["bao_vs_best_single_agent"]["accuracy"]["percent_gain"], 0.0)


if __name__ == "__main__":
    unittest.main()
