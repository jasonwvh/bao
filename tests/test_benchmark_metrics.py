from __future__ import annotations

import unittest

from benchmark.metrics import compute_metrics


class BenchmarkMetricsTests(unittest.TestCase):
    def test_utility_fields_are_reported(self) -> None:
        metrics = compute_metrics(
            predictions=[0, 1, 1, 0],
            labels=[0, 1, 0, 1],
            probabilities=[0.1, 0.9, 0.8, 0.2],
            query_costs=[1.0, 1.0, 1.0, 1.0],
            action_costs=[0.0, 0.0, 5.0, 500.0],
            approach="unit",
        )
        self.assertEqual(metrics["query_cost_total"], 4.0)
        self.assertEqual(metrics["action_cost_total"], 505.0)
        self.assertEqual(metrics["utility_cost_total"], 509.0)
        self.assertEqual(metrics["utility_cost_per_flow"], 127.25)


if __name__ == "__main__":
    unittest.main()
