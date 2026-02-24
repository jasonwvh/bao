from __future__ import annotations

import unittest

from orchestrator.decision import (
    DecisionCosts,
    approximate_voi,
    min_expected_action_cost,
    realized_action_cost,
    select_expected_cost_action,
)


class DecisionPolicyTests(unittest.TestCase):
    def test_expected_cost_accept_for_low_risk(self) -> None:
        costs = DecisionCosts(c_fn=500.0, c_fp=5.0, c_h=5000.0)
        action, _ = select_expected_cost_action(0.001, costs)
        self.assertEqual(action, "accept")

    def test_expected_cost_reject_for_high_risk(self) -> None:
        costs = DecisionCosts(c_fn=500.0, c_fp=5.0, c_h=5000.0)
        action, _ = select_expected_cost_action(0.8, costs)
        self.assertEqual(action, "reject")

    def test_expected_cost_can_defer(self) -> None:
        costs = DecisionCosts(c_fn=500.0, c_fp=5.0, c_h=0.1)
        action, _ = select_expected_cost_action(0.5, costs)
        self.assertEqual(action, "defer")

    def test_voi_threshold_logic_inputs(self) -> None:
        costs = DecisionCosts(c_fn=500.0, c_fp=5.0, c_h=5000.0)
        voi = approximate_voi(0.5, costs, rho=0.7)
        self.assertGreater(voi, 1.0)
        self.assertLess(voi, 2.0)

    def test_min_expected_action_cost(self) -> None:
        costs = DecisionCosts(c_fn=100.0, c_fp=10.0, c_h=1000.0)
        self.assertAlmostEqual(min_expected_action_cost(0.1, costs), 9.0, places=12)

    def test_realized_action_cost_for_prediction(self) -> None:
        costs = DecisionCosts(c_fn=100.0, c_fp=10.0, c_h=1000.0)
        c_fp = realized_action_cost(decision=None, prediction=1, true_label=0, costs=costs)
        c_fn = realized_action_cost(decision=None, prediction=0, true_label=1, costs=costs)
        self.assertAlmostEqual(c_fp, 10.0, places=12)
        self.assertAlmostEqual(c_fn, 100.0, places=12)


if __name__ == "__main__":
    unittest.main()
