from __future__ import annotations

import unittest

from orchestrator.decision import DecisionCosts, approximate_voi, select_expected_cost_action


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


if __name__ == "__main__":
    unittest.main()
