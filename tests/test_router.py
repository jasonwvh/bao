from __future__ import annotations

import unittest

from orchestrator.decision import DecisionCosts
from orchestrator.router import AdaptiveRouter
from orchestrator.types import AgentRuntimeHandle


class _Belief:
    def get_global_reliability(self, agent_id: str) -> float:
        return 0.5


def _handle(agent_id: str, cost: float) -> AgentRuntimeHandle:
    return AgentRuntimeHandle(
        agent_id=agent_id,
        endpoint="http://localhost",
        transport="http-json",
        timeout_ms=1000,
        cost=float(cost),
        capabilities=[],
        health_path="/a2a/health",
        infer_path="/a2a/infer",
        capabilities_path="/a2a/capabilities",
        meta={},
    )


class RouterTests(unittest.TestCase):
    def test_utilization_penalty_affects_selection(self) -> None:
        router = AdaptiveRouter(decision_costs=DecisionCosts(c_fn=25.0, c_fp=1.0, c_h=100.0))
        handles = {"b": _handle("b", 1.0), "c": _handle("c", 1.0)}
        next_agent, scores = router.select_next_agent(
            current_probability=0.5,
            source_agent="a",
            candidate_agents=["b", "c"],
            agent_handles=handles,
            belief_manager=_Belief(),
            min_expected_gain=-100.0,
            utilization_penalties={"b": 0.0, "c": 5.0},
        )
        self.assertIsNotNone(next_agent)
        self.assertEqual(next_agent, "b")
        self.assertAlmostEqual(
            float(scores["b"].expected_gain_adjusted),
            float(scores["b"].expected_gain),
            places=12,
        )
        self.assertLess(float(scores["c"].expected_gain_adjusted), float(scores["c"].expected_gain))
        self.assertGreater(float(scores["c"].utilization_penalty), 0.0)


if __name__ == "__main__":
    unittest.main()
