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
    def test_utilization_adjustment_affects_selection(self) -> None:
        router = AdaptiveRouter(decision_costs=DecisionCosts(c_fn=25.0, c_fp=1.0, c_h=100.0), seed=7)
        handles = {"b": _handle("b", 1.0), "c": _handle("c", 1.0)}
        next_agent, scores, mode = router.select_next_agent(
            current_probability=0.5,
            source_agent="a",
            candidate_agents=["b", "c"],
            agent_handles=handles,
            belief_manager=_Belief(),
            min_expected_gain=-100.0,
            utilization_adjustments={"b": 0.0, "c": -5.0},
            exploration_enabled=False,
        )
        self.assertIsNotNone(next_agent)
        self.assertEqual(next_agent, "b")
        self.assertEqual(mode, "exploit")
        self.assertAlmostEqual(
            float(scores["b"].expected_gain_adjusted),
            float(scores["b"].expected_gain),
            places=12,
        )
        self.assertLess(float(scores["c"].expected_gain_adjusted), float(scores["c"].expected_gain))
        self.assertLess(float(scores["c"].utilization_adjustment), 0.0)

    def test_seeded_exploration_is_reproducible(self) -> None:
        handles = {"b": _handle("b", 1.0), "c": _handle("c", 1.0)}
        r1 = AdaptiveRouter(decision_costs=DecisionCosts(c_fn=25.0, c_fp=1.0, c_h=100.0), seed=123)
        r2 = AdaptiveRouter(decision_costs=DecisionCosts(c_fn=25.0, c_fp=1.0, c_h=100.0), seed=123)
        seq1 = []
        seq2 = []
        for _ in range(20):
            n1, _s1, _m1 = r1.select_next_agent(
                current_probability=0.5,
                source_agent="a",
                candidate_agents=["b", "c"],
                agent_handles=handles,
                belief_manager=_Belief(),
                min_expected_gain=100.0,
                utilization_adjustments={"b": 0.5, "c": 0.5},
                exploration_enabled=True,
                exploration_base_rate=1.0,
                exploration_max_rate=1.0,
            )
            n2, _s2, _m2 = r2.select_next_agent(
                current_probability=0.5,
                source_agent="a",
                candidate_agents=["b", "c"],
                agent_handles=handles,
                belief_manager=_Belief(),
                min_expected_gain=100.0,
                utilization_adjustments={"b": 0.5, "c": 0.5},
                exploration_enabled=True,
                exploration_base_rate=1.0,
                exploration_max_rate=1.0,
            )
            seq1.append(n1)
            seq2.append(n2)
        self.assertEqual(seq1, seq2)

    def test_force_under_target_topup_can_query_below_gain_threshold(self) -> None:
        router = AdaptiveRouter(decision_costs=DecisionCosts(c_fn=25.0, c_fp=1.0, c_h=100.0), seed=1)
        handles = {"b": _handle("b", 1.0)}
        next_agent, _scores, mode = router.select_next_agent(
            current_probability=0.5,
            source_agent="a",
            candidate_agents=["b"],
            agent_handles=handles,
            belief_manager=_Belief(),
            min_expected_gain=100.0,
            utilization_adjustments={"b": 1.0},
            exploration_enabled=False,
            force_under_target_topup=True,
            uncertainty_allows_exploration=False,
        )
        self.assertEqual(next_agent, "b")
        self.assertEqual(mode, "force_topup")


if __name__ == "__main__":
    unittest.main()
