from __future__ import annotations

import unittest

from orchestrator.core import (
    UtilizationTarget,
    build_utilization_penalties,
    order_candidates,
    resolve_first_agent,
)
from orchestrator.types import AgentRuntimeHandle


def _handle(agent_id: str, cost: float) -> AgentRuntimeHandle:
    return AgentRuntimeHandle(
        agent_id=agent_id,
        endpoint="http://localhost",
        transport="http-json",
        timeout_ms=1000,
        cost=float(cost),
        capabilities=["flow_tabular"],
        health_path="/a2a/health",
        infer_path="/a2a/infer",
        capabilities_path="/a2a/capabilities",
        meta={},
    )


class CoreTests(unittest.TestCase):
    def test_dynamic_cheapest_first_agent(self) -> None:
        handles = {"a": _handle("a", 3.0), "b": _handle("b", 1.0), "c": _handle("c", 5.0)}
        first = resolve_first_agent(
            candidates=["a", "b", "c"],
            agent_handles=handles,
            strategy="dynamic_cheapest",
            explicit_first_agent=None,
        )
        self.assertEqual(first, "b")
        self.assertEqual(order_candidates(["a", "b", "c"], first), ["b", "a", "c"])

    def test_explicit_first_agent(self) -> None:
        handles = {"a": _handle("a", 1.0), "b": _handle("b", 0.5)}
        first = resolve_first_agent(
            candidates=["a", "b"],
            agent_handles=handles,
            strategy="explicit",
            explicit_first_agent="a",
        )
        self.assertEqual(first, "a")

    def test_utilization_penalty_warmup_and_bounds(self) -> None:
        targets = {
            "b": UtilizationTarget(
                agent_id="b",
                min_rate=0.10,
                max_rate=0.25,
                penalty_under=2.0,
                penalty_over=3.0,
            )
        }
        # During warmup there is no penalty.
        p = build_utilization_penalties(
            candidate_agents=["b"],
            agent_calls={"b": 0},
            flows_processed=100,
            targets=targets,
            warmup_flows=500,
        )
        self.assertAlmostEqual(float(p["b"]), 0.0, places=12)

        # Post-warmup under-utilization incurs penalty.
        p = build_utilization_penalties(
            candidate_agents=["b"],
            agent_calls={"b": 0},
            flows_processed=1000,
            targets=targets,
            warmup_flows=500,
        )
        self.assertGreater(float(p["b"]), 0.0)

        # Inside band gives no penalty.
        p = build_utilization_penalties(
            candidate_agents=["b"],
            agent_calls={"b": 150},
            flows_processed=1000,
            targets=targets,
            warmup_flows=500,
        )
        self.assertAlmostEqual(float(p["b"]), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
