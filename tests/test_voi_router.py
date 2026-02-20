from __future__ import annotations

from bao.belief_state import BayesianBeliefState
from bao.voi_router import VOIRouter


class StubAgent:
    def __init__(self, cost: float):
        self.cost = cost


def test_voi_exact_negative_when_cost_dominates_without_model():
    agents = {"agent_a": StubAgent(cost=10.0)}
    router = VOIRouter(agents=agents, observation_models={}, c_fn=100.0, c_fp=1.0, c_h=5.0)
    belief = BayesianBeliefState(flow_id="f1")

    voi = router.estimate_voi_exact("agent_a", belief, {"packet_count": 1})
    assert voi < 0


def test_voi_surrogate_fallback_path_runs_when_untrained():
    agents = {"agent_a": StubAgent(cost=1.0)}
    router = VOIRouter(agents=agents, observation_models={}, use_surrogate=True)
    belief = BayesianBeliefState(flow_id="f1")

    voi = router.estimate_voi("agent_a", belief, {"packet_count": 10, "byte_count": 1000, "flow_duration": 1})
    assert isinstance(voi, float)
