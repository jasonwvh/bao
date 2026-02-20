from __future__ import annotations

from bao.belief_state import BayesianBeliefState


def test_belief_update_moves_probability_in_expected_direction():
    b = BayesianBeliefState(flow_id="f1")
    p0 = b.get_compromise_prob()

    b.variational_update({"proba": [0.1, 0.9]}, agent_id="agent_a", learning_rate=0.2)
    p1 = b.get_compromise_prob()
    assert p1 > p0

    b.variational_update({"proba": [0.9, 0.1]}, agent_id="agent_a", learning_rate=0.2)
    p2 = b.get_compromise_prob()
    assert p2 < p1


def test_reliability_beta_converges_with_correct_predictions():
    b = BayesianBeliefState(flow_id="f2")
    for _ in range(20):
        b.update_agent_reliability("agent_a", prediction=1, true_label=1)

    mean, _ = b.get_reliability_estimate("agent_a")
    assert mean > 0.8
