from __future__ import annotations

import math
import unittest

from orchestrator.belief_state import BayesianBeliefState


class BeliefStateTests(unittest.TestCase):
    def test_posterior_first_matches_first_agent_exactly(self) -> None:
        belief = BayesianBeliefState(flow_id="f1")
        result = belief.update_from_agent_output(
            agent_output={"proba": [0.2, 0.8]},
            agent_id="a1",
            update_mode="posterior_first",
            weight=1.0,
            eps=1e-6,
            likelihood_sanity_gate=True,
        )
        self.assertAlmostEqual(result["compromise_prob"], 0.8, places=12)

    def test_logit_pool_two_agents_is_deterministic(self) -> None:
        belief = BayesianBeliefState(flow_id="f2")
        belief.update_from_agent_output(
            agent_output={"proba": [0.2, 0.8]},
            agent_id="a1",
            update_mode="posterior_first",
            weight=1.0,
            eps=1e-6,
            likelihood_sanity_gate=True,
        )
        result = belief.update_from_agent_output(
            agent_output={"proba": [0.8, 0.2]},
            agent_id="a2",
            update_mode="posterior_first",
            weight=1.0,
            eps=1e-6,
            likelihood_sanity_gate=True,
        )
        self.assertAlmostEqual(result["compromise_prob"], 0.5, places=12)

    def test_likelihood_mode_reproduces_analytic_update(self) -> None:
        belief = BayesianBeliefState(flow_id="f3")
        belief.set_compromise_prob(0.2)

        result = belief.update_from_agent_output(
            agent_output={
                "likelihoods": {
                    "p_obs_given_attack": 0.8,
                    "p_obs_given_clean": 0.2,
                }
            },
            agent_id="a1",
            update_mode="likelihood_strict",
            weight=1.0,
            eps=1e-9,
            likelihood_sanity_gate=False,
        )
        # Prior odds = 0.2/0.8 = 0.25; LR = 4 => posterior odds = 1 => p=0.5
        self.assertAlmostEqual(result["compromise_prob"], 0.5, places=12)

    def test_likelihood_sanity_gate_falls_back_to_posterior(self) -> None:
        belief = BayesianBeliefState(flow_id="f4")
        result = belief.update_from_agent_output(
            agent_output={
                "proba": [0.1, 0.9],
                "likelihoods": {
                    "p_obs_given_attack": 0.1,
                    "p_obs_given_clean": 0.9,
                },
            },
            agent_id="a1",
            update_mode="likelihood_strict",
            weight=1.0,
            eps=1e-6,
            likelihood_sanity_gate=True,
        )

        self.assertAlmostEqual(result["compromise_prob"], 0.9, places=12)


if __name__ == "__main__":
    unittest.main()
