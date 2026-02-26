from __future__ import annotations

import unittest
from types import SimpleNamespace

from main import _apply_overrides


class MainOverridesTests(unittest.TestCase):
    def test_router_profile_override_resolves_to_absolute_path(self) -> None:
        cfg = {"routing": {}}
        args = SimpleNamespace(
            seed=7,
            update_mode=None,
            engine=None,
            first_agent_strategy=None,
            agent_sequence=None,
            fusion_method=None,
            cost_calibration_json=None,
            prediction_source=None,
            utility_evaluation=None,
            reset_state=None,
            write_manifest=None,
            max_agents=None,
            query_policy=None,
            first_agent=None,
            min_expected_gain=None,
            router_profile="artifacts/replay/router_profile.json",
        )
        out = _apply_overrides(cfg, args)
        path = str(out["routing"]["profile_path"])
        self.assertTrue(path.startswith("/"))
        self.assertTrue(path.endswith("artifacts/replay/router_profile.json"))


if __name__ == "__main__":
    unittest.main()
