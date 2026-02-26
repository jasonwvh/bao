from __future__ import annotations

import math
import unittest

from agents.common.calibration import class_uncertainty_from_probability, normalize_uncertainty


class AgentUncertaintyTests(unittest.TestCase):
    def test_class_uncertainty_bounds(self) -> None:
        self.assertAlmostEqual(class_uncertainty_from_probability(0.0), 0.0, places=8)
        self.assertAlmostEqual(class_uncertainty_from_probability(1.0), 0.0, places=8)
        self.assertAlmostEqual(class_uncertainty_from_probability(0.5), 1.0, places=12)

    def test_normalize_uncertainty_enforces_entropy_semantics(self) -> None:
        out = normalize_uncertainty(epistemic=1.2, aleatoric=2.0)
        self.assertAlmostEqual(out["epistemic"], 1.0, places=12)
        self.assertAlmostEqual(out["aleatoric"], math.log(2.0), places=12)
        self.assertAlmostEqual(out["total_entropy"], math.log(2.0), places=12)

        out = normalize_uncertainty(epistemic=0.2, aleatoric=0.1)
        self.assertAlmostEqual(out["total_entropy"], max(0.1, 0.2 * math.log(2.0)), places=12)


if __name__ == "__main__":
    unittest.main()
