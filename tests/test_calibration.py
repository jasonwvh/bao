from __future__ import annotations

import unittest

import numpy as np

from agents.common.calibration import fit_logistic_calibrator, logistic_probability, select_probability_threshold


class CalibrationTests(unittest.TestCase):
    def test_logistic_probability_is_monotonic(self) -> None:
        scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        labels = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        calibrator = fit_logistic_calibrator(scores, labels, seed=42)
        probs = np.asarray(logistic_probability(scores, calibrator))
        self.assertTrue(np.all(np.diff(probs) >= 0.0))

    def test_threshold_selection_is_deterministic(self) -> None:
        probs = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9], dtype=np.float64)
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        thr1, score1 = select_probability_threshold(probs, labels)
        thr2, score2 = select_probability_threshold(probs, labels)
        self.assertEqual(thr1, thr2)
        self.assertEqual(score1, score2)


if __name__ == "__main__":
    unittest.main()

