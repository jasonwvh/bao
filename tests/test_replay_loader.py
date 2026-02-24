from __future__ import annotations

import unittest

from orchestrator.data.replay import load_replay_dataset


class ReplayLoaderTests(unittest.TestCase):
    def test_categorical_fields_are_retained(self) -> None:
        rows = load_replay_dataset("data/UNSW_NB15_testing-set.csv", max_rows=3)
        self.assertGreater(len(rows), 0)
        feats = rows[0]["flow_features"]
        self.assertIn("proto", feats)
        self.assertIn("service", feats)
        self.assertIn("state", feats)
        self.assertIsInstance(feats["proto"], str)
        self.assertIsInstance(feats["service"], str)
        self.assertIsInstance(feats["state"], str)

    def test_numeric_fields_stay_numeric(self) -> None:
        rows = load_replay_dataset("data/UNSW_NB15_testing-set.csv", max_rows=3)
        feats = rows[0]["flow_features"]
        self.assertIsInstance(feats["dur"], float)
        self.assertIsInstance(feats["spkts"], float)


if __name__ == "__main__":
    unittest.main()

