from __future__ import annotations

import unittest

import numpy as np

from agents.common.preprocessing import fit_preprocessor, load_csv, transform_frame


class PreprocessingTests(unittest.TestCase):
    def test_scaled_features_are_clipped_and_bounded(self) -> None:
        df = load_csv("data/UNSW_NB15_training-set.csv").head(2000)
        pre = fit_preprocessor(df, iqr_floor=1.0, clip_min=-15.0, clip_max=15.0)
        num, _ = transform_frame(pre, df)

        self.assertTrue(np.all(np.isfinite(num)))
        self.assertLessEqual(float(np.max(num)), 15.0)
        self.assertGreaterEqual(float(np.min(num)), -15.0)

    def test_preprocessor_roundtrip_preserves_clip_config(self) -> None:
        df = load_csv("data/UNSW_NB15_training-set.csv").head(100)
        pre = fit_preprocessor(df, iqr_floor=1.0, clip_min=-15.0, clip_max=15.0)
        loaded = type(pre).from_dict(pre.to_dict())
        self.assertEqual(float(loaded.iqr_floor), 1.0)
        self.assertEqual(float(loaded.clip_min), -15.0)
        self.assertEqual(float(loaded.clip_max), 15.0)


if __name__ == "__main__":
    unittest.main()

