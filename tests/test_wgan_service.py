from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from agents.wgan_gp.service import Critic, Generator, WGANGPAgent


class WGANGPServiceTests(unittest.TestCase):
    def test_wgan_payload_load_and_infer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "wgan.pt"
            g = Generator(z_dim=4, out_dim=1, hidden_dim=8)
            c = Critic(in_dim=1, hidden_dim=8)
            payload = {
                "generator_state_dict": g.state_dict(),
                "critic_state_dict": c.state_dict(),
                "preprocessor": {
                    "numeric_cols": ["x"],
                    "categorical_cols": [],
                    "vocabularies": {},
                    "log1p_cols": [],
                    "medians": {"x": 0.0},
                    "iqrs": {"x": 1.0},
                    "iqr_floor": 1.0,
                    "clip_min": -15.0,
                    "clip_max": 15.0,
                },
                "cat_cardinalities": [],
                "model_config": {"in_dim": 1, "z_dim": 4, "hidden_dim": 8},
                "calibration": {"coef": 1.0, "intercept": 0.0},
                "threshold_probability": 0.5,
                "probability_clip": [0.001, 0.999],
                "score_stats": {
                    "mean": 0.0,
                    "std": 1.0,
                    "benign_mean": 0.0,
                    "benign_std": 1.0,
                    "mal_mean": 1.0,
                    "mal_std": 1.0,
                },
            }
            torch.save(payload, tmp)

            agent = WGANGPAgent(model_path=tmp, cost=5.0)
            out = agent.predict_with_uncertainty({"x": 0.1})
            self.assertEqual(out["agent_id"], "wgan_gp")
            self.assertEqual(out["metadata"]["model_type"], "wgan_gp")
            self.assertTrue(0.0 <= float(out["uncertainty"]["epistemic"]) <= 1.0)
            self.assertTrue(0.0 <= float(out["uncertainty"]["aleatoric"]) <= 0.6931472)


if __name__ == "__main__":
    unittest.main()
