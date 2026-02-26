from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from agents.lstm_autoencoder.service import LSTMAutoencoderAgent, SequenceLSTMAutoencoder


class LSTMServiceTests(unittest.TestCase):
    def test_sequence_payload_load_and_stream_state_affects_scores(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "lstm.pt"
            model = SequenceLSTMAutoencoder(in_dim=1, hidden_dim=4, num_layers=1, dropout=0.0)
            payload = {
                "state_dict": model.state_dict(),
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
                "model_config": {
                    "in_dim": 1,
                    "hidden_dim": 4,
                    "num_layers": 1,
                    "dropout": 0.0,
                    "window_size": 3,
                    "stride": 1,
                },
                "calibration": {"coef": 1.0, "intercept": 0.0},
                "threshold_probability": 0.5,
                "probability_clip": [0.001, 0.999],
                "loss_stats": {
                    "mean": 0.0,
                    "std": 1.0,
                    "benign_mean": 0.0,
                    "benign_std": 1.0,
                    "mal_mean": 1.0,
                    "mal_std": 1.0,
                },
                "meta": {"model_type": "sequence_lstm_autoencoder"},
            }
            torch.save(payload, tmp)

            agent = LSTMAutoencoderAgent(model_path=tmp, cost=3.0)
            out1 = agent.predict_with_uncertainty({"x": 0.1}, stream_id="s1")
            out2 = agent.predict_with_uncertainty({"x": 5.0}, stream_id="s1")
            self.assertEqual(out1["metadata"]["model_type"], "sequence_lstm_autoencoder")
            self.assertEqual(out2["metadata"]["model_type"], "sequence_lstm_autoencoder")
            self.assertNotEqual(
                float(out1["metadata"]["anomaly_score"]),
                float(out2["metadata"]["anomaly_score"]),
            )


if __name__ == "__main__":
    unittest.main()
