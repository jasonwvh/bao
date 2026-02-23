from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("lstm_autoencoder")


class HybridLSTMAutoencoder(nn.Module):
    def __init__(self, num_dim: int, cat_cardinalities: list[int], hidden_dim: int = 64):
        super().__init__()
        self.cat_embeddings = nn.ModuleList()
        for card in cat_cardinalities:
            emb_dim = int(min(16, max(4, card // 4)))
            self.cat_embeddings.append(nn.Embedding(int(max(2, card)), emb_dim))
        input_dim = num_dim + sum(int(e.embedding_dim) for e in self.cat_embeddings)
        self.num_dim = num_dim
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, num_dim)

    def _embed(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        parts = [x_num]
        for i, emb in enumerate(self.cat_embeddings):
            parts.append(emb(x_cat[:, :, i]))
        return torch.cat(parts, dim=-1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self._embed(x_num, x_cat)
        _, (h_n, _) = self.encoder(embedded)
        seq_len = embedded.shape[1]
        dec_in = h_n[-1].unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder(dec_in)
        recon = self.out(dec_out)
        return recon, x_num


class LSTMAutoencoderAgent:
    def __init__(self, model_path: str | Path, cost: float = 1.0):
        self.agent_id = "lstm_autoencoder"
        self.cost = float(cost)
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. Train with: python3 -m agents.lstm_autoencoder.train"
            )

        payload = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.preprocessor = dict(payload["preprocessor"])
        self.numeric_cols = list(self.preprocessor.get("numeric_cols", []))
        self.categorical_cols = list(self.preprocessor.get("categorical_cols", []))
        self.vocabularies = {k: dict(v) for k, v in self.preprocessor.get("vocabularies", {}).items()}
        self.log1p_cols = set(self.preprocessor.get("log1p_cols", []))
        self.medians = {k: float(v) for k, v in self.preprocessor.get("medians", {}).items()}
        self.iqrs = {k: float(v) for k, v in self.preprocessor.get("iqrs", {}).items()}
        cfg = payload["model_config"]
        self.window_size = int(cfg.get("window_size", 8))

        self.model = HybridLSTMAutoencoder(
            num_dim=int(cfg["num_dim"]),
            cat_cardinalities=[int(x) for x in cfg["cat_cardinalities"]],
            hidden_dim=int(cfg.get("hidden_dim", 64)),
        )
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

        self.loss_stats = payload.get("loss_stats", {})
        self.invert_score = bool(self.loss_stats.get("invert_score", False))
        self.score_mapping = payload.get("score_mapping", {})
        self.calibration = payload.get("calibration", {})
        self.histories: Dict[str, Deque[Tuple[np.ndarray, np.ndarray]]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _transform_row(self, flow_features: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        n = np.zeros(len(self.numeric_cols), dtype=np.float32)
        c = np.zeros(len(self.categorical_cols), dtype=np.int64)

        for i, col in enumerate(self.numeric_cols):
            pp_key = f"pp_num_{col}"
            if pp_key in flow_features:
                n[i] = self._to_float(flow_features.get(pp_key), 0.0)
                continue
            x = self._to_float(flow_features.get(col, 0.0), 0.0)
            if col in self.log1p_cols and x >= 0.0:
                x = float(np.log1p(x))
            x = (x - self.medians.get(col, 0.0)) / max(self.iqrs.get(col, 1.0), 1e-6)
            n[i] = x

        for i, col in enumerate(self.categorical_cols):
            pp_key = f"pp_cat_{col}"
            if pp_key in flow_features:
                c[i] = int(self._to_float(flow_features.get(pp_key), 0.0))
                continue
            vocab = self.vocabularies.get(col, {"<UNK>": 0})
            token = str(flow_features.get(col, "<UNK>")).strip()
            c[i] = int(vocab.get(token, 0))

        return n, c

    def _pdf(self, x: float, mu: float, sigma: float) -> float:
        s = max(float(sigma), 1e-6)
        coeff = 1.0 / (s * math.sqrt(2.0 * math.pi))
        return float(max(1e-9, coeff * math.exp(-((x - mu) ** 2) / (2.0 * s * s))))

    def _score_to_prob(self, mse: float) -> float:
        score = -float(mse) if self.invert_score else float(mse)
        use_calibration = (
            self.calibration
            and "coef" in self.calibration
            and "intercept" in self.calibration
            and abs(float(self.calibration.get("coef", 0.0))) > 1e-6
        )
        if use_calibration:
            z = float(self.calibration["coef"]) * score + float(self.calibration["intercept"])
        else:
            threshold = float(self.score_mapping.get("threshold", 0.0))
            scale = float(self.score_mapping.get("scale", 1.0))
            z = (score - threshold) / max(scale * 0.5, 1e-9)
        p = 1.0 / (1.0 + math.exp(-z))
        return float(min(0.999, max(0.001, p)))

    def _build_sequence(self, flow_features: Dict[str, Any], flow_id: Optional[str]) -> tuple[np.ndarray, np.ndarray]:
        n, c = self._transform_row(flow_features)
        fid = str(flow_id or "__default__")
        hist = self.histories[fid]
        hist.append((n, c))

        items = list(hist)
        if len(items) < self.window_size:
            pad = [items[0]] * (self.window_size - len(items))
            items = pad + items

        seq_num = np.stack([x[0] for x in items[-self.window_size :]], axis=0)
        seq_cat = np.stack([x[1] for x in items[-self.window_size :]], axis=0)
        return seq_num, seq_cat

    def predict_with_uncertainty(
        self,
        flow_features: Dict[str, Any],
        seed: Optional[int] = None,
        flow_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        seq_num, seq_cat = self._build_sequence(flow_features, flow_id=flow_id)
        x_num_t = torch.from_numpy(seq_num).unsqueeze(0)
        x_cat_t = torch.from_numpy(seq_cat).unsqueeze(0)

        with torch.no_grad():
            recon, target = self.model(x_num_t, x_cat_t)
            mse = float(torch.mean((recon - target) ** 2).item())

        p = self._score_to_prob(mse)
        entropy = -(p * math.log(max(p, 1e-9)) + (1.0 - p) * math.log(max(1.0 - p, 1e-9)))
        epistemic = float(max(0.0, 1.0 - min(abs(2.0 * p - 1.0), 1.0)))

        p_obs_given_attack = self._pdf(
            mse,
            float(self.loss_stats.get("mal_mean", self.loss_stats.get("mean", 0.0))),
            float(self.loss_stats.get("mal_std", self.loss_stats.get("std", 1.0))),
        )
        p_obs_given_clean = self._pdf(
            mse,
            float(self.loss_stats.get("benign_mean", self.loss_stats.get("mean", 0.0))),
            float(self.loss_stats.get("benign_std", self.loss_stats.get("std", 1.0))),
        )

        label = "malicious" if p >= 0.5 else "benign"
        return {
            "proba": [1.0 - p, p],
            "prediction": {"label": label, "probability": p},
            "uncertainty": {
                "epistemic": epistemic,
                "aleatoric": float(entropy),
                "total_entropy": float(max(entropy, epistemic)),
            },
            "likelihoods": {
                "p_obs_given_attack": float(p_obs_given_attack),
                "p_obs_given_clean": float(p_obs_given_clean),
            },
            "cost": self.cost,
            "agent_id": self.agent_id,
            "metadata": {
                "model": "lstm_autoencoder",
                "model_path": str(self.model_path),
                "anomaly_score": mse,
                "window_size": self.window_size,
            },
        }


DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "lstm_autoencoder.pt")
AGENT = LSTMAutoencoderAgent(
    model_path=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH),
    cost=float(os.getenv("AGENT_COST", "3.0")),
)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/a2a/health":
            return self._send({"status": "ok", "agent_id": AGENT.agent_id, "version": "v1"})
        if self.path == "/a2a/capabilities":
            return self._send(
                {
                    "agent_id": AGENT.agent_id,
                    "capabilities": ["flow_tabular", "unsw_nb15", "deep_inspection", "anomaly_score", "temporal"],
                    "cost": AGENT.cost,
                }
            )
        self.send_error(404)

    def do_POST(self):
        if self.path != "/a2a/infer":
            return self.send_error(404)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body.decode("utf-8") or "{}")

        feats = payload.get("flow_features", {})
        seed = payload.get("context", {}).get("seed")
        out = AGENT.predict_with_uncertainty(feats, seed=seed, flow_id=payload.get("flow_id"))
        return self._send(out)

    def _send(self, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        return


def main():
    port = int(os.getenv("PORT", "8082"))
    server = HTTPServer(("0.0.0.0", port), Handler)
    logger.info(f"Starting LSTM-autoencoder agent server on port {port}...")
    server.serve_forever()


if __name__ == "__main__":
    main()
