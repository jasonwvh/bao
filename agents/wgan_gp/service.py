from __future__ import annotations

import json
import logging
import math
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("wgan_gp")


class Critic(nn.Module):
    def __init__(self, num_dim: int, cat_cardinalities: list[int], emb_dims: list[int] | None = None):
        super().__init__()
        self.cat_embeddings = nn.ModuleList()
        self.emb_dims = emb_dims if emb_dims is not None else []
        if emb_dims is None:
            self.emb_dims = []
            for card in cat_cardinalities:
                emb_dim = int(min(16, max(4, card // 4)))
                self.cat_embeddings.append(nn.Embedding(int(max(2, card)), emb_dim))
                self.emb_dims.append(emb_dim)
        else:
            for card, emb_dim in zip(cat_cardinalities, emb_dims):
                self.cat_embeddings.append(nn.Embedding(int(max(2, card)), emb_dim))
        
        self.num_dim = num_dim
        input_dim = num_dim + sum(self.emb_dims)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def embed_categorical(self, x_cat: torch.Tensor) -> torch.Tensor:
        if len(self.cat_embeddings) == 0:
            return torch.empty(x_cat.size(0), 0, device=x_cat.device)
        parts = []
        for i, emb in enumerate(self.cat_embeddings):
            parts.append(emb(x_cat[:, i]))
        return torch.cat(parts, dim=-1)

    def prepare_input(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        embedded_cat = self.embed_categorical(x_cat)
        return torch.cat([x_num, embedded_cat], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WGANGPAgent:
    def __init__(
        self,
        cost: float = 8.0,
        model_path: str | Path = Path(__file__).resolve().parent / "models" / "wgan_gp.pt",
    ):
        self.agent_id = "wgan_gp"
        self.cost = float(cost)
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. Train with: python3 -m agents.wgan_gp.train"
            )

        payload = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.preprocessor = dict(payload["preprocessor"])
        self.numeric_cols = list(self.preprocessor.get("numeric_cols", []))
        self.categorical_cols = list(self.preprocessor.get("categorical_cols", []))
        self.vocabularies = {k: dict(v) for k, v in self.preprocessor.get("vocabularies", {}).items()}
        self.log1p_cols = set(self.preprocessor.get("log1p_cols", []))
        self.medians = {k: float(v) for k, v in self.preprocessor.get("medians", {}).items()}
        self.iqrs = {k: float(v) for k, v in self.preprocessor.get("iqrs", {}).items()}
        self.cat_cardinalities = [int(x) for x in payload.get("cat_cardinalities", [])]
        cfg = payload.get("model_config", {})
        self.invert_score = bool(cfg.get("invert_score", False))

        num_dim = int(cfg.get("num_dim", len(self.numeric_cols)))
        emb_dims = [int(x) for x in cfg.get("emb_dims", [])]
        self.critic = Critic(num_dim=num_dim, cat_cardinalities=self.cat_cardinalities, emb_dims=emb_dims)
        self.critic.load_state_dict(payload["critic"])
        self.critic.eval()

        self.calibration = payload.get("calibration", {})
        self.score_stats = payload.get("score_stats", {})

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

    def predict_with_uncertainty(self, flow_features: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        n, c = self._transform_row(flow_features)
        x_num = torch.from_numpy(n.reshape(1, -1))
        x_cat = torch.from_numpy(c.reshape(1, -1))
        
        with torch.no_grad():
            x = self.critic.prepare_input(x_num, x_cat)
            score = float(-self.critic(x).item())
        if self.invert_score:
            score = -score

        if self.calibration and "coef" in self.calibration and "intercept" in self.calibration:
            z = float(self.calibration["coef"]) * score + float(self.calibration["intercept"])
            p = 1.0 / (1.0 + math.exp(-z))
        else:
            mean = float(self.score_stats.get("mean", 0.0))
            std = float(self.score_stats.get("std", 1.0))
            z = (score - mean) / max(std, 1e-9)
            p = 1.0 / (1.0 + math.exp(-z))

        p = float(max(0.001, min(0.999, p)))
        entropy = -(p * math.log(max(p, 1e-9)) + (1.0 - p) * math.log(max(1.0 - p, 1e-9)))
        epistemic = float(max(0.0, 1.0 - min(abs(2.0 * p - 1.0), 1.0)))

        p_obs_given_attack = self._pdf(
            score,
            float(self.score_stats.get("mal_mean", self.score_stats.get("mean", 0.0))),
            float(self.score_stats.get("mal_std", self.score_stats.get("std", 1.0))),
        )
        p_obs_given_clean = self._pdf(
            score,
            float(self.score_stats.get("benign_mean", self.score_stats.get("mean", 0.0))),
            float(self.score_stats.get("benign_std", self.score_stats.get("std", 1.0))),
        )

        return {
            "proba": [1.0 - p, p],
            "prediction": {"label": "malicious" if p >= 0.5 else "benign", "probability": p},
            "uncertainty": {
                "epistemic": epistemic,
                "aleatoric": float(entropy),
                "total_entropy": float(max(epistemic, entropy)),
            },
            "likelihoods": {
                "p_obs_given_attack": float(p_obs_given_attack),
                "p_obs_given_clean": float(p_obs_given_clean),
            },
            "cost": self.cost,
            "agent_id": self.agent_id,
            "metadata": {
                "model": "wgan_gp_anomaly",
                "model_path": str(self.model_path),
                "anomaly_score": score,
            },
        }


DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "wgan_gp.pt")
AGENT = WGANGPAgent(
    cost=float(os.getenv("AGENT_COST", "8.0")),
    model_path=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH),
)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/a2a/health":
            return self._send({"status": "ok", "agent_id": AGENT.agent_id, "version": "v1"})
        if self.path == "/a2a/capabilities":
            return self._send(
                {
                    "agent_id": AGENT.agent_id,
                    "capabilities": ["flow_tabular", "unsw_nb15", "anomaly_score", "generative_anomaly"],
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
        out = AGENT.predict_with_uncertainty(feats, seed=seed)
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
    port = int(os.getenv("PORT", "8084"))
    server = HTTPServer(("0.0.0.0", port), Handler)
    logger.info(f"Starting WGAN-GP agent server on port {port}...")
    server.serve_forever()


if __name__ == "__main__":
    main()
