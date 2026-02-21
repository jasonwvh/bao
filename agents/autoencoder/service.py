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

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(name)s] %(message)s')
logger = logging.getLogger("autoencoder")


class AutoencoderModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class Autoencoder:
    def __init__(self, model_path: str | Path, cost: float = 2.0):
        self.agent_id = "autoencoder"
        self.cost = float(cost)
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. Train with: python -m agents.autoencoder.train"
            )

        payload = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.feature_names = list(payload["feature_names"])
        self.feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(payload["feature_std"], dtype=np.float32)
        self.loss_mean = float(payload.get("loss_mean", 0.0))
        self.loss_std = float(payload.get("loss_std", 1.0))
        self.loss_p95 = float(payload.get("loss_p95", self.loss_mean + 1.645 * self.loss_std))

        self.model = AutoencoderModel(input_dim=len(self.feature_names))
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def _vectorize(self, flow_features: Dict[str, Any]) -> np.ndarray:
        vec = np.zeros(len(self.feature_names), dtype=np.float32)
        for i, name in enumerate(self.feature_names):
            try:
                vec[i] = float(flow_features.get(name, 0.0))
            except Exception:
                vec[i] = 0.0
        return vec

    def predict_with_uncertainty(self, flow_features: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        x_raw = self._vectorize(flow_features)
        x = (x_raw - self.feature_mean) / np.maximum(self.feature_std, 1e-6)

        x_t = torch.from_numpy(x).unsqueeze(0)
        with torch.no_grad():
            recon = self.model(x_t)
            mse = float(torch.mean((recon - x_t) ** 2).item())

        z = (mse - self.loss_mean) / max(self.loss_std, 1e-9)
        p = 1.0 / (1.0 + math.exp(-z))
        p = float(max(0.001, min(0.999, p)))
        entropy = -(p * math.log(max(p, 1e-9)) + (1 - p) * math.log(max(1 - p, 1e-9)))

        epistemic = float(max(0.0, 1.0 - min(abs(z) / 3.0, 1.0)))
        label = "malicious" if p >= 0.5 else "benign"

        return {
            "proba": [1.0 - p, p],
            "prediction": {"label": label, "probability": p},
            "uncertainty": {
                "epistemic": epistemic,
                "aleatoric": float(entropy),
                "total_entropy": float(entropy),
            },
            "cost": self.cost,
            "agent_id": self.agent_id,
            "metadata": {
                "model": "pytorch-autoencoder",
                "model_path": str(self.model_path),
                "anomaly_score": mse,
                "score_p95": self.loss_p95,
            },
        }


DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "autoencoder.pt")
AGENT = Autoencoder(
    model_path=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH),
    cost=float(os.getenv("AGENT_COST", "2.0")),
)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/a2a/health":
            return self._send({"status": "ok", "agent_id": AGENT.agent_id, "version": "v1"})
        if self.path == "/a2a/capabilities":
            return self._send(
                {
                    "agent_id": AGENT.agent_id,
                    "capabilities": ["flow_tabular", "unsw_nb15", "deep_inspection", "anomaly_score"],
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
        
        logger.info("[INPUT] flow_id=%s, features_keys=%s", 
                   payload.get("flow_id"), 
                   list(payload.get("flow_features", {}).keys()))
        
        feats = payload.get("flow_features", {})
        seed = payload.get("context", {}).get("seed")
        out = AGENT.predict_with_uncertainty(feats, seed=seed)
        
        logger.info("[OUTPUT] flow_id=%s, proba=%s, prediction=%s", 
                   payload.get("flow_id"), 
                   out.get("proba"), 
                   out.get("prediction"))
        
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


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(name)s] %(message)s')
logger = logging.getLogger("autoencoder")

if __name__ == "__main__":
    logger.info("[STARTUP] agent_id=%s, model_path=%s, cost=%.2f", 
               AGENT.agent_id, AGENT.model_path, AGENT.cost)

def main():
    port = int(os.getenv("PORT", "8082"))
    server = HTTPServer(("0.0.0.0", port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
