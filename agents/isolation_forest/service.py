from __future__ import annotations

import json
import logging
import math
import os
import pickle
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(name)s] %(message)s')
logger = logging.getLogger("isolation_forest")


class IsolationForestAgent:
    def __init__(self, model_path: str | Path, cost: float = 1.0):
        self.agent_id = "isolation_forest"
        self.cost = float(cost)
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. Train with: python -m agents.isolation_forest.train"
            )

        with open(self.model_path, "rb") as f:
            payload = pickle.load(f)

        self.model = payload["model"]
        self.feature_names = list(payload["feature_names"])
        self.score_mean = float(payload.get("score_mean", 0.0))
        self.score_std = float(payload.get("score_std", 1.0))
        self.score_p95 = float(payload.get("score_p95", self.score_mean + 1.645 * self.score_std))

    def _vectorize(self, flow_features: Dict[str, Any]) -> list[float]:
        vec = []
        for name in self.feature_names:
            try:
                vec.append(float(flow_features.get(name, 0.0)))
            except Exception:
                vec.append(0.0)
        return vec

    def predict_with_uncertainty(self, flow_features: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        x = self._vectorize(flow_features)
        x_df = pd.DataFrame([x], columns=self.feature_names)
        anomaly_score = float(-self.model.score_samples(x_df)[0])

        z = (anomaly_score - self.score_mean) / max(self.score_std, 1e-9)
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
                "model": "isolation-forest",
                "model_path": str(self.model_path),
                "anomaly_score": anomaly_score,
                "score_p95": self.score_p95,
            },
        }


DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "isolation_forest.pkl")
AGENT = IsolationForestAgent(
    model_path=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH),
    cost=float(os.getenv("AGENT_COST", "1.0")),
)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/a2a/health":
            return self._send({"status": "ok", "agent_id": AGENT.agent_id, "version": "v1"})
        if self.path == "/a2a/capabilities":
            return self._send(
                {
                    "agent_id": AGENT.agent_id,
                    "capabilities": ["flow_tabular", "unsw_nb15", "anomaly_score"],
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
logger = logging.getLogger("isolation_forest")

if __name__ == "__main__":
    logger.info("[STARTUP] agent_id=%s, model_path=%s, cost=%.2f", 
               AGENT.agent_id, AGENT.model_path, AGENT.cost)

def main():
    port = int(os.getenv("PORT", "8081"))
    server = HTTPServer(("0.0.0.0", port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
