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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("ocsvm")


class OCSVMAgent:
    def __init__(self, model_path: str | Path, cost: float = 2.5):
        self.agent_id = "ocsvm"
        self.cost = float(cost)
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. Train with: python3 -m agents.ocsvm.train"
            )

        with open(self.model_path, "rb") as f:
            payload = pickle.load(f)

        self.model = payload["model"]
        self.calibrator = payload.get("calibrator")
        self.preprocessor = dict(payload["preprocessor"])
        self.numeric_cols = list(self.preprocessor.get("numeric_cols", []))
        self.categorical_cols = list(self.preprocessor.get("categorical_cols", []))
        self.vocabularies = {k: dict(v) for k, v in self.preprocessor.get("vocabularies", {}).items()}
        self.log1p_cols = set(self.preprocessor.get("log1p_cols", []))
        self.medians = {k: float(v) for k, v in self.preprocessor.get("medians", {}).items()}
        self.iqrs = {k: float(v) for k, v in self.preprocessor.get("iqrs", {}).items()}
        self.cat_cardinalities = [int(x) for x in payload.get("cat_cardinalities", [])]
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

    def _vectorize(self, flow_features: Dict[str, Any]) -> np.ndarray:
        n, c = self._transform_row(flow_features)
        if len(self.cat_cardinalities) == 0:
            return n.astype(np.float32)

        total_cat_dim = int(sum(self.cat_cardinalities))
        cat_oh = np.zeros(total_cat_dim, dtype=np.float32)
        offset = 0
        for i, card in enumerate(self.cat_cardinalities):
            idx = int(np.clip(c[i], 0, max(card - 1, 0)))
            cat_oh[offset + idx] = 1.0
            offset += card

        return np.concatenate([n.astype(np.float32), cat_oh], axis=0)

    def _pdf(self, x: float, mu: float, sigma: float) -> float:
        s = max(float(sigma), 1e-6)
        coeff = 1.0 / (s * math.sqrt(2.0 * math.pi))
        return float(max(1e-9, coeff * math.exp(-((x - mu) ** 2) / (2.0 * s * s))))

    def predict_with_uncertainty(self, flow_features: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        x = self._vectorize(flow_features).reshape(1, -1)
        anomaly_score = float(-self.model.decision_function(x)[0])

        if self.calibrator is not None:
            p = float(self.calibrator.predict_proba([[anomaly_score]])[0, 1])
        else:
            mean = float(self.score_stats.get("mean", 0.0))
            std = float(self.score_stats.get("std", 1.0))
            z = (anomaly_score - mean) / max(std, 1e-9)
            p = 1.0 / (1.0 + math.exp(-z))

        p = float(max(0.001, min(0.999, p)))
        entropy = -(p * math.log(max(p, 1e-9)) + (1 - p) * math.log(max(1 - p, 1e-9)))
        epistemic = float(max(0.0, 1.0 - min(abs(2.0 * p - 1.0), 1.0)))

        p_obs_given_attack = self._pdf(
            anomaly_score,
            float(self.score_stats.get("mal_mean", self.score_stats.get("mean", 0.0))),
            float(self.score_stats.get("mal_std", self.score_stats.get("std", 1.0))),
        )
        p_obs_given_clean = self._pdf(
            anomaly_score,
            float(self.score_stats.get("benign_mean", self.score_stats.get("mean", 0.0))),
            float(self.score_stats.get("benign_std", self.score_stats.get("std", 1.0))),
        )

        label = "malicious" if p >= 0.5 else "benign"
        return {
            "proba": [1.0 - p, p],
            "prediction": {"label": label, "probability": p},
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
                "model": "ocsvm",
                "model_path": str(self.model_path),
                "anomaly_score": anomaly_score,
            },
        }


DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ocsvm.pkl")
AGENT = OCSVMAgent(
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
                    "capabilities": ["flow_tabular", "unsw_nb15", "anomaly_score", "one_class"],
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
    port = int(os.getenv("PORT", "8081"))
    server = HTTPServer(("0.0.0.0", port), Handler)
    logger.info(f"Starting One-Class SVM agent server on port {port}...")
    server.serve_forever()


if __name__ == "__main__":
    main()
