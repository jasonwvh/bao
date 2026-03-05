from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from torch import nn

from agents.common.calibration import (
    align_probability_threshold,
    class_uncertainty_from_probability,
    entropy_from_probability,
    logistic_probability,
    normalize_uncertainty,
)
from agents.common.versioning import collect_library_versions, compare_versions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("wgan_gp")


class InferContext(BaseModel):
    belief: Dict[str, float] = Field(default_factory=dict)
    requested_capabilities: list[str] = Field(default_factory=list)
    seed: Optional[int] = None
    stream_id: Optional[str] = None
    session_id: Optional[str] = None
    elicit_likelihood: Optional[bool] = True


class InferRequest(BaseModel):
    request_id: str
    flow_id: str
    timestamp: float
    flow_features: Dict[str, Any]
    context: InferContext = Field(default_factory=InferContext)


class LegacyTabularAutoencoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class Generator(nn.Module):
    def __init__(self, z_dim: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(z_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim) * 2),
            nn.ReLU(),
            nn.Linear(int(hidden_dim) * 2, int(out_dim)),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim) * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(hidden_dim), 1),
        )

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
        self.clip_min = float(self.preprocessor.get("clip_min", -15.0))
        self.clip_max = float(self.preprocessor.get("clip_max", 15.0))
        self.cat_cardinalities = [int(x) for x in payload.get("cat_cardinalities", [])]

        cfg = payload.get("model_config", {})
        in_dim = int(cfg.get("in_dim"))
        if "critic_state_dict" in payload:
            self.generator = Generator(
                z_dim=int(cfg.get("z_dim", 32)),
                out_dim=in_dim,
                hidden_dim=int(cfg.get("hidden_dim", 128)),
            )
            self.critic = Critic(in_dim=in_dim, hidden_dim=int(cfg.get("hidden_dim", 128)))
            self.generator.load_state_dict(payload["generator_state_dict"])
            self.critic.load_state_dict(payload["critic_state_dict"])
            self.generator.eval()
            self.critic.eval()
            self.legacy_model = None
            self.model_type = "wgan_gp"
        else:
            self.legacy_model = LegacyTabularAutoencoder(
                in_dim=in_dim,
                latent_dim=int(cfg.get("latent_dim", 64)),
                dropout=float(cfg.get("dropout", 0.1)),
            )
            self.legacy_model.load_state_dict(payload["state_dict"])
            self.legacy_model.eval()
            self.generator = None
            self.critic = None
            self.model_type = "legacy_tabular_autoencoder"

        self.calibration = payload.get("calibration", {})
        self.threshold_probability = float(payload.get("threshold_probability", 0.5))
        self.probability_clip = payload.get("probability_clip", [0.001, 0.999])
        self.p_lo = float(self.probability_clip[0])
        self.p_hi = float(self.probability_clip[1])
        self.score_stats = payload.get("score_stats", {})
        self.model_meta = dict(payload.get("meta") or {})
        self.model_library_versions = dict(self.model_meta.get("library_versions") or {})
        self.runtime_library_versions = collect_library_versions()
        versions_match, version_mismatches = compare_versions(
            expected=self.model_library_versions,
            actual=self.runtime_library_versions,
        )
        self.version_check = {
            "versions_match": bool(versions_match),
            "mismatches": list(version_mismatches),
        }
        if not versions_match:
            logger.warning("Model/service version mismatch detected: %s", "; ".join(version_mismatches))

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
            x = float(np.clip(x, self.clip_min, self.clip_max))
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

    def _vectorize(self, n: np.ndarray, c: np.ndarray) -> np.ndarray:
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

    def _score_to_prob(self, score: float) -> float:
        p_raw = float(
            logistic_probability(
                score,
                self.calibration,
                clip_lo=self.p_lo,
                clip_hi=self.p_hi,
            )
        )
        return align_probability_threshold(p_raw, self.threshold_probability)

    def _score(self, vector: np.ndarray) -> float:
        x = torch.from_numpy(vector.reshape(1, -1))
        if self.critic is not None:
            with torch.no_grad():
                return float((-self.critic(x).squeeze(1)).item())
        with torch.no_grad():
            recon = self.legacy_model(x)
            return float(torch.mean((recon - x) ** 2).item())

    def predict_with_uncertainty(self, flow_features: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        del seed
        n, c = self._transform_row(flow_features)
        vector = self._vectorize(n, c)
        score = self._score(vector)

        p = self._score_to_prob(score)
        entropy = entropy_from_probability(p)
        epistemic = class_uncertainty_from_probability(p)
        uncertainty = normalize_uncertainty(epistemic=epistemic, aleatoric=entropy)

        raw_attack = self._pdf(
            score,
            float(self.score_stats.get("mal_mean", self.score_stats.get("mean", 0.0))),
            float(self.score_stats.get("mal_std", self.score_stats.get("std", 1.0))),
        )
        raw_clean = self._pdf(
            score,
            float(self.score_stats.get("benign_mean", self.score_stats.get("mean", 0.0))),
            float(self.score_stats.get("benign_std", self.score_stats.get("std", 1.0))),
        )
        raw_sum = max(1e-9, raw_attack + raw_clean)
        score_attack = raw_attack / raw_sum
        score_clean = raw_clean / raw_sum

        p_obs_given_attack = max(1e-9, 0.7 * p + 0.3 * score_attack)
        p_obs_given_clean = max(1e-9, 0.7 * (1.0 - p) + 0.3 * score_clean)

        return {
            "proba": [1.0 - p, p],
            "prediction": {
                "label": "malicious" if p >= 0.5 else "benign",
                "probability": p,
            },
            "uncertainty": uncertainty,
            "likelihoods": {
                "p_obs_given_attack": float(p_obs_given_attack),
                "p_obs_given_clean": float(p_obs_given_clean),
            },
            "cost": self.cost,
            "agent_id": self.agent_id,
            "metadata": {
                "model": "wgan_gp",
                "model_type": self.model_type,
                "model_path": str(self.model_path),
                "anomaly_score": score,
                "threshold_probability": self.threshold_probability,
                "threshold_aligned": 0.5,
                "raw_score_likelihood_attack": raw_attack,
                "raw_score_likelihood_clean": raw_clean,
            },
        }


DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "wgan_gp.pt")
AGENT = WGANGPAgent(
    cost=float(os.getenv("AGENT_COST", "8.0")),
    model_path=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH),
)

app = FastAPI(title="wgan-gp-agent", version="v1")


@app.get("/a2a/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "agent_id": AGENT.agent_id, "version": "v1"}


@app.get("/a2a/capabilities")
def capabilities() -> Dict[str, Any]:
    return {
        "agent_id": AGENT.agent_id,
        "capabilities": ["flow_tabular", "unsw_nb15", "anomaly_score", "generative_anomaly"],
        "cost": AGENT.cost,
        "metadata": {
            "model_type": AGENT.model_type,
            "runtime_library_versions": AGENT.runtime_library_versions,
            "model_library_versions": AGENT.model_library_versions,
            "version_check": AGENT.version_check,
        },
    }


@app.post("/a2a/infer")
def infer(req: InferRequest) -> Dict[str, Any]:
    return AGENT.predict_with_uncertainty(req.flow_features, seed=req.context.seed)


def main() -> None:
    port = int(os.getenv("PORT", "8084"))
    logger.info("Starting WGAN-GP agent server on port %s", port)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
