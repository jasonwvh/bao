from __future__ import annotations

import logging
import math
import os
from collections import OrderedDict, deque
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
from agents.common.likelihoods import lookup_likelihoods, validate_likelihood_model
from agents.common.streaming import derive_stream_id
from agents.common.versioning import collect_library_versions, compare_versions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("lstm_autoencoder")


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


class SequenceLSTMAutoencoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        recurrent_dropout = float(dropout if int(num_layers) > 1 else 0.0)
        self.encoder = nn.LSTM(
            input_size=int(in_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.decoder = nn.LSTM(
            input_size=int(hidden_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.out_proj = nn.Linear(int(hidden_dim), int(in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = int(x.shape[1])
        _, (h_n, _) = self.encoder(x)
        latent = h_n[-1]
        repeated = latent.unsqueeze(1).repeat(1, seq_len, 1)
        dec, _ = self.decoder(repeated)
        return self.out_proj(dec)


class StreamSequenceBuffer:
    def __init__(self, window_size: int, max_streams: int = 2048):
        self.window_size = max(1, int(window_size))
        self.max_streams = max(1, int(max_streams))
        self._buffers: OrderedDict[str, deque[np.ndarray]] = OrderedDict()

    def append_and_get(self, stream_id: str, vector: np.ndarray) -> np.ndarray:
        sid = str(stream_id or "__global_stream__")
        if sid not in self._buffers:
            self._buffers[sid] = deque(maxlen=self.window_size)
        self._buffers.move_to_end(sid)
        self._buffers[sid].append(vector.astype(np.float32))
        while len(self._buffers) > self.max_streams:
            self._buffers.popitem(last=False)

        seq = list(self._buffers[sid])
        if len(seq) < self.window_size:
            pad = [seq[0]] * (self.window_size - len(seq))
            seq = pad + seq
        else:
            seq = seq[-self.window_size :]
        return np.stack(seq, axis=0).astype(np.float32)


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
        self.clip_min = float(self.preprocessor.get("clip_min", -15.0))
        self.clip_max = float(self.preprocessor.get("clip_max", 15.0))

        self.cat_cardinalities = [int(x) for x in payload.get("cat_cardinalities", [])]
        cfg = payload.get("model_config", {})
        self.window_size = max(1, int(cfg.get("window_size", 1)))
        self.stream_buffers = StreamSequenceBuffer(window_size=self.window_size)

        in_dim = int(cfg.get("in_dim"))
        self.model = SequenceLSTMAutoencoder(
            in_dim=in_dim,
            hidden_dim=int(cfg.get("hidden_dim", cfg.get("latent_dim", 64))),
            num_layers=int(cfg.get("num_layers", 1)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
        self.model_type = "sequence_lstm_autoencoder"
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

        self.loss_stats = payload.get("loss_stats", {})
        self.likelihood_model = validate_likelihood_model(payload.get("likelihood_model"))
        self.calibration = payload.get("calibration", {})
        self.threshold_probability = float(payload.get("threshold_probability", 0.5))
        self.probability_clip = payload.get("probability_clip", [0.001, 0.999])
        self.p_lo = float(self.probability_clip[0])
        self.p_hi = float(self.probability_clip[1])
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

    def _score_flow(self, vector: np.ndarray, stream_id: str) -> float:
        seq = self.stream_buffers.append_and_get(stream_id, vector)
        x = torch.from_numpy(seq.reshape(1, self.window_size, -1))
        with torch.no_grad():
            recon = self.model(x)
            return float(torch.mean((recon - x) ** 2).item())

    def predict_with_uncertainty(
        self,
        flow_features: Dict[str, Any],
        seed: Optional[int] = None,
        flow_id: Optional[str] = None,
        stream_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        del seed
        sid = derive_stream_id(
            flow_features=flow_features,
            flow_id=flow_id,
            explicit_stream_id=stream_id,
        )
        if session_id:
            sid = f"{session_id}:{sid}"
        n, c = self._transform_row(flow_features)
        vector = self._vectorize(n, c)
        score = self._score_flow(vector, sid)

        p = self._score_to_prob(score)
        entropy = entropy_from_probability(p)
        epistemic = class_uncertainty_from_probability(p)
        uncertainty = normalize_uncertainty(epistemic=epistemic, aleatoric=entropy)

        likelihood_lookup = lookup_likelihoods(score, self.likelihood_model)
        if likelihood_lookup is not None:
            p_obs_given_attack, p_obs_given_clean, score_bin_index = likelihood_lookup
            raw_attack = p_obs_given_attack
            raw_clean = p_obs_given_clean
        else:
            raw_attack = self._pdf(
                score,
                float(self.loss_stats.get("mal_mean", self.loss_stats.get("mean", 0.0))),
                float(self.loss_stats.get("mal_std", self.loss_stats.get("std", 1.0))),
            )
            raw_clean = self._pdf(
                score,
                float(self.loss_stats.get("benign_mean", self.loss_stats.get("mean", 0.0))),
                float(self.loss_stats.get("benign_std", self.loss_stats.get("std", 1.0))),
            )
            raw_sum = max(1e-9, raw_attack + raw_clean)
            p_obs_given_attack = raw_attack / raw_sum
            p_obs_given_clean = raw_clean / raw_sum
            score_bin_index = -1

        label = "malicious" if p >= 0.5 else "benign"
        return {
            "proba": [1.0 - p, p],
            "prediction": {"label": label, "probability": p},
            "uncertainty": uncertainty,
            "likelihoods": {
                "p_obs_given_attack": float(p_obs_given_attack),
                "p_obs_given_clean": float(p_obs_given_clean),
            },
            "cost": self.cost,
            "agent_id": self.agent_id,
            "metadata": {
                "model": "lstm_autoencoder",
                "model_type": self.model_type,
                "model_path": str(self.model_path),
                "anomaly_score": score,
                "threshold_probability": self.threshold_probability,
                "threshold_aligned": 0.5,
                "stream_id": sid,
                "window_size": int(self.window_size),
                "score_bin_index": int(score_bin_index),
                "raw_score_likelihood_attack": raw_attack,
                "raw_score_likelihood_clean": raw_clean,
            },
        }


DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "lstm_autoencoder.pt")
AGENT = LSTMAutoencoderAgent(
    model_path=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH),
    cost=float(os.getenv("AGENT_COST", "3.0")),
)

app = FastAPI(title="lstm-autoencoder-agent", version="v1")


@app.get("/a2a/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "agent_id": AGENT.agent_id, "version": "v1"}


@app.get("/a2a/capabilities")
def capabilities() -> Dict[str, Any]:
    return {
        "agent_id": AGENT.agent_id,
        "capabilities": ["flow_tabular", "unsw_nb15", "deep_inspection", "anomaly_score", "temporal"],
        "cost": AGENT.cost,
        "metadata": {
            "model_type": AGENT.model_type,
            "runtime_library_versions": AGENT.runtime_library_versions,
            "model_library_versions": AGENT.model_library_versions,
            "version_check": AGENT.version_check,
            "threshold_probability": AGENT.threshold_probability,
            "likelihood_model": AGENT.likelihood_model,
        },
    }


@app.post("/a2a/infer")
def infer(req: InferRequest) -> Dict[str, Any]:
    return AGENT.predict_with_uncertainty(
        req.flow_features,
        seed=req.context.seed,
        flow_id=req.flow_id,
        stream_id=req.context.stream_id,
        session_id=req.context.session_id,
    )


def main() -> None:
    port = int(os.getenv("PORT", "8082"))
    logger.info("Starting LSTM-autoencoder agent server on port %s", port)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
