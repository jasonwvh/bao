from __future__ import annotations

import logging
import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from agents.common.calibration import (
    align_probability_threshold,
    class_uncertainty_from_probability,
    entropy_from_probability,
    logistic_probability,
    normalize_uncertainty,
)
from agents.common.versioning import collect_library_versions, compare_versions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("ocsvm")


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


class OCSVMAgent:
    def __init__(self, model_path: str | Path, cost: float = 2.5):
        self.agent_id = "ocsvm"
        self.cost = float(cost)
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. Train with: python3 -m agents.ocsvm.train"
            )

        with self.model_path.open("rb") as f:
            payload = pickle.load(f)

        self.model = payload["model"]
        self.calibrator = payload.get("calibrator") or {"coef": 1.0, "intercept": 0.0}
        self.threshold_probability = float(payload.get("threshold_probability", 0.5))
        self.probability_clip = payload.get("probability_clip", [0.001, 0.999])
        self.p_lo = float(self.probability_clip[0])
        self.p_hi = float(self.probability_clip[1])

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
        density_model = payload.get("density_model", {})
        self.density_bin_edges = np.asarray(density_model.get("bin_edges", []), dtype=np.float64)
        self.density_values = np.asarray(density_model.get("density", []), dtype=np.float64)
        if self.density_bin_edges.size < 2 or self.density_values.size == 0:
            self.density_bin_edges = np.asarray([0.0, 1.0], dtype=np.float64)
            self.density_values = np.asarray([1.0], dtype=np.float64)

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

    def _local_density(self, score: float) -> float:
        edges = self.density_bin_edges
        vals = self.density_values
        if edges.size < 2 or vals.size == 0:
            return 1.0
        idx = int(np.searchsorted(edges, float(score), side="right") - 1)
        idx = int(np.clip(idx, 0, len(vals) - 1))
        return float(np.clip(vals[idx], 0.0, 1.0))

    def predict_with_uncertainty(self, flow_features: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        del seed
        x = self._vectorize(flow_features).reshape(1, -1)
        anomaly_score = float(-self.model.decision_function(x)[0])

        p_raw = float(
            logistic_probability(
                anomaly_score,
                self.calibrator,
                clip_lo=self.p_lo,
                clip_hi=self.p_hi,
            )
        )
        p = align_probability_threshold(p_raw, self.threshold_probability)

        entropy = entropy_from_probability(p)
        class_uncertainty = class_uncertainty_from_probability(p)
        local_density = self._local_density(anomaly_score)
        ood_uncertainty = float(max(0.0, 1.0 - local_density))
        epistemic = float(max(class_uncertainty, ood_uncertainty))
        uncertainty = normalize_uncertainty(epistemic=epistemic, aleatoric=entropy)

        raw_attack = self._pdf(
            anomaly_score,
            float(self.score_stats.get("mal_mean", self.score_stats.get("mean", 0.0))),
            float(self.score_stats.get("mal_std", self.score_stats.get("std", 1.0))),
        )
        raw_clean = self._pdf(
            anomaly_score,
            float(self.score_stats.get("benign_mean", self.score_stats.get("mean", 0.0))),
            float(self.score_stats.get("benign_std", self.score_stats.get("std", 1.0))),
        )
        raw_sum = max(1e-9, raw_attack + raw_clean)
        score_attack = raw_attack / raw_sum
        score_clean = raw_clean / raw_sum

        # Keep likelihood-ratio updates aligned with calibrated posterior behavior.
        p_obs_given_attack = max(1e-9, 0.7 * p + 0.3 * score_attack)
        p_obs_given_clean = max(1e-9, 0.7 * (1.0 - p) + 0.3 * score_clean)

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
                "model": "ocsvm",
                "model_type": "one_class_svm",
                "model_path": str(self.model_path),
                "anomaly_score": anomaly_score,
                "threshold_probability": self.threshold_probability,
                "threshold_aligned": 0.5,
                "ood_local_density": local_density,
                "ood_uncertainty": ood_uncertainty,
                "raw_score_likelihood_attack": raw_attack,
                "raw_score_likelihood_clean": raw_clean,
            },
        }


DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ocsvm.pkl")
AGENT = OCSVMAgent(
    model_path=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH),
    cost=float(os.getenv("AGENT_COST", "1.0")),
)

app = FastAPI(title="ocsvm-agent", version="v1")


@app.get("/a2a/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "agent_id": AGENT.agent_id, "version": "v1"}


@app.get("/a2a/capabilities")
def capabilities() -> Dict[str, Any]:
    return {
        "agent_id": AGENT.agent_id,
        "capabilities": ["flow_tabular", "unsw_nb15", "anomaly_score", "one_class"],
        "cost": AGENT.cost,
        "metadata": {
            "model_type": "one_class_svm",
            "runtime_library_versions": AGENT.runtime_library_versions,
            "model_library_versions": AGENT.model_library_versions,
            "version_check": AGENT.version_check,
        },
    }


@app.post("/a2a/infer")
def infer(req: InferRequest) -> Dict[str, Any]:
    return AGENT.predict_with_uncertainty(req.flow_features, seed=req.context.seed)


def main() -> None:
    port = int(os.getenv("PORT", "8081"))
    logger.info("Starting One-Class SVM agent server on port %s", port)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
