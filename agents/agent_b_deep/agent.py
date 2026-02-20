from __future__ import annotations

import math
import random
from typing import Any, Dict, Optional


class BayesianDeepAgent:
    def __init__(self, cost: float = 5.0, mc_samples: int = 15):
        self.agent_id = "agent_b"
        self.cost = float(cost)
        self.mc_samples = int(mc_samples)

    def _base_logit(self, flow_features: Dict[str, Any]) -> float:
        packet_count = float(flow_features.get("packet_count", 0.0))
        byte_count = float(flow_features.get("byte_count", 0.0))
        duration = float(flow_features.get("flow_duration", 0.0))
        src_port = float(flow_features.get("src_port", 0.0))
        dst_port = float(flow_features.get("dst_port", 0.0))

        return (
            1.10 * math.log1p(packet_count)
            + 1.20 * math.log1p(byte_count / 120.0)
            + 0.95 * math.log1p(duration)
            + (0.95 if dst_port in (22.0, 23.0, 3389.0, 4444.0) else 0.0)
            + (0.45 if src_port < 1024.0 else 0.0)
            - 7.8
        )

    def predict_with_uncertainty(self, flow_features: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        base = self._base_logit(flow_features)
        rnd = random.Random((seed or 0) + int(flow_features.get("packet_count", 0)) + 17)

        probs = []
        for _ in range(self.mc_samples):
            z = base + rnd.uniform(-0.55, 0.55)
            p = 1.0 / (1.0 + math.exp(-z))
            probs.append(max(0.001, min(0.999, p)))

        p_mean = sum(probs) / len(probs)
        p_var = sum((x - p_mean) ** 2 for x in probs) / max(1, len(probs) - 1)
        entropy = -(p_mean * math.log(max(p_mean, 1e-9)) + (1 - p_mean) * math.log(max(1 - p_mean, 1e-9)))

        return {
            "proba": [1.0 - p_mean, p_mean],
            "prediction": {"label": "malicious" if p_mean > 0.5 else "benign", "probability": p_mean},
            "uncertainty": {
                "epistemic": float(min(1.0, p_var * 10.0)),
                "aleatoric": float(entropy),
                "total_entropy": float(entropy),
            },
            "cost": self.cost,
            "agent_id": self.agent_id,
            "metadata": {"model": "mc-dropout-proxy", "mc_samples": self.mc_samples},
        }
