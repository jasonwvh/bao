from __future__ import annotations

import math
from typing import Any, Dict, Optional


class LightweightAgent:
    def __init__(self, cost: float = 1.0):
        self.agent_id = "agent_a"
        self.cost = float(cost)

    def predict_with_uncertainty(self, flow_features: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        packet_count = float(flow_features.get("packet_count", 0.0))
        byte_count = float(flow_features.get("byte_count", 0.0))
        duration = float(flow_features.get("flow_duration", 0.0))
        dst_port = float(flow_features.get("dst_port", 0.0))

        z = (
            0.75 * math.log1p(packet_count)
            + 0.95 * math.log1p(byte_count / 100.0)
            + 0.25 * math.log1p(duration)
            + (0.7 if dst_port in (22.0, 23.0, 3389.0) else -0.15)
            - 5.8
        )
        p = 1.0 / (1.0 + math.exp(-z))
        p = float(max(0.001, min(0.999, p)))
        entropy = -(p * math.log(max(p, 1e-9)) + (1 - p) * math.log(max(1 - p, 1e-9)))

        return {
            "proba": [1.0 - p, p],
            "prediction": {"label": "malicious" if p > 0.5 else "benign", "probability": p},
            "uncertainty": {"epistemic": entropy * 0.4, "aleatoric": entropy * 0.6, "total_entropy": entropy},
            "cost": self.cost,
            "agent_id": self.agent_id,
            "metadata": {"model": "lightweight-logit"},
        }
