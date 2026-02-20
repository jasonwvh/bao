from __future__ import annotations

from typing import Any, Dict, Optional


class ContextualGraphAgent:
    def __init__(self, cost: float = 10.0):
        self.agent_id = "agent_c"
        self.cost = float(cost)

    def predict_with_uncertainty(self, flow_features: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        return {
            "proba": [0.5, 0.5],
            "prediction": {"label": "unknown", "probability": 0.5},
            "uncertainty": {"epistemic": 1.0, "aleatoric": 1.0, "total_entropy": 1.0},
            "cost": self.cost,
            "agent_id": self.agent_id,
            "metadata": {"status": "not_enabled"},
        }
