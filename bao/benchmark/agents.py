from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict


@dataclass
class LightweightAgent:
    name: str = "agent_a"
    cost: float = 1.0

    def predict_proba(self, features: Dict[str, float], seed: int) -> float:
        # Simple tabular heuristic for fast baseline model.
        packet_count = float(features.get("packet_count", 0.0))
        byte_count = float(features.get("byte_count", 0.0))
        flow_duration = float(features.get("flow_duration", 0.0))
        dst_port = float(features.get("dst_port", 0.0))

        z = (
            0.8 * math.log1p(packet_count)
            + 1.1 * math.log1p(byte_count / 100.0)
            + 0.3 * math.log1p(flow_duration)
            + (0.8 if dst_port in (22.0, 23.0, 3389.0) else -0.2)
            - 6.0
        )
        p = 1.0 / (1.0 + math.exp(-z))
        return max(0.001, min(0.999, p))


@dataclass
class BayesianDeepAgent:
    name: str = "agent_b"
    cost: float = 5.0

    def predict_proba(self, features: Dict[str, float], seed: int) -> float:
        # Higher-capacity synthetic proxy with slight stochasticity to mimic MC-dropout spread.
        packet_count = float(features.get("packet_count", 0.0))
        byte_count = float(features.get("byte_count", 0.0))
        flow_duration = float(features.get("flow_duration", 0.0))
        src_port = float(features.get("src_port", 0.0))
        dst_port = float(features.get("dst_port", 0.0))

        z = (
            1.1 * math.log1p(packet_count)
            + 1.3 * math.log1p(byte_count / 120.0)
            + 0.9 * math.log1p(flow_duration)
            + (1.0 if dst_port in (22.0, 23.0, 3389.0, 4444.0) else 0.0)
            + (0.5 if src_port < 1024 else 0.0)
            - 8.0
        )

        # Deterministic seeded noise so repeated seed runs are reproducible.
        rnd = random.Random(seed + int(packet_count) + int(byte_count) + int(dst_port))
        z += rnd.uniform(-0.4, 0.4)

        p = 1.0 / (1.0 + math.exp(-z))
        return max(0.001, min(0.999, p))


def build_default_agents() -> Dict[str, object]:
    return {
        "agent_a": LightweightAgent(),
        "agent_b": BayesianDeepAgent(),
    }
