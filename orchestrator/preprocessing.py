from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from agents.common.preprocessing import UNSWPreprocessor, schema_from_json, transform_row

logger = logging.getLogger("orchestrator.preprocessing")


class OrchestratorPreprocessor:
    """Preprocesses flow features for agents using UNSW-NB15 schema.

    Loads a preprocessor schema from JSON and applies transformations
    to raw flow features, producing pp_num_* and pp_cat_* fields that
    agents can consume directly.
    """

    def __init__(self, schema_path: Optional[str | Path] = None):
        self.schema_path = Path(schema_path).resolve() if schema_path else None
        self.preprocessor: Optional[UNSWPreprocessor] = None
        self.window_size: int = 1
        self.stride: int = 1

        if self.schema_path:
            if self.schema_path.exists():
                try:
                    payload = schema_from_json(self.schema_path)
                    pre = payload.get("preprocessor")
                    if isinstance(pre, dict):
                        self.preprocessor = UNSWPreprocessor.from_dict(pre)
                        self.window_size = int(payload.get("window_size", 1))
                        self.stride = int(payload.get("stride", 1))
                        logger.info(
                            "Loaded preprocessor from %s (window=%d, stride=%d)",
                            self.schema_path,
                            self.window_size,
                            self.stride,
                        )
                    else:
                        logger.warning("Schema missing 'preprocessor' key: %s", self.schema_path)
                except Exception as exc:
                    logger.warning("Failed to load schema from %s: %s", self.schema_path, exc)
            else:
                logger.warning("Schema file not found: %s", self.schema_path)

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def transform(self, flow_features: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(flow_features)

        spkts = self._to_float(out.get("spkts", 0.0))
        dpkts = self._to_float(out.get("dpkts", 0.0))
        sbytes = self._to_float(out.get("sbytes", 0.0))
        dbytes = self._to_float(out.get("dbytes", 0.0))
        dur = self._to_float(out.get("dur", 0.0))

        out.setdefault("packet_count", spkts + dpkts)
        out.setdefault("byte_count", sbytes + dbytes)
        out.setdefault("flow_duration", dur)

        if self.preprocessor is None:
            return out

        n, c = transform_row(self.preprocessor, out)
        for i, col in enumerate(self.preprocessor.numeric_cols):
            out[f"pp_num_{col}"] = float(n[i])
        for i, col in enumerate(self.preprocessor.categorical_cols):
            out[f"pp_cat_{col}"] = int(c[i])

        return out
