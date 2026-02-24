from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "agent_training.yaml"

DEFAULTS: Dict[str, Any] = {
    "shared": {
        "seed": 42,
        "validation_fraction": 0.2,
        "preprocessing": {
            "categorical_cols": ["proto", "service", "state"],
            "iqr_floor": 1.0,
            "clip_min": -15.0,
            "clip_max": 15.0,
        },
        "calibration": {
            "method": "logistic",
            "probability_clip": [0.001, 0.999],
        },
    },
    "ocsvm": {},
    "lstm_autoencoder": {},
    "wgan_gp": {},
}


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
            continue
        merged[key] = value
    return merged


def load_agent_training_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    path = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return dict(DEFAULTS)

    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        return dict(DEFAULTS)

    return _deep_merge(DEFAULTS, payload)

