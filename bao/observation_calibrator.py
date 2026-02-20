from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

try:
    from scipy.stats import gaussian_kde
except Exception:  # pragma: no cover
    gaussian_kde = None

try:
    from river import drift as river_drift
except Exception:  # pragma: no cover
    river_drift = None


class _FallbackDriftDetector:
    def __init__(self, threshold: float = 0.15, window: int = 40):
        self.threshold = threshold
        self.window = window
        self.values: Deque[float] = deque(maxlen=window * 2)
        self.drift_detected = False

    def update(self, value: float) -> None:
        self.values.append(value)
        self.drift_detected = False
        if len(self.values) < self.window * 2:
            return
        vals = list(self.values)
        a = vals[-self.window :]
        b = vals[-2 * self.window : -self.window]
        if abs((sum(a) / len(a)) - (sum(b) / len(b))) > self.threshold:
            self.drift_detected = True


class ExperienceReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer: Deque[Dict[str, Any]] = deque(maxlen=capacity)

    def add(self, observation: Dict[str, Any]) -> None:
        self.buffer.append(observation)

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self.buffer)


class ObservationModel:
    def __init__(self, agent_id: str, enable_refit: bool = False):
        self.agent_id = agent_id
        self.enable_refit = enable_refit
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.is_calibrator_fitted = False
        self.kde_benign = None
        self.kde_malicious = None
        self.experience_buffer = ExperienceReplayBuffer()

        self.online_mean = {0: 0.5, 1: 0.5}
        self.online_var = {0: 0.08, 1: 0.08}
        self.n_samples = {0: 0, 1: 0}

        self.drift_detected_count = 0
        self.adwin_enabled = river_drift is not None
        self.drift_detector = river_drift.ADWIN() if river_drift else _FallbackDriftDetector()

    def _extract_proba(self, agent_output: Dict[str, Any]) -> float:
        p = agent_output.get("proba", [0.5, 0.5])
        if isinstance(p, (list, tuple)):
            return float(p[1] if len(p) > 1 else p[0])
        return float(p)

    def fit(self, outputs: List[Dict[str, Any]], labels: List[int]) -> None:
        if len(outputs) < 5:
            return
        probs = np.array([self._extract_proba(o) for o in outputs], dtype=float)
        ys = np.array(labels, dtype=int)

        self.calibrator.fit(probs, ys)
        self.is_calibrator_fitted = True

        for y in [0, 1]:
            cls = probs[ys == y]
            if len(cls) > 1:
                self.online_mean[y] = float(np.mean(cls))
                self.online_var[y] = float(np.var(cls) + 1e-4)
                self.n_samples[y] = int(len(cls))

        if gaussian_kde is not None:
            benign = probs[ys == 0]
            malicious = probs[ys == 1]
            if len(benign) > 10:
                self.kde_benign = gaussian_kde(benign)
            if len(malicious) > 10:
                self.kde_malicious = gaussian_kde(malicious)

    def calibrate_proba(self, raw_proba: float) -> float:
        p = min(0.999, max(0.001, float(raw_proba)))
        if not self.is_calibrator_fitted:
            return p
        return float(np.clip(self.calibrator.predict([p])[0], 0.001, 0.999))

    def likelihood(self, output: Dict[str, Any], y: int) -> float:
        z = self._extract_proba(output)
        if y == 0 and self.kde_benign is not None:
            return float(max(1e-9, self.kde_benign.evaluate([z])[0]))
        if y == 1 and self.kde_malicious is not None:
            return float(max(1e-9, self.kde_malicious.evaluate([z])[0]))
        mu = self.online_mean.get(y, 0.5)
        var = max(1e-4, self.online_var.get(y, 0.08))
        sigma = math.sqrt(var)
        coeff = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
        return float(max(1e-9, coeff * math.exp(-((z - mu) ** 2) / (2.0 * var))))

    def sample_observation(self, y: int, rng: np.random.Generator) -> float:
        if y == 0 and self.kde_benign is not None:
            return float(np.clip(self.kde_benign.resample(1)[0, 0], 0.001, 0.999))
        if y == 1 and self.kde_malicious is not None:
            return float(np.clip(self.kde_malicious.resample(1)[0, 0], 0.001, 0.999))
        mu = self.online_mean.get(y, 0.5)
        sigma = math.sqrt(max(1e-4, self.online_var.get(y, 0.08)))
        return float(np.clip(rng.normal(mu, sigma), 0.001, 0.999))

    def online_update(self, output: Dict[str, Any], true_label: int) -> None:
        p = self._extract_proba(output)
        y = int(true_label)

        self.experience_buffer.add({"output": output, "label": y})
        n = self.n_samples[y] + 1
        delta = p - self.online_mean[y]
        new_mean = self.online_mean[y] + delta / n
        new_var = ((n - 1) * self.online_var[y] + delta * (p - new_mean)) / max(1, n)
        self.online_mean[y] = float(new_mean)
        self.online_var[y] = float(max(1e-4, new_var))
        self.n_samples[y] = n

        err = abs(y - p)
        self.drift_detector.update(float(err))
        drift = getattr(self.drift_detector, "drift_detected", False)
        if drift and self.enable_refit:
            self.drift_detected_count += 1
            self.refit_from_experience()

        total = self.n_samples[0] + self.n_samples[1]
        if self.enable_refit and total % 100 == 0:
            self.refit_from_experience()

    def refit_from_experience(self) -> None:
        items = self.experience_buffer.get_all()
        if len(items) < 20:
            return
        outputs = [x["output"] for x in items[-1000:]]
        labels = [int(x["label"]) for x in items[-1000:]]
        self.fit(outputs, labels)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "n_benign": self.n_samples[0],
            "n_malicious": self.n_samples[1],
            "benign_mean": self.online_mean[0],
            "malicious_mean": self.online_mean[1],
            "drift_detected_count": self.drift_detected_count,
            "is_calibrated": self.is_calibrator_fitted,
            "adwin_enabled": self.adwin_enabled,
        }


class ObservationModelCalibrator:
    def __init__(self):
        self.models: Dict[str, ObservationModel] = {}

    def get_or_create_model(self, agent_id: str) -> ObservationModel:
        if agent_id not in self.models:
            self.models[agent_id] = ObservationModel(agent_id)
        return self.models[agent_id]

    def online_update(self, agent_id: str, output: Dict[str, Any], true_label: int) -> None:
        self.get_or_create_model(agent_id).online_update(output, true_label)

    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        return {aid: model.get_statistics() for aid, model in self.models.items()}
