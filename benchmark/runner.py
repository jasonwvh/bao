from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from benchmark.metrics import compute_metrics
from orchestrator.config import file_sha256, load_orchestrator_config
from orchestrator.decision import DecisionCosts, realized_action_cost, select_expected_cost_action


MODEL_PATH_BY_AGENT = {
    "ocsvm": Path("agents/ocsvm/models/ocsvm.pkl"),
    "lstm_autoencoder": Path("agents/lstm_autoencoder/models/lstm_autoencoder.pt"),
    "wgan_gp": Path("agents/wgan_gp/models/wgan_gp.pt"),
}


def label_to_decision(label: object) -> Optional[str]:
    if label is None:
        return None
    x = str(label).strip().lower()
    if x in {"malicious", "attack", "reject"}:
        return "reject"
    if x in {"benign", "clean", "accept"}:
        return "accept"
    return None


def load_default_decision_costs(config_path: Optional[str | Path] = None) -> DecisionCosts:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = (
        Path(config_path).resolve()
        if config_path is not None
        else (repo_root / "config" / "orchestrator_config.utility.yaml")
    )
    cfg = load_orchestrator_config(cfg_path)
    return DecisionCosts(c_fn=float(cfg.decision.c_fn), c_fp=float(cfg.decision.c_fp), c_h=float(cfg.decision.c_h))


@dataclass
class BenchmarkAccumulator:
    prediction_source: str = "probability"
    utility_evaluation: str = "cost_action_parity"
    predictions: List[int] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    probabilities: List[float] = field(default_factory=list)
    query_costs: List[float] = field(default_factory=list)
    action_costs: List[float] = field(default_factory=list)
    action_decisions: List[str] = field(default_factory=list)
    decision_costs: DecisionCosts = field(default_factory=load_default_decision_costs)

    def add_sample(
        self,
        *,
        true_label: int,
        probability: float,
        cost: float,
        decision: Optional[str] = None,
        action_decision: Optional[str] = None,
        label_hint: Optional[str] = None,
    ) -> None:
        p = float(probability)
        pred = infer_prediction(
            prediction_source=self.prediction_source,
            probability=p,
            decision=decision,
            label_hint=label_hint,
            decision_costs=self.decision_costs,
        )
        resolved_action_decision: Optional[str]
        if action_decision is not None:
            resolved_action_decision = str(action_decision).strip().lower()
        elif decision is not None:
            resolved_action_decision = str(decision).strip().lower()
        else:
            resolved_action_decision, _ = select_expected_cost_action(p, self.decision_costs)
        action_cost = realized_action_cost(
            decision=resolved_action_decision,
            prediction=pred,
            true_label=int(true_label),
            costs=self.decision_costs,
        )

        self.labels.append(int(true_label))
        self.probabilities.append(p)
        self.predictions.append(int(pred))
        self.query_costs.append(float(cost))
        self.action_costs.append(float(action_cost))
        self.action_decisions.append(str(resolved_action_decision or "").strip().lower())

    def compute(self, approach: str) -> Dict[str, Any]:
        return compute_metrics(
            predictions=self.predictions,
            labels=self.labels,
            probabilities=self.probabilities,
            costs=self.query_costs,
            query_costs=self.query_costs,
            action_costs=self.action_costs,
            action_decisions=self.action_decisions,
            utility_evaluation=self.utility_evaluation,
            approach=approach,
        )


def infer_prediction(
    *,
    prediction_source: str,
    probability: float,
    decision: Optional[str] = None,
    label_hint: Optional[str] = None,
    decision_costs: Optional[DecisionCosts] = None,
) -> int:
    src = str(prediction_source).strip().lower()
    p = float(probability)
    costs = decision_costs or load_default_decision_costs()

    if src == "decision":
        if decision is not None:
            d = str(decision).strip().lower()
            if d == "reject":
                return 1
            if d == "accept":
                return 0
            # For defer/more_agents/unknown, fall through to posterior threshold.
        action, _ = select_expected_cost_action(p, costs)
        if action == "reject":
            return 1
        if action == "accept":
            return 0
        return 1 if p >= 0.5 else 0

    if label_hint is not None:
        hint = str(label_hint).strip().lower()
        if hint in {"malicious", "attack", "reject"}:
            return 1
        if hint in {"benign", "clean", "accept"}:
            return 0

    return 1 if p >= 0.5 else 0


def dataset_composition(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    pos = 0
    neg = 0
    for row in rows:
        label = row.get("true_label")
        if label is None:
            continue
        total += 1
        if int(label) == 1:
            pos += 1
        else:
            neg += 1

    return {
        "rows_labeled": total,
        "label_1_count": pos,
        "label_0_count": neg,
        "label_1_rate": (pos / total) if total else 0.0,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def attach_routing_observability(metrics: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(metrics)
    if not isinstance(summary, dict):
        return out
    for key in ("agent_utilization", "utilization_band_diagnostics", "routing_selection_counts"):
        if key in summary:
            out[key] = summary[key]
    return out


def reset_sqlite_state(db_path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(db_path) + suffix)
        if candidate.exists():
            candidate.unlink()


def try_git_sha(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def build_benchmark_manifest(
    *,
    repo_root: Path,
    dataset_path: Path,
    config_path: Optional[Path],
    approach: str,
    agents_used: List[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model_hashes: Dict[str, Optional[str]] = {}
    for aid in agents_used:
        rel = MODEL_PATH_BY_AGENT.get(aid)
        if rel is None:
            model_hashes[aid] = None
            continue
        p = (repo_root / rel).resolve()
        model_hashes[aid] = file_sha256(p) if p.exists() else None

    payload: Dict[str, Any] = {
        "approach": approach,
        "timestamp_unix": time.time(),
        "git_sha": try_git_sha(repo_root),
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path) if dataset_path.exists() else None,
        "config_path": str(config_path) if config_path is not None else None,
        "config_sha256": file_sha256(config_path) if config_path is not None and config_path.exists() else None,
        "model_hashes": model_hashes,
        "agents_used": agents_used,
    }
    if extra:
        payload.update(extra)
    return payload
