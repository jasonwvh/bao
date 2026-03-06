from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class OrchestrationConfig:
    seed: int
    agent_registry_path: Path
    agent_sequence: list[str]


@dataclass(frozen=True)
class BeliefConfig:
    prior_attack_rate: float
    eps: float
    update_mode: str
    reliability_strength: float


@dataclass(frozen=True)
class FusionConfig:
    uncertainty_weight_gamma: float
    weight_floor: float
    agent_weights: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionConfig:
    c_fn: float
    c_fp: float
    c_h: float
    defer_enabled: bool
    defer_uncertainty_threshold: float
    defer_margin_from_half: float
    defer_require_all_agents_exhausted: bool


@dataclass(frozen=True)
class QueryConfig:
    first_agent: Optional[str]
    uncertainty_threshold: float
    max_agents: int
    detector_cost_fraction: float


@dataclass(frozen=True)
class VOIConfig:
    enabled: bool
    rho: float
    mode: str
    min_net_gain: float


@dataclass(frozen=True)
class BenchmarkConfig:
    reset_state: bool
    write_manifest: bool


@dataclass(frozen=True)
class MetricsConfig:
    warnings_enabled: bool


@dataclass(frozen=True)
class StateConfig:
    sqlite_path: Path


@dataclass(frozen=True)
class A2AConfig:
    retries: int


@dataclass(frozen=True)
class PreprocessingConfig:
    schema_path: Optional[Path]


@dataclass(frozen=True)
class OrchestratorConfig:
    config_path: Path
    raw: Dict[str, Any]
    orchestration: OrchestrationConfig
    belief: BeliefConfig
    fusion: FusionConfig
    decision: DecisionConfig
    query: QueryConfig
    voi: VOIConfig
    benchmark: BenchmarkConfig
    metrics: MetricsConfig
    state: StateConfig
    a2a: A2AConfig
    preprocessing: PreprocessingConfig


def _resolve_path(base_dir: Path, value: Optional[str], default: str) -> Path:
    raw = str(value if value not in (None, "") else default)
    p = Path(raw)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_list_of_str(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(x).strip() for x in value if str(x).strip()]


def _normalize_choice(value: Any, default: str, allowed: set[str]) -> str:
    v = str(value if value not in (None, "") else default).strip().lower()
    if v in allowed:
        return v
    return default


def load_config(path: str | Path) -> OrchestratorConfig:
    cfg_path = Path(path).resolve()
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("orchestrator config must be a mapping")

    base_dir = cfg_path.parent

    orch_raw = dict(raw.get("orchestration", {}) or {})
    belief_raw = dict(raw.get("belief", {}) or {})
    fusion_raw = dict(raw.get("fusion", {}) or {})
    decision_raw = dict(raw.get("decision", {}) or {})
    query_raw = dict(raw.get("query", {}) or {})
    voi_raw = dict(raw.get("voi", {}) or {})
    benchmark_raw = dict(raw.get("benchmark", {}) or {})
    metrics_raw = dict(raw.get("metrics", {}) or {})
    state_raw = dict(raw.get("state", {}) or {})
    a2a_raw = dict(raw.get("a2a", {}) or {})
    pre_raw = dict(raw.get("preprocessing", {}) or {})

    decision_costs_raw = dict(decision_raw.get("costs", {}) or {})
    defer_raw = dict(decision_raw.get("defer_policy", {}) or {})

    prior = max(1e-9, min(1.0 - 1e-9, _to_float(belief_raw.get("prior_attack_rate", 0.5), 0.5)))
    eps = max(1e-9, _to_float(belief_raw.get("eps", 1e-6), 1e-6))

    uncertainty_threshold = _to_float(query_raw.get("uncertainty_threshold", 0.6), 0.6)
    uncertainty_threshold = max(0.0, min(0.69314718056, uncertainty_threshold))

    defer_threshold = _to_float(defer_raw.get("uncertainty_threshold", 0.66), 0.66)
    defer_threshold = max(0.0, min(0.69314718056, defer_threshold))

    defer_margin = _to_float(defer_raw.get("margin_from_half", 0.08), 0.08)
    defer_margin = max(0.0, min(0.5, defer_margin))

    schema_value = pre_raw.get("schema_path")
    schema_path = _resolve_path(base_dir, schema_value, "") if schema_value not in (None, "") else None

    return OrchestratorConfig(
        config_path=cfg_path,
        raw=raw,
        orchestration=OrchestrationConfig(
            seed=_to_int(orch_raw.get("seed", 7), 7),
            agent_registry_path=_resolve_path(base_dir, orch_raw.get("agent_registry_path"), "agents.yaml"),
            agent_sequence=_as_list_of_str(orch_raw.get("agent_sequence", [])),
        ),
        belief=BeliefConfig(
            prior_attack_rate=prior,
            eps=eps,
            update_mode=_normalize_choice(
                belief_raw.get("update_mode", "likelihood_ratio"),
                default="likelihood_ratio",
                allowed={"likelihood_ratio", "probability_pool"},
            ),
            reliability_strength=max(0.0, _to_float(belief_raw.get("reliability_strength", 1.0), 1.0)),
        ),
        fusion=FusionConfig(
            uncertainty_weight_gamma=max(0.0, _to_float(fusion_raw.get("uncertainty_weight_gamma", 1.5), 1.5)),
            weight_floor=max(0.0, _to_float(fusion_raw.get("weight_floor", 0.1), 0.1)),
            agent_weights={
                str(k): max(1e-6, _to_float(v, 1.0))
                for k, v in dict(fusion_raw.get("agent_weights", {}) or {}).items()
            },
        ),
        decision=DecisionConfig(
            c_fn=_to_float(decision_costs_raw.get("c_fn", 25.0), 25.0),
            c_fp=_to_float(decision_costs_raw.get("c_fp", 2.0), 2.0),
            c_h=_to_float(decision_costs_raw.get("c_h", 2.0), 2.0),
            defer_enabled=_to_bool(defer_raw.get("enabled", True), True),
            defer_uncertainty_threshold=defer_threshold,
            defer_margin_from_half=defer_margin,
            defer_require_all_agents_exhausted=_to_bool(
                defer_raw.get("require_all_agents_exhausted", True),
                True,
            ),
        ),
        query=QueryConfig(
            first_agent=(str(query_raw.get("first_agent", "")).strip() or None),
            uncertainty_threshold=uncertainty_threshold,
            max_agents=max(1, _to_int(query_raw.get("max_agents", 2), 2)),
            detector_cost_fraction=max(1e-6, _to_float(query_raw.get("detector_cost_fraction", 0.1), 0.1)),
        ),
        voi=VOIConfig(
            enabled=_to_bool(voi_raw.get("enabled", True), True),
            rho=max(0.0, min(1.0, _to_float(voi_raw.get("rho", 0.7), 0.7))),
            mode=_normalize_choice(
                voi_raw.get("mode", "expected_cost_reduction"),
                default="expected_cost_reduction",
                allowed={"expected_cost_reduction", "legacy_approx"},
            ),
            min_net_gain=_to_float(
                voi_raw.get(
                    "min_net_gain",
                    query_raw.get("min_expected_gain", 0.0),
                ),
                0.0,
            ),
        ),
        benchmark=BenchmarkConfig(
            reset_state=_to_bool(benchmark_raw.get("reset_state", True), True),
            write_manifest=_to_bool(benchmark_raw.get("write_manifest", True), True),
        ),
        metrics=MetricsConfig(
            warnings_enabled=_to_bool(metrics_raw.get("warnings_enabled", True), True),
        ),
        state=StateConfig(
            sqlite_path=_resolve_path(base_dir, state_raw.get("sqlite_path"), "../artifacts/state/bao_state.sqlite")
        ),
        a2a=A2AConfig(
            retries=max(0, _to_int(a2a_raw.get("retries", 0), 0)),
        ),
        preprocessing=PreprocessingConfig(
            schema_path=schema_path,
        ),
    )
