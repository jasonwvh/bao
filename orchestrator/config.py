from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

UPDATE_MODES = {"posterior_first", "likelihood_strict"}
FUSION_METHODS = {"logit_pool"}
PREDICTION_SOURCES = {"decision", "probability"}


@dataclass(frozen=True)
class OrchestrationConfig:
    seed: int
    agent_registry_path: Path
    update_mode: str
    agent_sequence: list[str]


@dataclass(frozen=True)
class BeliefConfig:
    prior_attack_rate: float
    eps: float
    likelihood_sanity_gate: bool


@dataclass(frozen=True)
class FusionConfig:
    method: str
    agent_weights: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionConfig:
    policy: str
    c_fn: float
    c_fp: float
    c_h: float


@dataclass(frozen=True)
class QueryConfig:
    uncertainty_threshold: float
    uncertainty_threshold_stage1: float
    uncertainty_threshold_stage2: float
    max_agents: int


@dataclass(frozen=True)
class VOIConfig:
    enabled: bool
    rho: float


@dataclass(frozen=True)
class BenchmarkConfig:
    reset_state: bool
    prediction_source: str
    write_manifest: bool


@dataclass(frozen=True)
class A2AConfig:
    retries: int


@dataclass(frozen=True)
class StateConfig:
    sqlite_path: Path


@dataclass(frozen=True)
class LoggingConfig:
    jsonl_path: Path
    enable_mlflow: bool


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
    a2a: A2AConfig
    state: StateConfig
    logging: LoggingConfig
    preprocessing: PreprocessingConfig


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default


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


def _resolve_path(base_dir: Path, value: Optional[str], default: str) -> Path:
    raw = str(value if value is not None else default)
    p = Path(raw)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _as_list_of_str(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def load_orchestrator_config(path: str | Path) -> OrchestratorConfig:
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
    a2a_raw = dict(raw.get("a2a", {}) or {})
    state_raw = dict(raw.get("state", {}) or {})
    logging_raw = dict(raw.get("logging", {}) or {})
    pre_raw = dict(raw.get("preprocessing", {}) or {})

    # Backward compatibility with older config layout.
    legacy_costs = dict(raw.get("costs", {}) or {})
    legacy_thresholds = dict(raw.get("thresholds", {}) or {})

    update_mode = str(orch_raw.get("update_mode", "posterior_first")).strip().lower()
    if update_mode not in UPDATE_MODES:
        raise ValueError(f"unsupported orchestration.update_mode={update_mode!r}")

    fusion_method = str(fusion_raw.get("method", "logit_pool")).strip().lower()
    if fusion_method not in FUSION_METHODS:
        raise ValueError(f"unsupported fusion.method={fusion_method!r}")

    prediction_source = str(benchmark_raw.get("prediction_source", "decision")).strip().lower()
    if prediction_source not in PREDICTION_SOURCES:
        raise ValueError(f"unsupported benchmark.prediction_source={prediction_source!r}")

    rho = _to_float(voi_raw.get("rho", 0.7), 0.7)
    rho = max(0.0, min(1.0, rho))

    prior = _to_float(belief_raw.get("prior_attack_rate", 0.5), 0.5)
    prior = max(1e-9, min(1.0 - 1e-9, prior))

    eps = _to_float(belief_raw.get("eps", 1e-6), 1e-6)
    eps = max(1e-9, eps)

    uncertainty_threshold = _to_float(
        query_raw.get("uncertainty_threshold", legacy_thresholds.get("uncertainty", 0.6)),
        0.6,
    )
    # Entropy for Bernoulli lives in [0, ln 2]
    uncertainty_threshold = max(0.0, min(0.69314718056, uncertainty_threshold))
    uncertainty_threshold_stage1 = _to_float(
        query_raw.get("uncertainty_threshold_stage1", uncertainty_threshold),
        uncertainty_threshold,
    )
    uncertainty_threshold_stage1 = max(0.0, min(0.69314718056, uncertainty_threshold_stage1))
    uncertainty_threshold_stage2 = _to_float(
        query_raw.get("uncertainty_threshold_stage2", uncertainty_threshold_stage1),
        uncertainty_threshold_stage1,
    )
    uncertainty_threshold_stage2 = max(0.0, min(0.69314718056, uncertainty_threshold_stage2))

    max_agents = _to_int(query_raw.get("max_agents", orch_raw.get("max_iterations", 1)), 1)
    max_agents = max(1, max_agents)

    weights_raw = fusion_raw.get("agent_weights", {})
    weights: Dict[str, float] = {}
    if isinstance(weights_raw, Mapping):
        for aid, value in weights_raw.items():
            w = _to_float(value, 1.0)
            weights[str(aid)] = max(1e-6, w)

    state_path = _resolve_path(base_dir, state_raw.get("sqlite_path"), "../artifacts/state/bao_state.sqlite")
    jsonl_path = _resolve_path(base_dir, logging_raw.get("jsonl_path"), "../artifacts/replay/flows.jsonl")
    registry_path = _resolve_path(base_dir, orch_raw.get("agent_registry_path"), "agents.yaml")

    schema_value = pre_raw.get("schema_path")
    schema_path = _resolve_path(base_dir, schema_value, "") if schema_value else None

    config = OrchestratorConfig(
        config_path=cfg_path,
        raw=raw,
        orchestration=OrchestrationConfig(
            seed=_to_int(orch_raw.get("seed", 7), 7),
            agent_registry_path=registry_path,
            update_mode=update_mode,
            agent_sequence=_as_list_of_str(orch_raw.get("agent_sequence", [])),
        ),
        belief=BeliefConfig(
            prior_attack_rate=prior,
            eps=eps,
            likelihood_sanity_gate=_to_bool(belief_raw.get("likelihood_sanity_gate", True), True),
        ),
        fusion=FusionConfig(method=fusion_method, agent_weights=weights),
        decision=DecisionConfig(
            policy=str(decision_raw.get("policy", "expected_cost_min")).strip().lower(),
            c_fn=_to_float(decision_raw.get("costs", {}).get("c_fn", legacy_costs.get("c_fn", 500.0)), 500.0),
            c_fp=_to_float(decision_raw.get("costs", {}).get("c_fp", legacy_costs.get("c_fp", 5.0)), 5.0),
            c_h=_to_float(decision_raw.get("costs", {}).get("c_h", legacy_costs.get("c_h", 5000.0)), 5000.0),
        ),
        query=QueryConfig(
            uncertainty_threshold=uncertainty_threshold,
            uncertainty_threshold_stage1=uncertainty_threshold_stage1,
            uncertainty_threshold_stage2=uncertainty_threshold_stage2,
            max_agents=max_agents,
        ),
        voi=VOIConfig(
            enabled=_to_bool(voi_raw.get("enabled", True), True),
            rho=rho,
        ),
        benchmark=BenchmarkConfig(
            reset_state=_to_bool(benchmark_raw.get("reset_state", True), True),
            prediction_source=prediction_source,
            write_manifest=_to_bool(benchmark_raw.get("write_manifest", True), True),
        ),
        a2a=A2AConfig(retries=_to_int(a2a_raw.get("retries", 0), 0)),
        state=StateConfig(sqlite_path=state_path),
        logging=LoggingConfig(
            jsonl_path=jsonl_path,
            enable_mlflow=_to_bool(logging_raw.get("enable_mlflow", False), False),
        ),
        preprocessing=PreprocessingConfig(schema_path=schema_path),
    )

    return config


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
