from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

UPDATE_MODES = {"posterior_first", "likelihood_strict"}
FUSION_METHODS = {"logit_pool", "handoff_latest", "utility_select"}
QUERY_POLICIES = {"strict_cascade", "adaptive_router"}
ROUTING_TIE_BREAKS = {"agent_sequence"}
CALIBRATION_MODES = {"validation_derived"}
PREDICTION_SOURCES = {"decision", "probability"}
ORCHESTRATION_ENGINES = {"deterministic", "langgraph"}
FIRST_AGENT_STRATEGIES = {"dynamic_cheapest", "explicit"}
logger = logging.getLogger("orchestrator.config")


@dataclass(frozen=True)
class OrchestrationConfig:
    seed: int
    agent_registry_path: Path
    update_mode: str
    agent_sequence: list[str]
    engine: str
    first_agent_strategy: str


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
class CostCalibrationConfig:
    enabled: bool
    mode: str
    c_fn_grid: list[float] = field(default_factory=list)
    c_fp_grid: list[float] = field(default_factory=list)
    c_h_grid: list[float] = field(default_factory=list)
    min_expected_gain_grid: list[float] = field(default_factory=list)
    max_agents_grid: list[int] = field(default_factory=list)
    fusion_method: str = "handoff_latest"


@dataclass(frozen=True)
class DecisionConfig:
    policy: str
    c_fn: float
    c_fp: float
    c_h: float
    accuracy_floor_delta: float
    cost_calibration: CostCalibrationConfig


@dataclass(frozen=True)
class QueryConfig:
    policy: str
    uncertainty_threshold: float
    max_agents: int
    min_expected_gain: float
    first_agent: Optional[str]
    utilization_targets: list["UtilizationTargetConfig"]
    utilization_warmup_flows: int


@dataclass(frozen=True)
class VOIConfig:
    enabled: bool
    rho: float


@dataclass(frozen=True)
class RoutingConfig:
    profile_path: Optional[Path]
    bin_count: int
    min_samples_per_bin: int
    tie_break: str
    langgraph_perf_guardrail_overhead: float
    parity_tolerance: float


@dataclass(frozen=True)
class UtilizationTargetConfig:
    agent_id: str
    min_rate: float
    max_rate: float
    penalty_under: float
    penalty_over: float


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
    routing: RoutingConfig
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


def _as_list_of_float(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except Exception:
            continue
    return out


def _as_list_of_int(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def _as_utilization_targets(value: Any) -> list[UtilizationTargetConfig]:
    if not isinstance(value, list):
        return []
    out: list[UtilizationTargetConfig] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        agent_id = str(item.get("agent_id", "")).strip()
        if not agent_id:
            continue
        min_rate = _to_float(item.get("min_rate", 0.0), 0.0)
        max_rate = _to_float(item.get("max_rate", 1.0), 1.0)
        if max_rate < min_rate:
            min_rate, max_rate = max_rate, min_rate
        out.append(
            UtilizationTargetConfig(
                agent_id=agent_id,
                min_rate=max(0.0, min(1.0, min_rate)),
                max_rate=max(0.0, min(1.0, max_rate)),
                penalty_under=max(0.0, _to_float(item.get("penalty_under", 0.0), 0.0)),
                penalty_over=max(0.0, _to_float(item.get("penalty_over", 0.0), 0.0)),
            )
        )
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
    routing_raw = dict(raw.get("routing", {}) or {})
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
    engine = str(orch_raw.get("engine", "deterministic")).strip().lower()
    if engine not in ORCHESTRATION_ENGINES:
        raise ValueError(f"unsupported orchestration.engine={engine!r}")

    fusion_method = str(fusion_raw.get("method", "logit_pool")).strip().lower()
    if fusion_method not in FUSION_METHODS:
        raise ValueError(f"unsupported fusion.method={fusion_method!r}")

    query_policy = str(query_raw.get("policy", "strict_cascade")).strip().lower()
    if query_policy not in QUERY_POLICIES:
        raise ValueError(f"unsupported query.policy={query_policy!r}")

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
    # Backward compatibility: stage-specific thresholds are deprecated.
    if "uncertainty_threshold_stage1" in query_raw or "uncertainty_threshold_stage2" in query_raw:
        logger.warning(
            "query.uncertainty_threshold_stage1/stage2 are deprecated and ignored; "
            "use query.uncertainty_threshold only"
        )

    max_agents = _to_int(query_raw.get("max_agents", orch_raw.get("max_iterations", 1)), 1)
    max_agents = max(1, max_agents)

    min_expected_gain = _to_float(query_raw.get("min_expected_gain", 0.0), 0.0)
    first_agent_value = query_raw.get("first_agent", "")
    first_agent_raw = "" if first_agent_value is None else str(first_agent_value).strip()
    first_agent = first_agent_raw if first_agent_raw else None
    first_agent_strategy_raw = orch_raw.get("first_agent_strategy")
    if first_agent_strategy_raw is None:
        if first_agent is not None:
            first_agent_strategy = "explicit"
            logger.warning(
                "query.first_agent is set but orchestration.first_agent_strategy is missing; "
                "defaulting to 'explicit' for backward compatibility"
            )
        else:
            first_agent_strategy = "dynamic_cheapest"
    else:
        first_agent_strategy = str(first_agent_strategy_raw).strip().lower()
    if first_agent_strategy not in FIRST_AGENT_STRATEGIES:
        raise ValueError(f"unsupported orchestration.first_agent_strategy={first_agent_strategy!r}")

    utilization_targets = _as_utilization_targets(query_raw.get("utilization_targets", []))
    utilization_warmup_flows = max(0, _to_int(query_raw.get("utilization_warmup_flows", 500), 500))

    weights_raw = fusion_raw.get("agent_weights", {})
    weights: Dict[str, float] = {}
    if isinstance(weights_raw, Mapping):
        for aid, value in weights_raw.items():
            w = _to_float(value, 1.0)
            weights[str(aid)] = max(1e-6, w)

    cost_cal_raw = dict(decision_raw.get("cost_calibration", {}) or {})
    cost_cal_mode = str(cost_cal_raw.get("mode", "validation_derived")).strip().lower()
    if cost_cal_mode not in CALIBRATION_MODES:
        raise ValueError(f"unsupported decision.cost_calibration.mode={cost_cal_mode!r}")
    cost_cal_fusion = str(cost_cal_raw.get("fusion_method", "handoff_latest")).strip().lower()
    if cost_cal_fusion not in FUSION_METHODS:
        raise ValueError(f"unsupported decision.cost_calibration.fusion_method={cost_cal_fusion!r}")
    accuracy_floor_delta = _to_float(decision_raw.get("accuracy_floor_delta", 0.01), 0.01)
    accuracy_floor_delta = max(0.0, min(1.0, accuracy_floor_delta))
    decision_costs_raw = decision_raw.get("costs", {})
    decision_costs = dict(decision_costs_raw) if isinstance(decision_costs_raw, Mapping) else {}
    c_fn_raw = decision_costs.get("c_fn", legacy_costs.get("c_fn"))
    c_fp_raw = decision_costs.get("c_fp", legacy_costs.get("c_fp"))
    c_h_raw = decision_costs.get("c_h", legacy_costs.get("c_h"))
    if c_fn_raw is None or c_fp_raw is None or c_h_raw is None:
        raise ValueError("decision.costs.c_fn/c_fp/c_h are required in config")
    try:
        c_fn = float(c_fn_raw)
        c_fp = float(c_fp_raw)
        c_h = float(c_h_raw)
    except Exception as exc:
        raise ValueError("decision.costs.c_fn/c_fp/c_h must be numeric") from exc

    profile_value = routing_raw.get("profile_path")
    profile_path = _resolve_path(base_dir, profile_value, "") if profile_value else None
    tie_break = str(routing_raw.get("tie_break", "agent_sequence")).strip().lower()
    if tie_break not in ROUTING_TIE_BREAKS:
        raise ValueError(f"unsupported routing.tie_break={tie_break!r}")
    langgraph_perf_guardrail_overhead = _to_float(
        routing_raw.get("langgraph_perf_guardrail_overhead", 0.05),
        0.05,
    )
    langgraph_perf_guardrail_overhead = max(0.0, min(1.0, langgraph_perf_guardrail_overhead))
    parity_tolerance = _to_float(routing_raw.get("parity_tolerance", 1e-6), 1e-6)
    parity_tolerance = max(0.0, parity_tolerance)

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
            engine=engine,
            first_agent_strategy=first_agent_strategy,
        ),
        belief=BeliefConfig(
            prior_attack_rate=prior,
            eps=eps,
            likelihood_sanity_gate=_to_bool(belief_raw.get("likelihood_sanity_gate", True), True),
        ),
        fusion=FusionConfig(method=fusion_method, agent_weights=weights),
        decision=DecisionConfig(
            policy=str(decision_raw.get("policy", "expected_cost_min")).strip().lower(),
            c_fn=c_fn,
            c_fp=c_fp,
            c_h=c_h,
            accuracy_floor_delta=accuracy_floor_delta,
            cost_calibration=CostCalibrationConfig(
                enabled=_to_bool(cost_cal_raw.get("enabled", False), False),
                mode=cost_cal_mode,
                c_fn_grid=_as_list_of_float(cost_cal_raw.get("c_fn_grid", [])),
                c_fp_grid=_as_list_of_float(cost_cal_raw.get("c_fp_grid", [])),
                c_h_grid=_as_list_of_float(cost_cal_raw.get("c_h_grid", [])),
                min_expected_gain_grid=_as_list_of_float(cost_cal_raw.get("min_expected_gain_grid", [])),
                max_agents_grid=_as_list_of_int(cost_cal_raw.get("max_agents_grid", [])),
                fusion_method=cost_cal_fusion,
            ),
        ),
        query=QueryConfig(
            policy=query_policy,
            uncertainty_threshold=uncertainty_threshold,
            max_agents=max_agents,
            min_expected_gain=min_expected_gain,
            first_agent=first_agent,
            utilization_targets=utilization_targets,
            utilization_warmup_flows=utilization_warmup_flows,
        ),
        voi=VOIConfig(
            enabled=_to_bool(voi_raw.get("enabled", True), True),
            rho=rho,
        ),
        routing=RoutingConfig(
            profile_path=profile_path,
            bin_count=max(2, _to_int(routing_raw.get("bin_count", 20), 20)),
            min_samples_per_bin=max(1, _to_int(routing_raw.get("min_samples_per_bin", 20), 20)),
            tie_break=tie_break,
            langgraph_perf_guardrail_overhead=langgraph_perf_guardrail_overhead,
            parity_tolerance=parity_tolerance,
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
