from __future__ import annotations

import copy
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from bao.belief_state import BeliefStateManager
from bao.control.policy import weighted_consensus
from bao.control.registry import load_registry, to_runtime_handles
from bao.control.scheduler import filter_by_capability, select_enabled_agents
from bao.data_plane.a2a_client import A2AClient, A2AClientError
from bao.data_plane.state_sqlite import SQLiteStateBackend
from bao.observation_calibrator import ObservationModelCalibrator
from bao.types import DecisionThresholds, FullBAOState
from bao.voi_router import VOIRouter


class IntegratedBAOSystem:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.config = self._load_config(config_path)

        thresholds_cfg = self.config.get("thresholds", {})
        self.thresholds = DecisionThresholds(
            p_accept=float(thresholds_cfg.get("p_accept", 0.3)),
            p_reject=float(thresholds_cfg.get("p_reject", 0.7)),
            uncertainty=float(thresholds_cfg.get("uncertainty", 0.6)),
        )

        state_db = self.config.get("state", {}).get("sqlite_path", "artifacts/state/bao_state.sqlite")
        self.state_backend = SQLiteStateBackend(state_db)

        drift_cfg = self.config.get("drift", {})
        self.belief_manager = BeliefStateManager(
            drift_window=int(drift_cfg.get("window", 10)),
            drift_threshold=float(drift_cfg.get("threshold", 0.08)),
            backend=self.state_backend,
        )
        self.calibrator = ObservationModelCalibrator()

        registry_path = self._resolve_registry_path()
        registry = load_registry(registry_path)
        self.registry_routing = registry.get("routing", {})
        self.agent_handles = to_runtime_handles(registry)

        self.a2a = A2AClient(retries=int(self.config.get("a2a", {}).get("retries", 0)))

        if bool(self.registry_routing.get("require_healthy", True)):
            self._apply_health_filter()

        default_agents = list(self.registry_routing.get("default_agents", []))
        self.default_agents = select_enabled_agents(self.agent_handles, default_agents)
        if bool(self.registry_routing.get("require_healthy", True)):
            missing_defaults = [aid for aid in default_agents if aid not in self.default_agents]
            if missing_defaults:
                raise RuntimeError(
                    f"Default agents unavailable/unhealthy at startup: {missing_defaults}"
                )
        if not self.default_agents:
            raise RuntimeError("No available agents after registry/health initialization")

        costs = self.config.get("costs", {})
        self.voi_router = VOIRouter(
            agents={aid: self.agent_handles[aid] for aid in self.default_agents},
            observation_models=self.calibrator.models,
            c_fn=float(costs.get("c_fn", 100.0)),
            c_fp=float(costs.get("c_fp", 1.0)),
            c_h=float(costs.get("c_h", 5.0)),
            use_surrogate=bool(self.config.get("voi", {}).get("use_surrogate", True)),
            allow_exact=bool(self.config.get("voi", {}).get("allow_exact", False)),
        )

        self.metrics = {
            "flows_processed": 0,
            "decisions": {"accept": 0, "reject": 0, "defer": 0},
            "agent_calls": {aid: 0 for aid in self.default_agents},
            "total_cost": 0.0,
            "drift_count": 0,
            "hitl_count": 0,
        }

        self.metrics_output_path = Path(self.config.get("logging", {}).get("jsonl_path", "artifacts/replay/flows.jsonl"))
        self.metrics_output_path.parent.mkdir(parents=True, exist_ok=True)

        self.mlflow_enabled = bool(self.config.get("logging", {}).get("enable_mlflow", False))
        self._mlflow = None
        if self.mlflow_enabled:
            try:
                import mlflow

                self._mlflow = mlflow
            except Exception:
                self._mlflow = None

        try:
            from langgraph.graph import END, StateGraph
        except Exception as exc:
            raise RuntimeError("langgraph is required to run IntegratedBAOSystem; install langgraph>=0.2") from exc

        self._END = END
        self._StateGraph = StateGraph
        self.graph, self.graph_nodes = self._build_graph()

    def _resolve_registry_path(self) -> Path:
        configured = self.config.get("orchestration", {}).get("agent_registry_path")
        if configured:
            p = Path(configured)
            if not p.is_absolute():
                p = (self.config_path.parent / p).resolve()
            return p
        return (Path(__file__).resolve().parents[2] / "config" / "agents.yaml").resolve()

    def _apply_health_filter(self) -> None:
        healthy = {}
        for aid, handle in self.agent_handles.items():
            try:
                health = self.a2a.health(handle)
                if str(health.get("status", "")).lower() == "ok":
                    caps = self.a2a.capabilities(handle)
                    if str(caps.get("agent_id", "")) == aid:
                        healthy[aid] = handle
            except Exception:
                continue
        self.agent_handles = healthy

    def _load_config(self, config_path: str | Path) -> Dict[str, Any]:
        with open(config_path) as f:
            return json.load(f)

    def _build_graph(self):
        workflow = self._StateGraph(FullBAOState)
        nodes = [
            "initialize",
            "load_belief",
            "compute_voi",
            "select_agent",
            "call_agent",
            "update_belief",
            "check_drift",
            "calibrate",
            "check_uncertainty",
            "consensus",
            "make_decision",
            "defer_hitl",
            "execute_action",
            "collect_feedback",
            "update_models",
            "log_metrics",
        ]

        workflow.add_node("initialize", self._initialize_state)
        workflow.add_node("load_belief", self._load_belief)
        workflow.add_node("compute_voi", self._compute_voi)
        workflow.add_node("select_agent", self._select_agent)
        workflow.add_node("call_agent", self._call_agent)
        workflow.add_node("update_belief", self._update_belief)
        workflow.add_node("check_drift", self._check_drift)
        workflow.add_node("calibrate", self._calibrate)
        workflow.add_node("check_uncertainty", self._check_uncertainty)
        workflow.add_node("consensus", self._consensus)
        workflow.add_node("make_decision", self._make_decision)
        workflow.add_node("defer_hitl", self._defer_hitl)
        workflow.add_node("execute_action", self._execute_action)
        workflow.add_node("collect_feedback", self._collect_feedback)
        workflow.add_node("update_models", self._update_models)
        workflow.add_node("log_metrics", self._log_metrics)

        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "load_belief")
        workflow.add_edge("load_belief", "compute_voi")

        workflow.add_conditional_edges(
            "compute_voi",
            self._route_after_voi,
            {"query": "select_agent", "check": "check_uncertainty"},
        )

        workflow.add_edge("select_agent", "call_agent")
        workflow.add_edge("call_agent", "update_belief")
        workflow.add_edge("update_belief", "check_drift")

        workflow.add_conditional_edges(
            "check_drift",
            self._route_after_drift,
            {"calibrate": "calibrate", "loop": "compute_voi", "check": "check_uncertainty"},
        )

        workflow.add_edge("calibrate", "compute_voi")

        workflow.add_conditional_edges(
            "check_uncertainty",
            self._route_after_uncertainty,
            {"consensus": "consensus", "decide": "make_decision"},
        )

        workflow.add_edge("consensus", "make_decision")

        workflow.add_conditional_edges(
            "make_decision",
            lambda s: "defer" if s["decision"] == "defer" else "act",
            {"defer": "defer_hitl", "act": "execute_action"},
        )

        workflow.add_edge("defer_hitl", "collect_feedback")
        workflow.add_edge("execute_action", "collect_feedback")
        workflow.add_edge("collect_feedback", "update_models")
        workflow.add_edge("update_models", "log_metrics")
        workflow.add_edge("log_metrics", self._END)

        return workflow.compile(), nodes

    async def _initialize_state(self, state: FullBAOState) -> FullBAOState:
        state["agents_available"] = list(self.default_agents)
        state["agents_queried"] = []
        state["agent_outputs"] = []
        state["voi_scores"] = {}
        state["selected_agent"] = None
        state["selected_voi"] = None
        state["iteration"] = 0
        state["max_iterations"] = int(self.config.get("orchestration", {}).get("max_iterations", 2))

        state["belief_mu"] = 0.0
        state["belief_var"] = 1.0
        state["compromise_prob"] = 0.5
        state["epistemic_uncertainty"] = 0.69314718056
        state["agent_reliabilities"] = {}

        state["drift_detected"] = False
        state["drift_score"] = 0.0
        state["needs_calibration"] = False
        state["consensus_triggered"] = False
        state["consensus_result"] = {"agreement": 1.0, "probability": state["compromise_prob"], "participants": []}

        state["decision"] = None
        state["decision_reasoning"] = []
        state["hitl_context"] = None

        state["inference_time_ms"] = 0.0
        state["cumulative_cost"] = 0.0
        state["confidence"] = 0.0
        state["total_time_ms"] = 0.0
        return state

    async def _load_belief(self, state: FullBAOState) -> FullBAOState:
        belief = self.belief_manager.get_or_create_belief(state["flow_id"])
        state["belief_mu"] = belief.mu
        state["belief_var"] = belief.get_variance()
        state["compromise_prob"] = belief.get_compromise_prob()
        state["epistemic_uncertainty"] = belief.get_epistemic_uncertainty()
        state["agent_reliabilities"] = {
            aid: self.belief_manager.get_global_reliability(aid) for aid in state["agents_available"]
        }
        return state

    async def _compute_voi(self, state: FullBAOState) -> FullBAOState:
        t0 = time.perf_counter()
        belief = self.belief_manager.get_belief(state["flow_id"])
        if belief is None:
            belief = self.belief_manager.get_or_create_belief(state["flow_id"])

        requested_caps = list(state["flow_features"].get("required_capabilities", []))
        candidate_agents = filter_by_capability(state["agents_available"], self.agent_handles, requested_caps)

        _, _, scores = self.voi_router.select_best_agent(
            belief_state=belief,
            flow_features=state["flow_features"],
            queried_agents=[a for a in state["agents_queried"] if a in candidate_agents],
        )
        # Ensure only currently available/capable agents are retained
        state["voi_scores"] = {aid: s for aid, s in scores.items() if aid in candidate_agents}
        state["inference_time_ms"] += (time.perf_counter() - t0) * 1000.0
        return state

    def _route_after_voi(self, state: FullBAOState) -> str:
        if state["iteration"] == 0 and not state["agents_queried"] and state["agents_available"]:
            return "query"
        if state["iteration"] >= state["max_iterations"]:
            return "check"
        if not state["voi_scores"]:
            return "check"
        if max(state["voi_scores"].values()) > 0:
            return "query"
        return "check"

    async def _select_agent(self, state: FullBAOState) -> FullBAOState:
        if not state["voi_scores"]:
            state["selected_agent"] = None
            state["selected_voi"] = None
            return state
        best = max(state["voi_scores"], key=state["voi_scores"].get)
        state["selected_agent"] = best
        state["selected_voi"] = state["voi_scores"][best]
        state["decision_reasoning"].append(f"selected={best},voi={state['selected_voi']:.4f}")
        return state

    async def _call_agent(self, state: FullBAOState) -> FullBAOState:
        t0 = time.perf_counter()
        aid = state["selected_agent"]
        if aid is None or aid not in self.agent_handles:
            return state

        handle = self.agent_handles[aid]
        payload = {
            "request_id": str(uuid.uuid4()),
            "flow_id": state["flow_id"],
            "timestamp": state["timestamp"],
            "flow_features": state["flow_features"],
            "context": {
                "belief": {
                    "p_mal": state["compromise_prob"],
                    "uncertainty": state["epistemic_uncertainty"],
                },
                "requested_capabilities": state["flow_features"].get("required_capabilities", []),
                "seed": int(self.config.get("orchestration", {}).get("seed", 0)),
            },
        }

        try:
            output = self.a2a.infer(handle, payload)
        except A2AClientError:
            if aid in state["agents_available"]:
                state["agents_available"] = [x for x in state["agents_available"] if x != aid]
            state["decision_reasoning"].append(f"agent_failed={aid}")
            return state

        state["agent_outputs"].append(output)
        state["agents_queried"].append(aid)
        state["iteration"] += 1
        state["cumulative_cost"] += float(output.get("cost", handle.cost))

        if aid in self.metrics["agent_calls"]:
            self.metrics["agent_calls"][aid] += 1
        else:
            self.metrics["agent_calls"][aid] = 1

        state["inference_time_ms"] += (time.perf_counter() - t0) * 1000.0
        return state

    async def _update_belief(self, state: FullBAOState) -> FullBAOState:
        if not state["agent_outputs"]:
            return state
        t0 = time.perf_counter()

        belief = self.belief_manager.get_or_create_belief(state["flow_id"])
        output = dict(state["agent_outputs"][-1])
        agent_id = state["agents_queried"][-1]

        model = self.calibrator.get_or_create_model(agent_id)
        p_raw = float(output.get("proba", [0.5, 0.5])[1])
        p_cal = model.calibrate_proba(p_raw)
        output["proba"] = [1.0 - p_cal, p_cal]
        state["agent_outputs"][-1] = output

        updated = belief.variational_update(
            output,
            agent_id=agent_id,
            learning_rate=float(self.config.get("orchestration", {}).get("learning_rate", 0.25)),
            use_natural_gradient=False,
        )
        self.belief_manager.persist_belief(state["flow_id"])

        state["belief_mu"] = float(updated["mu"])
        state["belief_var"] = float(updated["var"])
        state["compromise_prob"] = float(updated["compromise_prob"])
        state["epistemic_uncertainty"] = float(updated["epistemic_uncertainty"])
        state["confidence"] = float(max(state["compromise_prob"], 1.0 - state["compromise_prob"]))
        state["decision_reasoning"].append(
            f"update:p={state['compromise_prob']:.4f},h={state['epistemic_uncertainty']:.4f}"
        )

        state["inference_time_ms"] += (time.perf_counter() - t0) * 1000.0
        return state

    async def _check_drift(self, state: FullBAOState) -> FullBAOState:
        belief = self.belief_manager.get_or_create_belief(state["flow_id"])
        stats = belief.detect_drift()
        state["drift_detected"] = bool(stats.drift_detected)
        state["drift_score"] = float(stats.drift_score)
        state["needs_calibration"] = bool(stats.drift_detected)

        if stats.drift_detected:
            self.metrics["drift_count"] += 1
            state["decision_reasoning"].append(f"drift={stats.drift_score:.4f}")
        return state

    def _route_after_drift(self, state: FullBAOState) -> str:
        if state["needs_calibration"]:
            return "calibrate"
        if state["iteration"] < state["max_iterations"]:
            if state.get("voi_scores") and max(state["voi_scores"].values()) > 0:
                return "loop"
        return "check"

    async def _calibrate(self, state: FullBAOState) -> FullBAOState:
        for aid in state["agents_queried"]:
            model = self.calibrator.get_or_create_model(aid)
            model.refit_from_experience()
        state["decision_reasoning"].append("calibrated")
        return state

    async def _check_uncertainty(self, state: FullBAOState) -> FullBAOState:
        return state

    def _route_after_uncertainty(self, state: FullBAOState) -> str:
        thresh = float(self.thresholds.uncertainty)
        if (
            state["epistemic_uncertainty"] > thresh
            and not state["consensus_triggered"]
            and len(state["agent_outputs"]) >= 1
        ):
            return "consensus"
        return "decide"

    async def _consensus(self, state: FullBAOState) -> FullBAOState:
        reliability_lookup = {
            aid: self.belief_manager.get_global_reliability(aid) for aid in state["agents_available"]
        }
        result = weighted_consensus(state["agent_outputs"], reliability_lookup)

        belief = self.belief_manager.get_or_create_belief(state["flow_id"])
        belief.set_compromise_prob(float(result["probability"]))
        self.belief_manager.persist_belief(state["flow_id"])

        state["compromise_prob"] = float(result["probability"])
        state["epistemic_uncertainty"] = belief.get_epistemic_uncertainty()
        state["consensus_triggered"] = True
        state["consensus_result"] = result
        state["decision_reasoning"].append(
            f"consensus:p={state['compromise_prob']:.4f},agreement={result['agreement']:.4f}"
        )
        return state

    async def _make_decision(self, state: FullBAOState) -> FullBAOState:
        belief = self.belief_manager.get_or_create_belief(state["flow_id"])
        p = belief.get_compromise_prob()
        h = belief.get_epistemic_uncertainty()

        local_thresholds = copy.copy(self.thresholds)
        if state["consensus_triggered"]:
            agreement = float(state["consensus_result"].get("agreement", 1.0))
            if agreement < 0.7:
                local_thresholds = DecisionThresholds(
                    p_accept=self.thresholds.p_accept,
                    p_reject=self.thresholds.p_reject,
                    uncertainty=self.thresholds.uncertainty * 0.85,
                )

        decision = local_thresholds.decide(p, h)
        state["decision"] = decision
        state["confidence"] = max(p, 1.0 - p)
        state["decision_reasoning"].append(f"decision={decision},p={p:.4f},h={h:.4f}")
        return state

    async def _defer_hitl(self, state: FullBAOState) -> FullBAOState:
        state["hitl_context"] = {
            "flow_id": state["flow_id"],
            "compromise_prob": state["compromise_prob"],
            "epistemic_uncertainty": state["epistemic_uncertainty"],
            "agents_queried": state["agents_queried"],
            "agent_outputs": state["agent_outputs"],
            "reasoning": state["decision_reasoning"],
        }
        self.metrics["hitl_count"] += 1
        return state

    async def _execute_action(self, state: FullBAOState) -> FullBAOState:
        return state

    async def _collect_feedback(self, state: FullBAOState) -> FullBAOState:
        return state

    async def _update_models(self, state: FullBAOState) -> FullBAOState:
        true_label = state.get("true_label")
        if true_label is None:
            return state

        belief = self.belief_manager.get_or_create_belief(state["flow_id"])
        for aid, out in zip(state["agents_queried"], state["agent_outputs"]):
            pred = 1 if out["proba"][1] >= 0.5 else 0
            belief.update_agent_reliability(aid, pred, int(true_label))
            self.belief_manager.update_global_reliabilities(aid, pred == int(true_label))
            self.calibrator.online_update(aid, out, int(true_label))
            self.state_backend.update_observation_stats(
                aid,
                {
                    "true_label": int(true_label),
                    "pred": pred,
                    "proba": float(out["proba"][1]),
                    "timestamp": time.time(),
                },
            )
        self.belief_manager.persist_belief(state["flow_id"])
        return state

    async def _log_metrics(self, state: FullBAOState) -> FullBAOState:
        state["total_time_ms"] = state["inference_time_ms"]

        decision = state.get("decision") or "defer"
        self.metrics["flows_processed"] += 1
        self.metrics["decisions"][decision] += 1
        self.metrics["total_cost"] += state["cumulative_cost"]

        event = {
            "flow_id": state["flow_id"],
            "timestamp": state.get("timestamp", time.time()),
            "decision": state["decision"],
            "compromise_prob": state["compromise_prob"],
            "epistemic_uncertainty": state["epistemic_uncertainty"],
            "cumulative_cost": state["cumulative_cost"],
            "agents_queried": state["agents_queried"],
            "voi_scores": state.get("voi_scores", {}),
            "confidence": state["confidence"],
            "drift_detected": state["drift_detected"],
        }
        with open(self.metrics_output_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        if self._mlflow is not None:
            self._mlflow.log_metrics(
                {
                    "compromise_prob": state["compromise_prob"],
                    "epistemic_uncertainty": state["epistemic_uncertainty"],
                    "cumulative_cost": state["cumulative_cost"],
                }
            )
        return state

    async def process_flow(
        self,
        flow_features: Dict[str, Any],
        flow_id: str,
        timestamp: float,
        true_label: Optional[int] = None,
    ) -> Dict[str, Any]:
        init_state: FullBAOState = {
            "flow_id": flow_id,
            "flow_features": flow_features,
            "timestamp": timestamp,
            "true_label": true_label,
        }
        return await self.graph.ainvoke(init_state)

    def get_system_statistics(self) -> Dict[str, Any]:
        n = max(1, self.metrics["flows_processed"])
        return {
            "flows_processed": self.metrics["flows_processed"],
            "decision_counts": self.metrics["decisions"],
            "avg_cost_per_flow": self.metrics["total_cost"] / n,
            "hitl_count": self.metrics["hitl_count"],
            "drift_count": self.metrics["drift_count"],
            "agent_utilization": {k: v / n for k, v in self.metrics["agent_calls"].items()},
            "observation_model_stats": self.calibrator.get_all_statistics(),
        }
