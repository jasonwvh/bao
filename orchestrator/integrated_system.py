from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from orchestrator.belief_state import BeliefStateManager
from orchestrator.config import OrchestratorConfig, load_orchestrator_config
from orchestrator.control.registry import load_registry, to_runtime_handles
from orchestrator.control.scheduler import filter_by_capability
from orchestrator.data_plane.a2a_client import A2AClient, A2AClientError
from orchestrator.data_plane.state_sqlite import SQLiteStateBackend
from orchestrator.decision import (
    DecisionCosts,
    approximate_voi,
    realized_action_cost,
    select_expected_cost_action,
)
from orchestrator.core import (
    UtilizationTarget,
    apply_fusion_update,
    build_utilization_penalties,
    classify_from_probability,
    compute_utilization_rates,
    order_candidates,
    resolve_first_agent,
)
from orchestrator.langgraph_runtime import LangGraphRuntime
from orchestrator.preprocessing import OrchestratorPreprocessor
from orchestrator.router import AdaptiveRouter

logger = logging.getLogger("orchestrator")


class IntegratedBAOSystem:
    """Config-driven BAO runtime with deterministic default and optional LangGraph engine."""

    def __init__(self, config_path: str | Path):
        self.config_obj: OrchestratorConfig = load_orchestrator_config(config_path)
        self.config = self.config_obj.raw
        self.config_path = self.config_obj.config_path

        self.state_backend = SQLiteStateBackend(self.config_obj.state.sqlite_path)
        drift_cfg = self.config.get("drift", {}) if isinstance(self.config, dict) else {}
        self.belief_manager = BeliefStateManager(
            drift_window=int(drift_cfg.get("window", 10)),
            drift_threshold=float(drift_cfg.get("threshold", 0.08)),
            backend=self.state_backend,
            eps=self.config_obj.belief.eps,
        )

        self.preprocessor = OrchestratorPreprocessor(schema_path=self.config_obj.preprocessing.schema_path)
        self.a2a = A2AClient(retries=self.config_obj.a2a.retries)

        registry = load_registry(self.config_obj.orchestration.agent_registry_path)
        self.registry_routing = dict(registry.get("routing", {}))
        self.agent_handles = to_runtime_handles(registry)

        if bool(self.registry_routing.get("require_healthy", True)):
            self._apply_health_filter()

        self.agent_sequence = self._resolve_agent_sequence()
        if not self.agent_sequence:
            raise RuntimeError("No available agents after registry/health filtering")

        self.metrics_output_path = self.config_obj.logging.jsonl_path
        self.metrics_output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.config_obj.decision.policy != "expected_cost_min":
            raise ValueError(f"unsupported decision.policy={self.config_obj.decision.policy!r}")

        self.decision_costs = DecisionCosts(
            c_fn=float(self.config_obj.decision.c_fn),
            c_fp=float(self.config_obj.decision.c_fp),
            c_h=float(self.config_obj.decision.c_h),
        )

        self.query_policy = self.config_obj.query.policy
        self.fusion_method = self.config_obj.fusion.method
        self.engine = self.config_obj.orchestration.engine
        self.router: Optional[AdaptiveRouter] = None
        if self.query_policy == "adaptive_router":
            self.router = AdaptiveRouter(
                decision_costs=self.decision_costs,
                profile_path=self.config_obj.routing.profile_path,
                min_samples_per_bin=int(self.config_obj.routing.min_samples_per_bin),
            )
        self.langgraph_runtime: Optional[LangGraphRuntime] = None
        if self.engine == "langgraph":
            self.langgraph_runtime = LangGraphRuntime(self)

        self.utilization_targets = {
            t.agent_id: UtilizationTarget(
                agent_id=t.agent_id,
                min_rate=float(t.min_rate),
                max_rate=float(t.max_rate),
                penalty_under=float(t.penalty_under),
                penalty_over=float(t.penalty_over),
            )
            for t in self.config_obj.query.utilization_targets
        }
        self.utilization_warmup_flows = int(self.config_obj.query.utilization_warmup_flows)

        self.metrics = {
            "flows_processed": 0,
            "decisions": {"accept": 0, "reject": 0, "defer": 0},
            "agent_calls": {aid: 0 for aid in self.agent_sequence},
            "total_cost": 0.0,  # Backward-compatible alias for query cost.
            "total_query_cost": 0.0,
            "total_action_cost": 0.0,
            "total_utility_cost": 0.0,
            "hitl_count": 0,
        }

    def _apply_health_filter(self) -> None:
        healthy = {}
        for aid, handle in self.agent_handles.items():
            try:
                health = self.a2a.health(handle)
                if str(health.get("status", "")).lower() != "ok":
                    continue
                caps = self.a2a.capabilities(handle)
                if str(caps.get("agent_id", "")) != aid:
                    continue
                healthy[aid] = handle
            except Exception:
                continue
        self.agent_handles = healthy

    def _resolve_agent_sequence(self) -> List[str]:
        configured = list(self.config_obj.orchestration.agent_sequence)
        if not configured:
            configured = list(self.registry_routing.get("default_agents", []))
        if not configured:
            configured = list(self.agent_handles.keys())

        seen = set()
        sequence: List[str] = []
        for aid in configured:
            if aid in seen:
                continue
            if aid not in self.agent_handles:
                continue
            seen.add(aid)
            sequence.append(aid)
        return sequence

    def _agent_weight(self, agent_id: str) -> float:
        return max(1e-6, float(self.config_obj.fusion.agent_weights.get(agent_id, 1.0)))

    def _candidate_agents(self, flow_features: Dict[str, Any]) -> List[str]:
        required_caps = list(flow_features.get("required_capabilities", []))
        candidates = filter_by_capability(self.agent_sequence, self.agent_handles, required_caps)
        first_agent = resolve_first_agent(
            candidates=list(candidates),
            agent_handles=self.agent_handles,
            strategy=self.config_obj.orchestration.first_agent_strategy,
            explicit_first_agent=self.config_obj.query.first_agent,
        )
        ordered = order_candidates(candidates, first_agent)
        max_agents = min(int(self.config_obj.query.max_agents), len(ordered))
        return ordered[:max_agents]

    def _query_decision_strict(
        self,
        p_mal: float,
        uncertainty: float,
        next_agent_cost: float,
    ) -> tuple[bool, float]:
        threshold = float(self.config_obj.query.uncertainty_threshold)

        if uncertainty <= threshold:
            return False, 0.0

        if not self.config_obj.voi.enabled:
            return True, 0.0

        voi = approximate_voi(
            p_mal=p_mal,
            costs=self.decision_costs,
            rho=float(self.config_obj.voi.rho),
        )
        return voi > float(next_agent_cost), voi

    def _build_payload(self, flow_id: str, timestamp: float, flow_features: Dict[str, Any], p_mal: float, uncertainty: float) -> Dict[str, Any]:
        return {
            "request_id": str(uuid.uuid4()),
            "flow_id": flow_id,
            "timestamp": timestamp,
            "flow_features": flow_features,
            "context": {
                "belief": {"p_mal": p_mal, "uncertainty": uncertainty},
                "requested_capabilities": list(flow_features.get("required_capabilities", [])),
                "elicit_likelihood": True,
                "seed": int(self.config_obj.orchestration.seed),
            },
        }

    def _clip_probability(self, value: Any) -> float:
        p = float(value)
        eps = self.config_obj.belief.eps
        return max(eps, min(1.0 - eps, p))

    def _query_single_agent(
        self,
        *,
        aid: str,
        belief: Any,
        features: Dict[str, Any],
        flow_id: str,
        timestamp: float,
        state: Dict[str, Any],
        queried_probabilities: Dict[str, float],
    ) -> bool:
        handle = self.agent_handles[aid]
        payload = self._build_payload(
            flow_id=flow_id,
            timestamp=timestamp,
            flow_features=features,
            p_mal=belief.get_compromise_prob(),
            uncertainty=belief.get_epistemic_uncertainty(),
        )

        try:
            output = self.a2a.infer(handle, payload)
        except A2AClientError as exc:
            state["decision_reasoning"].append(f"agent_failed={aid}:{exc}")
            return False

        p_agent = self._clip_probability((output.get("proba") or [0.5, 0.5])[1])
        output["proba"] = [1.0 - p_agent, p_agent]

        updated, fusion_note = apply_fusion_update(
            belief=belief,
            belief_manager=self.belief_manager,
            agent_output=output,
            agent_id=aid,
            p_agent=p_agent,
            queried_probabilities=queried_probabilities,
            fusion_method=self.fusion_method,
            update_mode=self.config_obj.orchestration.update_mode,
            agent_weight=self._agent_weight(aid),
            eps=self.config_obj.belief.eps,
            likelihood_sanity_gate=self.config_obj.belief.likelihood_sanity_gate,
            decision_costs=self.decision_costs,
        )

        state["agents_queried"].append(aid)
        state["agent_outputs"].append(output)
        state["iteration"] += 1
        state["cumulative_cost"] += float(handle.cost)
        state["belief_mu"] = float(updated["mu"])
        state["belief_var"] = float(updated["var"])
        state["compromise_prob"] = float(updated["compromise_prob"])
        state["epistemic_uncertainty"] = float(updated["epistemic_uncertainty"])
        state["confidence"] = max(state["compromise_prob"], 1.0 - state["compromise_prob"])

        self.metrics["agent_calls"][aid] = self.metrics["agent_calls"].get(aid, 0) + 1
        state["decision_reasoning"].append(
            f"agent={aid},p_agent={p_agent:.6f},p_post={state['compromise_prob']:.6f},h={state['epistemic_uncertainty']:.6f},fusion={fusion_note}"
        )
        return True

    async def _process_flow_deterministic(
        self,
        flow_features: Dict[str, Any],
        flow_id: str,
        timestamp: float,
        true_label: Optional[int] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        features = dict(flow_features)

        candidates = self._candidate_agents(features)
        belief = self.belief_manager.get_or_create_belief(
            flow_id=flow_id,
            prior_attack_rate=self.config_obj.belief.prior_attack_rate,
        )

        state: Dict[str, Any] = {
            "flow_id": flow_id,
            "timestamp": timestamp,
            "true_label": true_label,
            "flow_features": features,
            "agents_available": list(candidates),
            "agents_queried": [],
            "agent_outputs": [],
            "decision_reasoning": [],
            "voi_scores": {},
            "expected_gain_scores": {},
            "cumulative_cost": 0.0,
            "iteration": 0,
            "max_iterations": len(candidates),
            "drift_detected": False,
            "drift_score": 0.0,
            "hitl_context": None,
            "consensus_triggered": False,
            "consensus_result": {},
            "belief_mu": belief.mu,
            "belief_var": belief.get_variance(),
            "compromise_prob": belief.get_compromise_prob(),
            "epistemic_uncertainty": belief.get_epistemic_uncertainty(),
            "inference_time_ms": 0.0,
            "total_time_ms": 0.0,
            "confidence": max(belief.get_compromise_prob(), 1.0 - belief.get_compromise_prob()),
            "decision": None,
            "query_policy": self.query_policy,
            "fusion_method": self.fusion_method,
        }

        queried_probabilities: Dict[str, float] = {}

        if self.query_policy == "strict_cascade":
            for idx, aid in enumerate(candidates):
                ok = self._query_single_agent(
                    aid=aid,
                    belief=belief,
                    features=features,
                    flow_id=flow_id,
                    timestamp=timestamp,
                    state=state,
                    queried_probabilities=queried_probabilities,
                )
                if not ok:
                    continue

                if idx + 1 >= len(candidates):
                    break

                next_agent = candidates[idx + 1]
                should_query, voi_value = self._query_decision_strict(
                    p_mal=state["compromise_prob"],
                    uncertainty=state["epistemic_uncertainty"],
                    next_agent_cost=float(self.agent_handles[next_agent].cost),
                )
                state["voi_scores"][next_agent] = float(voi_value)
                if not should_query:
                    state["decision_reasoning"].append(
                        f"stop_after={aid},voi={voi_value:.6f},next_cost={float(self.agent_handles[next_agent].cost):.6f}"
                    )
                    break

        elif self.query_policy == "adaptive_router":
            if candidates:
                first_agent = candidates[0]
                first_ok = self._query_single_agent(
                    aid=first_agent,
                    belief=belief,
                    features=features,
                    flow_id=flow_id,
                    timestamp=timestamp,
                    state=state,
                    queried_probabilities=queried_probabilities,
                )
                last_agent = first_agent

                if not first_ok:
                    state["decision_reasoning"].append("adaptive_router:first_agent_failed")

                while self.router is not None and len(state["agents_queried"]) < len(candidates):
                    queried = set(state["agents_queried"])
                    remaining = [aid for aid in candidates if aid not in queried]
                    if not remaining:
                        break

                    source_agent = state["agents_queried"][-1] if state["agents_queried"] else last_agent
                    utilization_penalties = build_utilization_penalties(
                        candidate_agents=remaining,
                        agent_calls=self.metrics["agent_calls"],
                        flows_processed=int(self.metrics["flows_processed"]),
                        targets=self.utilization_targets,
                        warmup_flows=self.utilization_warmup_flows,
                    )
                    next_agent, scores = self.router.select_next_agent(
                        current_probability=float(state["compromise_prob"]),
                        source_agent=source_agent,
                        candidate_agents=remaining,
                        agent_handles=self.agent_handles,
                        belief_manager=self.belief_manager,
                        min_expected_gain=float(self.config_obj.query.min_expected_gain),
                        utilization_penalties=utilization_penalties,
                    )
                    state["expected_gain_scores"][source_agent] = {
                        aid: score.to_dict() for aid, score in scores.items()
                    }
                    # Keep compatibility key for external consumers.
                    state["voi_scores"].update({aid: float(score.expected_gain) for aid, score in scores.items()})

                    if next_agent is None:
                        state["decision_reasoning"].append(
                            f"adaptive_stop_after={source_agent},min_expected_gain={float(self.config_obj.query.min_expected_gain):.6f}"
                        )
                        break

                    ok = self._query_single_agent(
                        aid=next_agent,
                        belief=belief,
                        features=features,
                        flow_id=flow_id,
                        timestamp=timestamp,
                        state=state,
                        queried_probabilities=queried_probabilities,
                    )
                    if not ok:
                        state["decision_reasoning"].append(f"adaptive_skip_failed={next_agent}")
                        continue

        else:
            raise ValueError(f"unsupported query.policy={self.query_policy!r}")

        final_p = float(belief.get_compromise_prob())
        action_decision, action_costs = select_expected_cost_action(final_p, self.decision_costs)
        classification_decision = classify_from_probability(final_p)

        state["decision"] = classification_decision
        state["action_decision"] = action_decision
        state["compromise_prob"] = final_p
        state["epistemic_uncertainty"] = float(belief.get_epistemic_uncertainty())
        state["confidence"] = max(final_p, 1.0 - final_p)
        state["decision_reasoning"].append(
            f"decision={classification_decision},action_decision={action_decision},cost_accept={action_costs['accept']:.6f},cost_reject={action_costs['reject']:.6f},cost_defer={action_costs['defer']:.6f}"
        )

        if true_label is not None:
            y_true = int(true_label)
            for aid, output in zip(state["agents_queried"], state["agent_outputs"]):
                pred = 1 if float(output["proba"][1]) >= 0.5 else 0
                belief.update_agent_reliability(aid, pred, y_true)
                self.belief_manager.update_global_reliabilities(aid, pred == y_true)
                self.state_backend.update_observation_stats(
                    aid,
                    {
                        "true_label": y_true,
                        "pred": pred,
                        "proba": float(output["proba"][1]),
                        "timestamp": time.time(),
                    },
                )

        self.belief_manager.persist_belief(flow_id)

        state["inference_time_ms"] = (time.perf_counter() - t0) * 1000.0
        state["total_time_ms"] = state["inference_time_ms"]

        self.metrics["flows_processed"] += 1
        self.metrics["total_cost"] += float(state["cumulative_cost"])
        self.metrics["total_query_cost"] += float(state["cumulative_cost"])

        final_pred = 1 if final_p >= 0.5 else 0
        action_cost_value = realized_action_cost(
            decision=action_decision,
            prediction=final_pred,
            true_label=true_label,
            costs=self.decision_costs,
        )
        self.metrics["total_action_cost"] += float(action_cost_value)
        self.metrics["total_utility_cost"] += float(action_cost_value) + float(state["cumulative_cost"])

        if classification_decision not in self.metrics["decisions"]:
            self.metrics["decisions"][classification_decision] = 0
        self.metrics["decisions"][classification_decision] += 1
        if action_decision == "defer":
            self.metrics["hitl_count"] += 1

        event = {
            "flow_id": flow_id,
            "timestamp": timestamp,
            "decision": classification_decision,
            "action_decision": action_decision,
            "compromise_prob": state["compromise_prob"],
            "epistemic_uncertainty": state["epistemic_uncertainty"],
            "cumulative_cost": state["cumulative_cost"],
            "agents_queried": state["agents_queried"],
            "voi_scores": state["voi_scores"],
            "expected_gain_scores": state["expected_gain_scores"],
            "confidence": state["confidence"],
            "query_policy": self.query_policy,
            "fusion_method": self.fusion_method,
        }
        with self.metrics_output_path.open("a") as f:
            f.write(json.dumps(event) + "\n")

        return state

    async def process_flow(
        self,
        flow_features: Dict[str, Any],
        flow_id: str,
        timestamp: float,
        true_label: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.engine == "langgraph" and self.langgraph_runtime is not None:
            return await self.langgraph_runtime.process_flow(
                flow_features=flow_features,
                flow_id=flow_id,
                timestamp=timestamp,
                true_label=true_label,
            )
        return await self._process_flow_deterministic(
            flow_features=flow_features,
            flow_id=flow_id,
            timestamp=timestamp,
            true_label=true_label,
        )

    def get_system_statistics(self) -> Dict[str, Any]:
        n = max(1, self.metrics["flows_processed"])
        query_total = float(self.metrics["total_query_cost"])
        action_total = float(self.metrics["total_action_cost"])
        utility_total = float(self.metrics["total_utility_cost"])
        util_rates = compute_utilization_rates(
            agent_calls=self.metrics["agent_calls"],
            flows_processed=int(self.metrics["flows_processed"]),
        )
        return {
            "flows_processed": self.metrics["flows_processed"],
            "decision_counts": dict(self.metrics["decisions"]),
            "avg_cost_per_flow": query_total / n,
            "avg_query_cost_per_flow": query_total / n,
            "avg_utility_cost_per_flow": utility_total / n,
            "query_cost_total": query_total,
            "action_cost_total": action_total,
            "utility_cost_total": utility_total,
            "hitl_count": self.metrics["hitl_count"],
            "query_policy": self.query_policy,
            "fusion_method": self.fusion_method,
            "agent_utilization": util_rates,
        }
