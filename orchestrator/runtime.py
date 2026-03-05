from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.common.streaming import derive_stream_id
from orchestrator.a2a import A2AClient, A2AClientError, AgentHandle, load_registry
from orchestrator.belief import BeliefManager, reliability_weight_from_beta_params
from orchestrator.config import OrchestratorConfig, load_config
from orchestrator.data import DataAdapter
from orchestrator.decisioning import (
    DecisionCosts,
    approximate_voi,
    expected_cost_reduction,
    realized_action_cost,
    select_decision,
)
from orchestrator.state import SQLiteState


class BAORuntime:
    def __init__(
        self,
        config: str | OrchestratorConfig,
        state_sqlite_path: Optional[str | Path] = None,
    ):
        if isinstance(config, OrchestratorConfig):
            self.config = config
        else:
            self.config = load_config(str(config))

        sqlite_path = (
            Path(state_sqlite_path).resolve()
            if state_sqlite_path is not None
            else self.config.state.sqlite_path
        )
        self.state = SQLiteState(sqlite_path)
        self.beliefs = BeliefManager(self.state, eps=self.config.belief.eps)
        self.data = DataAdapter(schema_path=self.config.preprocessing.schema_path)
        self.a2a = A2AClient(retries=self.config.a2a.retries)
        self.session_id = str(uuid.uuid4())

        self.agent_handles: Dict[str, AgentHandle] = load_registry(self.config.orchestration.agent_registry_path)
        self.agent_sequence = self._resolve_agent_sequence()
        if not self.agent_sequence:
            raise RuntimeError("No enabled agents found in registry")

        self.costs = DecisionCosts(
            c_fn=float(self.config.decision.c_fn),
            c_fp=float(self.config.decision.c_fp),
            c_h=float(self.config.decision.c_h),
        )

        self.metrics = {
            "flows_processed": 0,
            "defer_count": 0,
            "agent_calls": {aid: 0 for aid in self.agent_sequence},
            "routing_selection_counts": {
                "escalate": 0,
                "stop": 0,
            },
            "routing_expected_net_gain_total": 0.0,
            "routing_expected_net_gain_count": 0,
            "query_cost_total": 0.0,
            "action_cost_total": 0.0,
            "utility_cost_total": 0.0,
            "warnings": {},
        }

    def _resolve_agent_sequence(self) -> List[str]:
        sequence = list(self.config.orchestration.agent_sequence)
        if not sequence:
            sequence = list(self.agent_handles.keys())

        filtered = [aid for aid in sequence if aid in self.agent_handles]
        seen = set()
        ordered = []
        for aid in filtered:
            if aid in seen:
                continue
            ordered.append(aid)
            seen.add(aid)

        first = self.config.query.first_agent
        if first and first in ordered:
            ordered = [first] + [aid for aid in ordered if aid != first]

        max_agents = min(max(1, int(self.config.query.max_agents)), len(ordered))
        return ordered[:max_agents]

    def _agent_weight(self, agent_id: str, agent_epistemic: float) -> float:
        base = float(self.config.fusion.agent_weights.get(agent_id, 1.0))
        gamma = float(self.config.fusion.uncertainty_weight_gamma)
        floor = float(self.config.fusion.weight_floor)
        ep = max(0.0, min(1.0, float(agent_epistemic)))
        return max(1e-6, (base * ((1.0 - ep) ** gamma)) + floor)

    def _combined_uncertainty_nats(self, belief_entropy: float, agent_epistemic: float) -> float:
        be = max(0.0, min(math.log(2.0), float(belief_entropy)))
        ae = max(0.0, min(1.0, float(agent_epistemic))) * math.log(2.0)
        return max(be, ae)

    def _record_warning(self, code: str, message: str) -> None:
        if not bool(self.config.metrics.warnings_enabled):
            return
        key = str(code).strip() or "runtime_warning"
        warnings_map = self.metrics["warnings"]
        current = warnings_map.get(key)
        if isinstance(current, dict):
            current["count"] = int(current.get("count", 0)) + 1
            current["message"] = str(current.get("message") or message)
        else:
            warnings_map[key] = {"code": key, "message": str(message), "count": 1}

    def _extract_likelihoods(self, output: Dict[str, Any]) -> Optional[tuple[float, float]]:
        likelihoods = dict(output.get("likelihoods") or {})
        p_attack = likelihoods.get("p_obs_given_attack")
        p_clean = likelihoods.get("p_obs_given_clean")
        try:
            p1 = float(p_attack)
            p0 = float(p_clean)
        except Exception:
            return None
        if not math.isfinite(p1) or not math.isfinite(p0):
            return None
        if p1 <= 0.0 or p0 <= 0.0:
            return None
        return (p1, p0)

    def _next_agent_reliability(self, agent_id: str) -> float:
        return max(0.0, min(1.0, float(self.beliefs.get_global_reliability(agent_id))))

    def _expected_voi_value(self, p_mal: float, combined_uncertainty: float, next_agent_id: str) -> float:
        if not bool(self.config.voi.enabled):
            return 0.0
        mode = str(self.config.voi.mode).strip().lower()
        if mode == "legacy_approx":
            return float(approximate_voi(p_mal, self.costs, rho=float(self.config.voi.rho)))

        next_rel = self._next_agent_reliability(next_agent_id)
        epistemic = max(0.0, min(1.0, float(combined_uncertainty) / math.log(2.0)))
        return float(
            expected_cost_reduction(
                p_mal=p_mal,
                costs=self.costs,
                reliability=next_rel,
                epistemic_uncertainty=epistemic,
                rho=float(self.config.voi.rho),
            )
        )

    def _should_escalate(
        self,
        *,
        p_mal: float,
        combined_uncertainty: float,
        next_agent_cost: float,
        next_agent_id: str,
    ) -> tuple[bool, float]:
        if float(combined_uncertainty) <= float(self.config.query.uncertainty_threshold):
            return False, float("-inf")

        voi_value = self._expected_voi_value(
            p_mal=float(p_mal),
            combined_uncertainty=float(combined_uncertainty),
            next_agent_id=next_agent_id,
        )
        expected_gain = float(voi_value) - float(next_agent_cost)
        return expected_gain >= float(self.config.voi.min_net_gain), float(expected_gain)

    def _build_payload(self, flow_id: str, timestamp: float, flow_features: Dict[str, Any], p_mal: float, uncertainty: float) -> Dict[str, Any]:
        stream_id = derive_stream_id(flow_features=flow_features, flow_id=flow_id)
        return {
            "request_id": str(uuid.uuid4()),
            "flow_id": flow_id,
            "timestamp": timestamp,
            "flow_features": flow_features,
            "context": {
                "belief": {"p_mal": float(p_mal), "uncertainty": float(uncertainty)},
                "requested_capabilities": list(flow_features.get("required_capabilities", [])),
                "seed": int(self.config.orchestration.seed),
                "stream_id": stream_id,
                "session_id": self.session_id,
                "elicit_likelihood": True,
            },
        }

    def _clip_probability(self, value: Any) -> float:
        p = float(value)
        eps = self.config.belief.eps
        return max(eps, min(1.0 - eps, p))

    def process_flow(
        self,
        *,
        flow_features: Dict[str, Any],
        flow_id: str,
        timestamp: float,
        true_label: Optional[int] = None,
    ) -> Dict[str, Any]:
        features = self.data.transform(dict(flow_features))

        belief = self.beliefs.get_or_create(
            flow_id=flow_id,
            prior_attack_rate=self.config.belief.prior_attack_rate,
        )

        agents_queried: List[str] = []
        outputs: List[Dict[str, Any]] = []
        cumulative_cost = 0.0
        last_epistemic = 0.5
        last_expected_net_gain = float("-inf")
        combined_uncertainty = self._combined_uncertainty_nats(belief.entropy(), last_epistemic)
        for idx, agent_id in enumerate(self.agent_sequence):
            handle = self.agent_handles[agent_id]
            payload = self._build_payload(
                flow_id=flow_id,
                timestamp=timestamp,
                flow_features=features,
                p_mal=belief.probability(),
                uncertainty=belief.entropy(),
            )

            try:
                output = self.a2a.infer(handle, payload)
            except A2AClientError as exc:
                self._record_warning(
                    "transport_failure",
                    f"A2A infer failed for {agent_id}: {exc}",
                )
                continue

            p_agent = self._clip_probability((output.get("proba") or [0.5, 0.5])[1])
            out_uncertainty = dict(output.get("uncertainty") or {})
            last_epistemic = max(0.0, min(1.0, float(out_uncertainty.get("epistemic", 0.5))))

            if str(self.config.belief.update_mode).strip().lower() == "likelihood_ratio":
                likelihoods = self._extract_likelihoods(output)
                if likelihoods is not None:
                    alpha, beta = self.beliefs.get_global_reliability_params(agent_id)
                    k_i = reliability_weight_from_beta_params(
                        alpha=alpha,
                        beta=beta,
                        reliability_strength=float(self.config.belief.reliability_strength),
                    )
                    belief.update_from_likelihood_ratio(
                        p_obs_given_attack=likelihoods[0],
                        p_obs_given_clean=likelihoods[1],
                        k=k_i,
                    )
                else:
                    self._record_warning(
                        "missing_or_invalid_likelihoods",
                        f"{agent_id} response missing valid likelihoods; fallback to probability pooling",
                    )
                    weight = self._agent_weight(agent_id, last_epistemic)
                    belief.update_from_agent_probability(p_agent, weight)
            else:
                weight = self._agent_weight(agent_id, last_epistemic)
                belief.update_from_agent_probability(p_agent, weight)
            combined_uncertainty = self._combined_uncertainty_nats(belief.entropy(), last_epistemic)

            agents_queried.append(agent_id)
            outputs.append(
                {
                    "agent_id": agent_id,
                    "p_agent": float(p_agent),
                    "epistemic": float(last_epistemic),
                }
            )
            cumulative_cost += float(handle.cost)
            self.metrics["agent_calls"][agent_id] = int(self.metrics["agent_calls"].get(agent_id, 0)) + 1

            if idx + 1 >= len(self.agent_sequence):
                continue

            next_agent = self.agent_sequence[idx + 1]
            should_escalate, expected_gain = self._should_escalate(
                p_mal=belief.probability(),
                combined_uncertainty=combined_uncertainty,
                next_agent_cost=float(self.agent_handles[next_agent].cost),
                next_agent_id=next_agent,
            )
            last_expected_net_gain = float(expected_gain)
            if math.isfinite(float(expected_gain)):
                self.metrics["routing_expected_net_gain_total"] += float(expected_gain)
                self.metrics["routing_expected_net_gain_count"] += 1
            if should_escalate:
                self.metrics["routing_selection_counts"]["escalate"] += 1
                continue

            self.metrics["routing_selection_counts"]["stop"] += 1
            break

        final_p = float(belief.probability())
        decision, _ = select_decision(final_p, self.costs)

        exhausted = len(agents_queried) >= len(self.agent_sequence)
        if (
            bool(self.config.decision.defer_enabled)
            and ((not bool(self.config.decision.defer_require_all_agents_exhausted)) or exhausted)
            and float(combined_uncertainty) >= float(self.config.decision.defer_uncertainty_threshold)
            and abs(float(final_p) - 0.5) <= float(self.config.decision.defer_margin_from_half)
        ):
            decision = "defer"

        if true_label is not None:
            y_true = int(true_label)
            for out in outputs:
                pred = 1 if float(out["p_agent"]) >= 0.5 else 0
                self.beliefs.update_global_reliability(out["agent_id"], pred == y_true)

        self.beliefs.persist(flow_id)

        action_cost = realized_action_cost(
            decision=decision,
            true_label=true_label,
            costs=self.costs,
        )

        self.metrics["flows_processed"] += 1
        self.metrics["query_cost_total"] += float(cumulative_cost)
        self.metrics["action_cost_total"] += float(action_cost)
        self.metrics["utility_cost_total"] += float(cumulative_cost) + float(action_cost)
        if decision == "defer":
            self.metrics["defer_count"] += 1

        return {
            "decision": str(decision),
            "compromise_prob": float(final_p),
            "epistemic_uncertainty": float(belief.entropy()),
            "combined_uncertainty": float(combined_uncertainty),
            "expected_net_gain": float(last_expected_net_gain),
            "agents_queried": list(agents_queried),
            "cumulative_cost": float(cumulative_cost),
        }

    def get_summary(self) -> Dict[str, Any]:
        n = max(1, int(self.metrics["flows_processed"]))
        agent_utilization = {
            aid: float(calls) / float(n)
            for aid, calls in self.metrics["agent_calls"].items()
        }
        routing_counts = dict(self.metrics["routing_selection_counts"])
        routing_total = max(1, int(routing_counts.get("escalate", 0) + routing_counts.get("stop", 0)))
        gain_count = max(1, int(self.metrics["routing_expected_net_gain_count"]))
        warnings = sorted(
            (dict(v) for v in dict(self.metrics["warnings"]).values()),
            key=lambda x: str(x.get("code", "")),
        )
        return {
            "flows_processed": int(self.metrics["flows_processed"]),
            "defer_count": int(self.metrics["defer_count"]),
            "defer_rate": float(self.metrics["defer_count"]) / float(n),
            "avg_query_cost_per_flow": float(self.metrics["query_cost_total"]) / float(n),
            "avg_utility_cost_per_flow": float(self.metrics["utility_cost_total"]) / float(n),
            "query_cost_total": float(self.metrics["query_cost_total"]),
            "action_cost_total": float(self.metrics["action_cost_total"]),
            "utility_cost_total": float(self.metrics["utility_cost_total"]),
            "agent_utilization": agent_utilization,
            "routing_selection_counts": routing_counts,
            "routing": {
                "escalation_rate": float(routing_counts.get("escalate", 0)) / float(routing_total),
                "avg_expected_net_gain": float(self.metrics["routing_expected_net_gain_total"]) / float(gain_count),
            },
            "warnings": warnings,
            "a2a": self.a2a.metadata(),
        }
