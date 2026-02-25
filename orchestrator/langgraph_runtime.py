from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TypedDict

try:
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover - optional dependency path
    END = None  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]


logger = logging.getLogger("orchestrator.langgraph")


class RuntimeState(TypedDict, total=False):
    flow_features: Dict[str, Any]
    flow_id: str
    timestamp: float
    true_label: Optional[int]
    result: Dict[str, Any]


class LangGraphRuntime:
    """Optional LangGraph runner that preserves deterministic semantics."""

    def __init__(self, system: Any):
        self.system = system
        self._app = None
        if StateGraph is None:
            logger.warning("LangGraph not available; falling back to deterministic engine")
            return
        try:
            self._app = self._build_graph()
        except Exception as exc:
            logger.warning("Failed to initialize LangGraph runtime (%s); using deterministic engine", exc)
            self._app = None

    def _build_graph(self):
        workflow = StateGraph(RuntimeState)

        async def run_deterministic(state: RuntimeState) -> RuntimeState:
            result = await self.system._process_flow_deterministic(
                flow_features=dict(state.get("flow_features") or {}),
                flow_id=str(state.get("flow_id")),
                timestamp=float(state.get("timestamp")),
                true_label=state.get("true_label"),
            )
            return {"result": result}

        workflow.add_node("run_deterministic", run_deterministic)
        workflow.set_entry_point("run_deterministic")
        workflow.add_edge("run_deterministic", END)
        return workflow.compile()

    async def process_flow(
        self,
        *,
        flow_features: Dict[str, Any],
        flow_id: str,
        timestamp: float,
        true_label: Optional[int],
    ) -> Dict[str, Any]:
        if self._app is None:
            return await self.system._process_flow_deterministic(
                flow_features=flow_features,
                flow_id=flow_id,
                timestamp=timestamp,
                true_label=true_label,
            )
        try:
            out = await self._app.ainvoke(
                {
                    "flow_features": flow_features,
                    "flow_id": flow_id,
                    "timestamp": timestamp,
                    "true_label": true_label,
                }
            )
            result = out.get("result")
            if isinstance(result, dict):
                return result
            return await self.system._process_flow_deterministic(
                flow_features=flow_features,
                flow_id=flow_id,
                timestamp=timestamp,
                true_label=true_label,
            )
        except Exception as exc:
            logger.warning("LangGraph execution failed (%s); falling back to deterministic run", exc)
            return await self.system._process_flow_deterministic(
                flow_features=flow_features,
                flow_id=flow_id,
                timestamp=timestamp,
                true_label=true_label,
            )
