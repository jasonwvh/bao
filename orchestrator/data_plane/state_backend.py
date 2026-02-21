from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple


class StateBackend(Protocol):
    def get_global_reliability(self, agent_id: str) -> Tuple[float, float]:
        ...

    def update_global_reliability(self, agent_id: str, correct: bool) -> Tuple[float, float]:
        ...

    def get_observation_stats(self, agent_id: str) -> Dict[str, Any]:
        ...

    def update_observation_stats(self, agent_id: str, sample: Dict[str, Any]) -> None:
        ...

    def save_flow_belief(self, flow_id: str, belief_json: Dict[str, Any]) -> None:
        ...

    def load_flow_belief(self, flow_id: str) -> Optional[Dict[str, Any]]:
        ...
