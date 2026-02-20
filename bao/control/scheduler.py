from __future__ import annotations

from typing import Dict, List

from bao.types import AgentRuntimeHandle


def select_enabled_agents(handles: Dict[str, AgentRuntimeHandle], default_agents: List[str]) -> List[str]:
    if default_agents:
        return [a for a in default_agents if a in handles]
    return list(handles.keys())


def filter_by_capability(
    candidate_agents: List[str],
    handles: Dict[str, AgentRuntimeHandle],
    required_capabilities: List[str],
) -> List[str]:
    if not required_capabilities:
        return candidate_agents

    required = set(required_capabilities)
    out: List[str] = []
    for aid in candidate_agents:
        caps = set(handles[aid].capabilities)
        if required.issubset(caps):
            out.append(aid)
    return out
