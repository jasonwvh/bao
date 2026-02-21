from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from orchestrator.types import AgentRegistryEntry, AgentRuntimeHandle


class RegistryValidationError(ValueError):
    pass


REQUIRED_AGENT_KEYS = {
    "id",
    "enabled",
    "endpoint",
    "transport",
    "timeout_ms",
    "cost",
    "capabilities",
    "health_path",
    "infer_path",
}


ALLOWED_FALLBACK = {"skip_unhealthy", "fail", "retry"}


def load_registry(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    raw = yaml.safe_load(p.read_text()) or {}
    if not isinstance(raw, dict):
        raise RegistryValidationError("Registry YAML must be a mapping")

    version = str(raw.get("version", "")).strip()
    if version != "v1":
        raise RegistryValidationError("Registry version must be 'v1'")

    agents = raw.get("agents", [])
    routing = raw.get("routing", {})
    if not isinstance(agents, list):
        raise RegistryValidationError("agents must be a list")
    if not isinstance(routing, dict):
        raise RegistryValidationError("routing must be a mapping")

    default_agents = routing.get("default_agents", [])
    if not isinstance(default_agents, list) or len(default_agents) == 0:
        raise RegistryValidationError("routing.default_agents must be a non-empty list")
    if not isinstance(routing.get("require_healthy", True), bool):
        raise RegistryValidationError("routing.require_healthy must be boolean")

    fallback = str(routing.get("fallback_strategy", "skip_unhealthy"))
    if fallback not in ALLOWED_FALLBACK:
        raise RegistryValidationError(
            f"routing.fallback_strategy must be one of {sorted(ALLOWED_FALLBACK)}"
        )

    entries: List[AgentRegistryEntry] = []
    for item in agents:
        if not isinstance(item, dict):
            raise RegistryValidationError("each agent entry must be a mapping")
        missing = REQUIRED_AGENT_KEYS - set(item.keys())
        if missing:
            raise RegistryValidationError(f"agent entry missing keys: {sorted(missing)}")

        endpoint = str(item["endpoint"])
        transport = str(item["transport"])
        if endpoint.startswith("local://"):
            raise RegistryValidationError("local:// endpoints are not allowed in remote-only mode")
        if not (endpoint.startswith("http://") or endpoint.startswith("https://")):
            raise RegistryValidationError(
                f"agent endpoint must be http:// or https://, got: {endpoint}"
            )
        if transport != "http-json":
            raise RegistryValidationError("transport must be 'http-json'")

        entry = AgentRegistryEntry(
            id=str(item["id"]),
            enabled=bool(item["enabled"]),
            endpoint=endpoint,
            transport=transport,
            timeout_ms=int(item["timeout_ms"]),
            cost=float(item["cost"]),
            capabilities=[str(x) for x in item.get("capabilities", [])],
            health_path=str(item.get("health_path", "/a2a/health")),
            infer_path=str(item.get("infer_path", "/a2a/infer")),
            capabilities_path=str(item.get("capabilities_path", "/a2a/capabilities")),
            meta=dict(item.get("meta", {})),
        )
        entries.append(entry)

    return {"version": version, "agents": entries, "routing": routing}


def to_runtime_handles(registry: Dict[str, Any]) -> Dict[str, AgentRuntimeHandle]:
    out: Dict[str, AgentRuntimeHandle] = {}
    for e in registry["agents"]:
        if not e.enabled:
            continue
        out[e.id] = AgentRuntimeHandle(
            agent_id=e.id,
            endpoint=e.endpoint,
            transport=e.transport,
            timeout_ms=e.timeout_ms,
            cost=e.cost,
            capabilities=list(e.capabilities),
            health_path=e.health_path,
            infer_path=e.infer_path,
            capabilities_path=e.capabilities_path,
            meta=dict(e.meta),
        )
    return out
