from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml


@dataclass(frozen=True)
class AgentHandle:
    agent_id: str
    endpoint: str
    timeout_ms: int
    cost: float
    capabilities: List[str]
    health_path: str
    infer_path: str
    capabilities_path: str
    agent_card_path: str


class A2AClientError(RuntimeError):
    pass


def _load_official_a2a_sdk() -> tuple[bool, Optional[type]]:
    """Load official A2A SDK bindings if available."""
    try:
        mod = importlib.import_module("a2a.client")
        resolver_cls = getattr(mod, "A2ACardResolver", None)
        if resolver_cls is None:
            return False, None
        return True, resolver_cls
    except Exception:
        return False, None


OFFICIAL_A2A_SDK_AVAILABLE, OFFICIAL_A2A_CARD_RESOLVER = _load_official_a2a_sdk()


def load_registry(path: str | Path) -> Dict[str, AgentHandle]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    raw = yaml.safe_load(p.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("Registry YAML must be a mapping")

    handles: Dict[str, AgentHandle] = {}
    for item in raw.get("agents", []) or []:
        if not isinstance(item, dict):
            continue
        if not bool(item.get("enabled", False)):
            continue

        agent_id = str(item.get("id", "")).strip()
        if not agent_id:
            continue

        handles[agent_id] = AgentHandle(
            agent_id=agent_id,
            endpoint=str(item.get("endpoint", "")).rstrip("/"),
            timeout_ms=int(item.get("timeout_ms", 1500)),
            cost=float(item.get("cost", 1.0)),
            capabilities=[str(x) for x in (item.get("capabilities") or [])],
            health_path=str(item.get("health_path", "/a2a/health")),
            infer_path=str(item.get("infer_path", "/a2a/infer")),
            capabilities_path=str(item.get("capabilities_path", "/a2a/capabilities")),
            agent_card_path=str(item.get("agent_card_path", "/.well-known/agent-card.json")),
        )
    return handles


def calibrate_handle_costs(
    handles: Dict[str, AgentHandle],
    *,
    human_review_cost: float,
    false_positive_cost: float,
    max_fraction_of_action_cost: float = 0.1,
) -> Dict[str, AgentHandle]:
    if not handles:
        return {}

    reference = max(
        1e-6,
        min(float(human_review_cost), float(false_positive_cost)) * max(1e-6, float(max_fraction_of_action_cost)),
    )
    raw_max = max(max(1e-6, float(handle.cost)) for handle in handles.values())
    return {
        agent_id: replace(handle, cost=(max(1e-6, float(handle.cost)) / raw_max) * reference)
        for agent_id, handle in handles.items()
    }


class A2AClient:
    """A2A transport adapter.

    Uses official A2A SDK card resolution when available while preserving this
    project's HTTP+JSON inference contract.
    """

    def __init__(self, retries: int = 0):
        self.retries = int(max(0, retries))
        self.official_sdk_available = OFFICIAL_A2A_SDK_AVAILABLE
        self._sdk_card_resolver_cls = OFFICIAL_A2A_CARD_RESOLVER
        self._sdk_card_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self._sdk_resolution_attempts = 0
        self._sdk_resolution_success = 0

    def _request_json(self, method: str, url: str, timeout_ms: int, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        last_err: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                with httpx.Client(timeout=float(timeout_ms) / 1000.0) as client:
                    if method == "GET":
                        resp = client.get(url)
                    else:
                        resp = client.post(url, json=payload)
                    resp.raise_for_status()
                    return resp.json() if resp.content else {}
            except Exception as exc:
                last_err = exc
        raise A2AClientError(f"A2A request failed: {method} {url}: {last_err}")

    def _sdk_cache_key(self, handle: AgentHandle) -> str:
        return f"{handle.endpoint.rstrip('/')}{handle.agent_card_path}"

    def _resolve_card_with_sdk(self, handle: AgentHandle) -> Optional[Dict[str, Any]]:
        if not self.official_sdk_available or self._sdk_card_resolver_cls is None:
            return None

        cache_key = self._sdk_cache_key(handle)
        if cache_key in self._sdk_card_cache:
            return self._sdk_card_cache[cache_key]

        async def _resolve() -> Dict[str, Any]:
            timeout_s = max(0.001, float(handle.timeout_ms) / 1000.0)
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                resolver = self._sdk_card_resolver_cls(
                    httpx_client=client,
                    base_url=handle.endpoint,
                    agent_card_path=handle.agent_card_path,
                )
                card = await resolver.get_agent_card()
                if hasattr(card, "model_dump"):
                    return card.model_dump(exclude_none=True)  # pydantic v2
                if hasattr(card, "dict"):
                    return card.dict()  # pydantic v1 fallback
                return {}

        self._sdk_resolution_attempts += 1
        try:
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None

            if running_loop and running_loop.is_running():
                loop = asyncio.new_event_loop()
                try:
                    card_payload = loop.run_until_complete(_resolve())
                finally:
                    loop.close()
            else:
                card_payload = asyncio.run(_resolve())
            self._sdk_resolution_success += 1
            self._sdk_card_cache[cache_key] = card_payload
            return card_payload
        except Exception:
            self._sdk_card_cache[cache_key] = None
            return None

    def _capabilities_from_card(self, card_payload: Dict[str, Any]) -> List[str]:
        caps: List[str] = []
        for skill in card_payload.get("skills") or []:
            if not isinstance(skill, dict):
                continue
            sid = str(skill.get("id", "")).strip()
            if sid:
                caps.append(sid)
        if caps:
            return caps

        raw_caps = card_payload.get("capabilities")
        if isinstance(raw_caps, dict):
            return sorted([str(k) for k, v in raw_caps.items() if bool(v)])
        return []

    def health(self, handle: AgentHandle) -> Dict[str, Any]:
        url = f"{handle.endpoint}{handle.health_path}"
        data = self._request_json("GET", url, handle.timeout_ms)
        if "status" not in data:
            raise A2AClientError(f"Invalid health response from {url}")
        return data

    def capabilities(self, handle: AgentHandle) -> Dict[str, Any]:
        sdk_card = self._resolve_card_with_sdk(handle)
        if isinstance(sdk_card, dict):
            capabilities = self._capabilities_from_card(sdk_card)
            return {
                "agent_id": handle.agent_id,
                "capabilities": capabilities or list(handle.capabilities),
                "cost": float(handle.cost),
                "metadata": {
                    "source": "official_a2a_sdk",
                    "card_url": sdk_card.get("url"),
                    "card_name": sdk_card.get("name"),
                    "protocol_version": sdk_card.get("protocol_version"),
                    "preferred_transport": sdk_card.get("preferred_transport"),
                },
            }

        url = f"{handle.endpoint}{handle.capabilities_path}"
        data = self._request_json("GET", url, handle.timeout_ms)
        if "agent_id" not in data:
            raise A2AClientError(f"Invalid capabilities response from {url}")
        return data

    def infer(self, handle: AgentHandle, payload: Dict[str, Any]) -> Dict[str, Any]:
        required = {"request_id", "flow_id", "timestamp", "flow_features", "context"}
        missing = required - set(payload.keys())
        if missing:
            raise A2AClientError(f"Infer payload missing keys: {sorted(missing)}")

        url = f"{handle.endpoint}{handle.infer_path}"
        data = self._request_json("POST", url, handle.timeout_ms, payload=payload)
        required_resp = {"agent_id", "proba", "prediction", "uncertainty", "cost"}
        missing_resp = required_resp - set(data.keys())
        if missing_resp:
            raise A2AClientError(f"Invalid infer response from {url}: missing {sorted(missing_resp)}")
        return data

    def metadata(self) -> Dict[str, Any]:
        return {
            "official_a2a_sdk_available": bool(self.official_sdk_available),
            "official_a2a_sdk_active": bool(self._sdk_resolution_success > 0),
            "sdk_card_resolution_attempts": int(self._sdk_resolution_attempts),
            "sdk_card_resolution_success": int(self._sdk_resolution_success),
            "transport": "httpx_json_with_sdk_card_resolution",
        }
