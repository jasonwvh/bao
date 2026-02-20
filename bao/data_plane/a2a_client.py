from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from bao.types import A2AInferRequest, AgentRuntimeHandle


class A2AClientError(RuntimeError):
    pass


class A2AClient:
    def __init__(self, retries: int = 0):
        self.retries = retries

    def health(self, handle: AgentRuntimeHandle) -> Dict[str, Any]:
        url = f"{handle.endpoint.rstrip('/')}{handle.health_path}"
        data = self._http_json("GET", url, timeout_ms=handle.timeout_ms)
        if "status" not in data:
            raise A2AClientError(f"Invalid health response from {url}: missing 'status'")
        return data

    def capabilities(self, handle: AgentRuntimeHandle) -> Dict[str, Any]:
        url = f"{handle.endpoint.rstrip('/')}{handle.capabilities_path}"
        data = self._http_json("GET", url, timeout_ms=handle.timeout_ms)
        if "agent_id" not in data or "capabilities" not in data:
            raise A2AClientError(f"Invalid capabilities response from {url}")
        return data

    def infer(self, handle: AgentRuntimeHandle, payload: A2AInferRequest) -> Dict[str, Any]:
        required = {"request_id", "flow_id", "timestamp", "flow_features", "context"}
        missing = required - set(payload.keys())
        if missing:
            raise A2AClientError(f"Infer payload missing keys: {sorted(missing)}")

        url = f"{handle.endpoint.rstrip('/')}{handle.infer_path}"
        data = self._http_json("POST", url, payload=payload, timeout_ms=handle.timeout_ms)

        required_resp = {"agent_id", "proba", "prediction", "uncertainty", "cost"}
        missing_resp = required_resp - set(data.keys())
        if missing_resp:
            raise A2AClientError(f"Invalid infer response from {url}: missing {sorted(missing_resp)}")
        return data

    def _http_json(
        self,
        method: str,
        url: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout_ms: int = 1000,
    ) -> Dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url=url, method=method, data=body)
        req.add_header("Content-Type", "application/json")

        attempt = 0
        while True:
            try:
                with urllib.request.urlopen(req, timeout=timeout_ms / 1000.0) as resp:
                    data = resp.read().decode("utf-8")
                    return json.loads(data) if data else {}
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt >= self.retries:
                    raise A2AClientError(f"A2A request failed: {method} {url}: {exc}") from exc
                attempt += 1
