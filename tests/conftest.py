from __future__ import annotations

import json
import math
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict

import pytest
import yaml


def _predict(agent_id: str, features: Dict[str, float], seed: int = 0):
    packet_count = float(features.get("packet_count", 0.0))
    byte_count = float(features.get("byte_count", 0.0))
    duration = float(features.get("flow_duration", 0.0))
    src_port = float(features.get("src_port", 0.0))
    dst_port = float(features.get("dst_port", 0.0))

    if agent_id == "agent_a":
        z = (
            0.75 * math.log1p(packet_count)
            + 0.95 * math.log1p(byte_count / 100.0)
            + 0.25 * math.log1p(duration)
            + (0.7 if dst_port in (22.0, 23.0, 3389.0) else -0.15)
            - 5.8
        )
        p = 1.0 / (1.0 + math.exp(-z))
        epistemic = 0.18
        cost = 1.0
    elif agent_id == "agent_b":
        z = (
            1.10 * math.log1p(packet_count)
            + 1.20 * math.log1p(byte_count / 120.0)
            + 0.95 * math.log1p(duration)
            + (0.95 if dst_port in (22.0, 23.0, 3389.0, 4444.0) else 0.0)
            + (0.45 if src_port < 1024.0 else 0.0)
            - 7.8
        )
        rnd = random.Random(seed + int(packet_count) + 17)
        z += rnd.uniform(-0.55, 0.55)
        p = 1.0 / (1.0 + math.exp(-z))
        epistemic = 0.12
        cost = 5.0
    elif agent_id == "agent_c":
        p = 0.5
        epistemic = 1.0
        cost = 10.0
    else:
        p = 0.5
        epistemic = 1.0
        cost = 20.0

    p = float(max(0.001, min(0.999, p)))
    entropy = -(p * math.log(max(p, 1e-9)) + (1 - p) * math.log(max(1 - p, 1e-9)))
    return {
        "agent_id": agent_id,
        "proba": [1.0 - p, p],
        "prediction": {"label": "malicious" if p > 0.5 else "benign", "probability": p},
        "uncertainty": {
            "epistemic": float(epistemic),
            "aleatoric": float(entropy),
            "total_entropy": float(entropy),
        },
        "cost": cost,
        "latency_ms": 1.0,
        "metadata": {},
    }


def _make_handler(agent_id: str, capabilities: list[str], cost: float):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/a2a/health":
                return self._send({"status": "ok", "agent_id": agent_id, "version": "v1"})
            if self.path == "/a2a/capabilities":
                return self._send({"agent_id": agent_id, "capabilities": capabilities, "cost": cost})
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            if self.path != "/a2a/infer":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8") or "{}")
            feats = payload.get("flow_features", {})
            seed = int(payload.get("context", {}).get("seed", 0))
            return self._send(_predict(agent_id, feats, seed=seed))

        def _send(self, payload):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            return

    return Handler


def _start_server(agent_id: str, capabilities: list[str], cost: float):
    server = HTTPServer(("127.0.0.1", 0), _make_handler(agent_id, capabilities, cost))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.fixture()
def a2a_stack(tmp_path: Path):
    specs = {
        "agent_a": (["flow_tabular", "baseline"], 1.0),
        "agent_b": (["flow_tabular", "uncertainty", "deep_inspection"], 5.0),
        "agent_c": (["graph_context"], 10.0),
        "agent_d": (["llm_reasoning"], 20.0),
    }

    running = {}
    try:
        for aid, (caps, cost) in specs.items():
            server, thread = _start_server(aid, caps, cost)
            running[aid] = {"server": server, "thread": thread, "url": f"http://127.0.0.1:{server.server_port}"}

        registry_path = tmp_path / "agents.yaml"
        registry = {
            "version": "v1",
            "agents": [
                {
                    "id": "agent_a",
                    "enabled": True,
                    "endpoint": running["agent_a"]["url"],
                    "transport": "http-json",
                    "timeout_ms": 500,
                    "cost": 1.0,
                    "capabilities": ["flow_tabular", "baseline"],
                    "health_path": "/a2a/health",
                    "infer_path": "/a2a/infer",
                    "capabilities_path": "/a2a/capabilities",
                    "meta": {},
                },
                {
                    "id": "agent_b",
                    "enabled": True,
                    "endpoint": running["agent_b"]["url"],
                    "transport": "http-json",
                    "timeout_ms": 500,
                    "cost": 5.0,
                    "capabilities": ["flow_tabular", "uncertainty", "deep_inspection"],
                    "health_path": "/a2a/health",
                    "infer_path": "/a2a/infer",
                    "capabilities_path": "/a2a/capabilities",
                    "meta": {},
                },
                {
                    "id": "agent_c",
                    "enabled": True,
                    "endpoint": running["agent_c"]["url"],
                    "transport": "http-json",
                    "timeout_ms": 500,
                    "cost": 10.0,
                    "capabilities": ["graph_context"],
                    "health_path": "/a2a/health",
                    "infer_path": "/a2a/infer",
                    "capabilities_path": "/a2a/capabilities",
                    "meta": {},
                },
                {
                    "id": "agent_d",
                    "enabled": True,
                    "endpoint": running["agent_d"]["url"],
                    "transport": "http-json",
                    "timeout_ms": 500,
                    "cost": 20.0,
                    "capabilities": ["llm_reasoning"],
                    "health_path": "/a2a/health",
                    "infer_path": "/a2a/infer",
                    "capabilities_path": "/a2a/capabilities",
                    "meta": {},
                },
            ],
            "routing": {
                "default_agents": ["agent_a", "agent_b"],
                "require_healthy": True,
                "max_parallel": 2,
                "fallback_strategy": "skip_unhealthy",
            },
        }
        registry_path.write_text(yaml.safe_dump(registry, sort_keys=False))

        yield {
            "registry_path": registry_path,
            "urls": {k: v["url"] for k, v in running.items()},
        }
    finally:
        for v in running.values():
            v["server"].shutdown()
