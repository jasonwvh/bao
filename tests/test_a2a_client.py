from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from bao.data_plane.a2a_client import A2AClient
from bao.types import AgentRuntimeHandle


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/a2a/health":
            self._send({"status": "ok", "agent_id": "agent_x", "version": "v1"})
        elif self.path == "/a2a/capabilities":
            self._send({"agent_id": "agent_x", "capabilities": ["x"], "cost": 1.0})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/a2a/infer":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        self._send(
            {
                "agent_id": "agent_x",
                "proba": [0.7, 0.3],
                "prediction": {"label": "benign", "probability": 0.3},
                "uncertainty": {"epistemic": 0.1, "aleatoric": 0.2, "total_entropy": 0.3},
                "cost": 1.0,
                "latency_ms": 2.0,
                "metadata": {},
            }
        )

    def _send(self, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        return


def test_a2a_client_http_contract():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        port = server.server_port
        client = A2AClient()
        handle = AgentRuntimeHandle(
            agent_id="agent_x",
            endpoint=f"http://127.0.0.1:{port}",
            transport="http-json",
            timeout_ms=500,
            cost=1.0,
            capabilities=["x"],
            health_path="/a2a/health",
            infer_path="/a2a/infer",
            capabilities_path="/a2a/capabilities",
            meta={},
        )

        h = client.health(handle)
        c = client.capabilities(handle)
        r = client.infer(
            handle,
            {
                "request_id": "x",
                "flow_id": "f1",
                "timestamp": 0.0,
                "flow_features": {},
                "context": {},
            },
        )

        assert h["status"] == "ok"
        assert c["agent_id"] == "agent_x"
        assert r["agent_id"] == "agent_x"
    finally:
        server.shutdown()
