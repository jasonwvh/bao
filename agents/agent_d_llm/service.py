from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

from .agent import LLMAgent


AGENT = LLMAgent(cost=float(os.getenv("AGENT_COST", "20.0")))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/a2a/health":
            return self._send({"status": "ok", "agent_id": AGENT.agent_id, "version": "v1"})
        if self.path == "/a2a/capabilities":
            return self._send({"agent_id": AGENT.agent_id, "capabilities": ["llm_reasoning"], "cost": AGENT.cost})
        self.send_error(404)

    def do_POST(self):
        if self.path != "/a2a/infer":
            return self.send_error(404)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body.decode("utf-8") or "{}")
        feats = payload.get("flow_features", {})
        seed = payload.get("context", {}).get("seed")
        out = AGENT.predict_with_uncertainty(feats, seed=seed)
        return self._send(out)

    def _send(self, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        return


def main():
    port = int(os.getenv("PORT", "8084"))
    server = HTTPServer(("0.0.0.0", port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
