from __future__ import annotations

import json
import logging
import math
import os
import re
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(name)s] %(message)s')
logger = logging.getLogger("llm")


class LLM:
    def __init__(
        self,
        cost: float = 20.0,
        model: str = "qwen3",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 10.0,
    ):
        self.agent_id = "llm"
        self.cost = float(cost)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            raise ValueError("empty LLM response")

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("no JSON object in response")
        return json.loads(match.group(0))

    def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url=url, method="POST", data=data)
        req.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = resp.read().decode("utf-8")
            decoded = json.loads(body)
            return self._extract_json(str(decoded.get("response", "")))

    def health(self) -> Dict[str, Any]:
        url = f"{self.base_url}/api/tags"
        req = urllib.request.Request(url=url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                if int(resp.status) == 200:
                    return {"status": "ok", "ollama": self.base_url, "model": self.model}
        except Exception:
            return {"status": "degraded", "ollama": self.base_url, "model": self.model}
        return {"status": "degraded", "ollama": self.base_url, "model": self.model}

    def predict_with_uncertainty(self, flow_features: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        prompt = (
            "You are a network intrusion triage model for UNSW-NB15-like flow features. "
            "Return only JSON with keys: p_mal (0..1), uncertainty (0..1), reason (string). "
            f"Features: {json.dumps(flow_features, sort_keys=True)}"
        )

        metadata: Dict[str, Any] = {"model": self.model, "ollama": self.base_url}
        try:
            parsed = self._call_ollama(prompt)
            p = float(parsed.get("p_mal", 0.5))
            u = float(parsed.get("uncertainty", 0.8))
            reason = str(parsed.get("reason", ""))
            metadata["reason"] = reason
        except (ValueError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            p = 0.5
            u = 1.0
            metadata["error"] = str(exc)

        p = float(max(0.001, min(0.999, p)))
        u = float(max(0.0, min(1.0, u)))
        entropy = -(p * math.log(max(p, 1e-9)) + (1.0 - p) * math.log(max(1.0 - p, 1e-9)))

        return {
            "proba": [1.0 - p, p],
            "prediction": {"label": "malicious" if p >= 0.5 else "benign", "probability": p},
            "uncertainty": {
                "epistemic": float(max(u, entropy)),
                "aleatoric": float(entropy),
                "total_entropy": float(max(u, entropy)),
            },
            "cost": self.cost,
            "agent_id": self.agent_id,
            "metadata": metadata,
        }


AGENT = LLM(
    cost=float(os.getenv("AGENT_COST", "20.0")),
    model=os.getenv("OLLAMA_MODEL", "qwen3"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    timeout_s=float(os.getenv("OLLAMA_TIMEOUT_S", "10.0")),
)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/a2a/health":
            health = AGENT.health()
            return self._send(
                {
                    "status": health.get("status", "degraded"),
                    "agent_id": AGENT.agent_id,
                    "version": "v1",
                    "ollama": health.get("ollama"),
                    "model": health.get("model"),
                }
            )
        if self.path == "/a2a/capabilities":
            return self._send(
                {
                    "agent_id": AGENT.agent_id,
                    "capabilities": ["llm_reasoning", "semantic_triage", "qwen3"],
                    "cost": AGENT.cost,
                }
            )
        self.send_error(404)

    def do_POST(self):
        if self.path != "/a2a/infer":
            return self.send_error(404)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body.decode("utf-8") or "{}")
        
        logger.info("[INPUT] flow_id=%s, features_keys=%s", 
                   payload.get("flow_id"), 
                   list(payload.get("flow_features", {}).keys()))
        
        feats = payload.get("flow_features", {})
        seed = payload.get("context", {}).get("seed")
        out = AGENT.predict_with_uncertainty(feats, seed=seed)
        
        logger.info("[OUTPUT] flow_id=%s, proba=%s, prediction=%s", 
                   payload.get("flow_id"), 
                   out.get("proba"), 
                   out.get("prediction"))
        
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
