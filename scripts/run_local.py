#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


AGENTS = [
    {"id": "agent_a", "module": "agents.agent_a_lightweight.service", "port": 8081},
    {"id": "agent_b", "module": "agents.agent_b_deep.service", "port": 8082},
    {"id": "agent_c", "module": "agents.agent_c_contextual.service", "port": 8083},
    {"id": "agent_d", "module": "agents.agent_d_llm.service", "port": 8084},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run BAO agents locally without Docker")
    p.add_argument("--registry", default=str(REPO_ROOT / "config" / "agents.local.yaml"), help="Path to local agents registry YAML")
    p.add_argument("--startup-timeout", type=float, default=20.0, help="Seconds to wait for all agents to become healthy")
    p.add_argument("--startup-only", action="store_true", help="Start agents, wait for health, then exit (terminates agents)")
    p.add_argument("--run-replay", default="", help="Optional dataset path to run replay after startup")
    p.add_argument("--replay-config", default=str(REPO_ROOT / "config" / "bao_config.json"), help="Base BAO config for replay")
    p.add_argument("--replay-output-dir", default=str(REPO_ROOT / "artifacts" / "replay"), help="Replay output directory")
    p.add_argument("--seed", type=int, default=7, help="Replay seed")
    return p.parse_args()


def _wait_for_health(port: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/a2a/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.2)
    return False


def _terminate_all(processes: list[subprocess.Popen]) -> None:
    for p in processes:
        if p.poll() is None:
            p.terminate()
    for p in processes:
        try:
            p.wait(timeout=5)
        except Exception:
            if p.poll() is None:
                p.kill()


def _run_replay(args: argparse.Namespace, registry_path: Path) -> int:
    cfg = json.loads(Path(args.replay_config).read_text())
    cfg.setdefault("orchestration", {})["agent_registry_path"] = str(registry_path)
    cfg.setdefault("orchestration", {})["seed"] = int(args.seed)

    output_dir = Path(args.replay_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_cfg = output_dir / "effective_bao_config.local.json"
    effective_cfg.write_text(json.dumps(cfg, indent=2))

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_replay.py"),
        "--dataset",
        args.run_replay,
        "--config",
        str(effective_cfg),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(args.seed),
    ]
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def main() -> int:
    args = parse_args()

    env_base = os.environ.copy()
    processes: list[subprocess.Popen] = []

    try:
        for spec in AGENTS:
            env = env_base.copy()
            env["PORT"] = str(spec["port"])
            p = subprocess.Popen(
                [sys.executable, "-m", spec["module"]],
                cwd=str(REPO_ROOT),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            processes.append(p)

        unhealthy = []
        for spec in AGENTS:
            ok = _wait_for_health(spec["port"], args.startup_timeout)
            if not ok:
                unhealthy.append(spec["id"])

        if unhealthy:
            print(f"Failed to start healthy agents: {unhealthy}")
            return 1

        print("All agents are healthy on localhost:")
        for spec in AGENTS:
            print(f"- {spec['id']}: http://127.0.0.1:{spec['port']}")

        registry_path = Path(args.registry).resolve()
        print(f"Use registry: {registry_path}")

        if args.run_replay:
            rc = _run_replay(args, registry_path)
            return rc

        if args.startup_only:
            return 0

        print("Press Ctrl+C to stop all local agent processes.")
        signal.pause()
        return 0

    except KeyboardInterrupt:
        return 0
    finally:
        _terminate_all(processes)


if __name__ == "__main__":
    raise SystemExit(main())
