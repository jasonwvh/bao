from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class SQLiteStateBackend:
    def __init__(self, db_path: str | Path, busy_retries: int = 5, busy_sleep_sec: float = 0.05):
        self.db_path = str(db_path)
        self.busy_retries = busy_retries
        self.busy_sleep_sec = busy_sleep_sec
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _run(self, fn):
        last_exc = None
        for _ in range(self.busy_retries):
            conn = None
            try:
                conn = self._connect()
                result = fn(conn)
                return result
            except sqlite3.OperationalError as exc:
                last_exc = exc
                if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                    raise
                time.sleep(self.busy_sleep_sec)
            finally:
                if conn:
                    conn.close()
        if last_exc:
            raise last_exc

    def _init_db(self) -> None:
        def _create(conn: sqlite3.Connection):
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_reliability (
                    agent_id TEXT PRIMARY KEY,
                    alpha REAL NOT NULL,
                    beta REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_observation_stats (
                    agent_id TEXT PRIMARY KEY,
                    stats_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS flow_beliefs (
                    flow_id TEXT PRIMARY KEY,
                    belief_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
        self._run(_create)

    def get_global_reliability(self, agent_id: str) -> Tuple[float, float]:
        def _q(conn: sqlite3.Connection):
            row = conn.execute(
                "SELECT alpha, beta FROM agent_reliability WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
            if row is None:
                return (1.0, 1.0)
            return (float(row[0]), float(row[1]))
        return self._run(_q)

    def update_global_reliability(self, agent_id: str, correct: bool) -> Tuple[float, float]:
        def _u(conn: sqlite3.Connection):
            alpha, beta = self.get_global_reliability(agent_id)
            if correct:
                alpha += 1.0
            else:
                beta += 1.0
            now = time.time()
            conn.execute(
                """
                INSERT INTO agent_reliability(agent_id, alpha, beta, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    alpha=excluded.alpha,
                    beta=excluded.beta,
                    updated_at=excluded.updated_at
                """,
                (agent_id, alpha, beta, now),
            )
            return (alpha, beta)
        return self._run(_u)

    def get_observation_stats(self, agent_id: str) -> Dict[str, Any]:
        def _q(conn: sqlite3.Connection):
            row = conn.execute(
                "SELECT stats_json FROM agent_observation_stats WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
            if row is None:
                return {}
            return json.loads(row[0])
        return self._run(_q)

    def update_observation_stats(self, agent_id: str, sample: Dict[str, Any]) -> None:
        def _u(conn: sqlite3.Connection):
            cur = self.get_observation_stats(agent_id)
            history = cur.get("history", [])
            history.append(sample)
            if len(history) > 2000:
                history = history[-2000:]
            cur["history"] = history
            cur["n_samples"] = int(cur.get("n_samples", 0)) + 1
            now = time.time()
            conn.execute(
                """
                INSERT INTO agent_observation_stats(agent_id, stats_json, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    stats_json=excluded.stats_json,
                    updated_at=excluded.updated_at
                """,
                (agent_id, json.dumps(cur), now),
            )
        self._run(_u)

    def save_flow_belief(self, flow_id: str, belief_json: Dict[str, Any]) -> None:
        def _u(conn: sqlite3.Connection):
            now = time.time()
            conn.execute(
                """
                INSERT INTO flow_beliefs(flow_id, belief_json, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(flow_id) DO UPDATE SET
                    belief_json=excluded.belief_json,
                    updated_at=excluded.updated_at
                """,
                (flow_id, json.dumps(belief_json), now),
            )
        self._run(_u)

    def load_flow_belief(self, flow_id: str) -> Optional[Dict[str, Any]]:
        def _q(conn: sqlite3.Connection):
            row = conn.execute(
                "SELECT belief_json FROM flow_beliefs WHERE flow_id = ?",
                (flow_id,),
            ).fetchone()
            if row is None:
                return None
            return json.loads(row[0])
        return self._run(_q)
