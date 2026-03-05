from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class SQLiteState:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=DELETE;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
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
                CREATE TABLE IF NOT EXISTS flow_beliefs (
                    flow_id TEXT PRIMARY KEY,
                    belief_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def get_global_reliability(self, agent_id: str) -> Tuple[float, float]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT alpha, beta FROM agent_reliability WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
        if row is None:
            return (4.0, 1.0)
        return (float(row[0]), float(row[1]))

    def update_global_reliability(self, agent_id: str, correct: bool) -> Tuple[float, float]:
        alpha, beta = self.get_global_reliability(agent_id)
        if bool(correct):
            alpha += 1.0
        else:
            beta += 1.0
        now = time.time()
        with self._connect() as conn:
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

    def save_belief(self, flow_id: str, belief_json: Dict[str, Any]) -> None:
        now = time.time()
        with self._connect() as conn:
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

    def load_belief(self, flow_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT belief_json FROM flow_beliefs WHERE flow_id = ?",
                (flow_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])
