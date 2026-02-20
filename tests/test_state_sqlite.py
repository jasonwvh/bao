from __future__ import annotations

from pathlib import Path

from bao.data_plane.state_sqlite import SQLiteStateBackend


def test_state_sqlite_reliability_and_flow_belief_roundtrip(tmp_path: Path):
    db = tmp_path / "state.sqlite"
    backend = SQLiteStateBackend(db)

    a, b = backend.get_global_reliability("agent_a")
    assert (a, b) == (1.0, 1.0)

    a2, b2 = backend.update_global_reliability("agent_a", True)
    assert a2 > a and b2 == b

    backend.save_flow_belief("flow_1", {"mu": 0.2, "var": 1.0})
    belief = backend.load_flow_belief("flow_1")
    assert belief == {"mu": 0.2, "var": 1.0}


def test_state_sqlite_observation_stats_upsert(tmp_path: Path):
    db = tmp_path / "state.sqlite"
    backend = SQLiteStateBackend(db)

    backend.update_observation_stats("agent_a", {"x": 1})
    backend.update_observation_stats("agent_a", {"x": 2})

    stats = backend.get_observation_stats("agent_a")
    assert stats.get("n_samples") == 2
    assert len(stats.get("history", [])) == 2
