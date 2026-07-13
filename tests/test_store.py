"""tests.test_store — RunStore persistence + TTL sweep."""

from core.store import RunStore


def test_create_update_get_roundtrip(tmp_path):
    store = RunStore(db_path=tmp_path / "runs.db")
    store.create("abc", now=1000.0)
    assert store.get("abc")["status"] == "pending"

    store.update("abc", now=1001.0, status="running", state={"logs": ["a", "b"]})
    rec = store.get("abc")
    assert rec["status"] == "running"
    assert rec["state"]["logs"] == ["a", "b"]

    assert store.get("missing") is None


def test_ttl_sweeps_old_runs(tmp_path):
    store = RunStore(db_path=tmp_path / "runs.db", ttl_seconds=100)
    store.create("old", now=0.0)
    # A create far in the future triggers the sweep, evicting the stale run.
    store.create("new", now=1000.0)
    assert store.get("old") is None
    assert store.get("new") is not None


def test_update_tolerates_non_json_values(tmp_path):
    import numpy as np

    store = RunStore(db_path=tmp_path / "runs.db")
    store.create("x", now=1.0)
    # numpy scalar would break a naive json.dumps — store coerces via default=str.
    store.update("x", now=2.0, state={"metric": np.float64(0.5)})
    assert store.get("x")["state"]["metric"] in (0.5, "0.5")
