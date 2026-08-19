"""
core.store — persistent run store for pipeline state.

Replaces the process-local dict that leaked memory and vanished on restart.
Backed by SQLite (stdlib, zero extra deps); each run is one JSON record keyed
by pipeline_id. Old runs are swept on write via a TTL.

The interface is deliberately tiny (create/update/get) so it can be swapped for
Redis/Postgres later without touching call sites.

ponytail: SQLite + a global write lock. Fine for a single container; swap the
backing store if you scale to many instances or high write throughput. On an
ephemeral filesystem (e.g. Cloud Run) the DB resets on cold start — that removes
the leak and gives in-session persistence, which is the goal here.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional


class RunStore:
    def __init__(self, db_path: str | Path = "runs.db", ttl_seconds: int = 24 * 3600):
        self.db_path = str(db_path)
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._init()

    def _init(self) -> None:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS runs ("
                "  id TEXT PRIMARY KEY,"
                "  created_at REAL,"
                "  updated_at REAL,"
                "  status TEXT,"
                "  record TEXT"
                ")"
            )

    def create(self, run_id: str, now: float) -> None:
        record = {"status": "pending", "state": None, "error": None}
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs (id, created_at, updated_at, status, record)"
                " VALUES (?, ?, ?, ?, ?)",
                (run_id, now, now, "pending", json.dumps(record, default=str)),
            )
        self._sweep(now)

    def update(self, run_id: str, now: float, **fields: Any) -> None:
        """
        Merge `fields` into the run's record (JSON-serialisable values only).

        Upserts. The old version built a default record when the row was absent
        and then ran an UPDATE that matched nothing, so a write against a swept
        or unknown id was discarded without a word — including the "failed"
        write that is the last thing a dying run does.
        """
        with self._lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT created_at, record FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            created_at = row[0] if row else now
            record = json.loads(row[1]) if row else {"status": "pending", "state": None}
            record.update(fields)
            conn.execute(
                "INSERT INTO runs (id, created_at, updated_at, status, record)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(id) DO UPDATE SET"
                "   updated_at = excluded.updated_at,"
                "   status = excluded.status,"
                "   record = excluded.record",
                (
                    run_id,
                    created_at,
                    now,
                    record.get("status", "pending"),
                    json.dumps(record, default=str),
                ),
            )

    def get(self, run_id: str) -> Optional[dict]:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT record FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _sweep(self, now: float) -> None:
        cutoff = now - self.ttl_seconds
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM runs WHERE created_at < ?", (cutoff,))
