"""
init_db.py — Splat-Grid SQLite Database Initialiser
====================================================
Creates (or resets) the local task-management database.

Usage:
    python init_db.py [--db splat_state.db] [--reset]
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [INIT_DB][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────

DDL_TASKS = """
CREATE TABLE IF NOT EXISTS tasks (
    id               TEXT PRIMARY KEY,       -- UUID
    chunk_id         TEXT NOT NULL,          -- human-readable voxel label e.g. "1_2_0"
    voxel_idx        TEXT NOT NULL,          -- JSON array [ix, iy, iz]
    bbox_min         TEXT NOT NULL,          -- JSON array [x, y, z]
    bbox_max         TEXT NOT NULL,          -- JSON array [x, y, z]
    image_ids        TEXT NOT NULL,          -- JSON array of integer image ids
    status           TEXT NOT NULL           -- PENDING | IN_PROGRESS | COMPLETED | FAILED
                     CHECK(status IN ('PENDING','IN_PROGRESS','COMPLETED','FAILED')),
    assigned_worker  TEXT,                   -- worker_id or NULL
    last_heartbeat   REAL,                   -- Unix timestamp or NULL
    result_path      TEXT,                   -- local path of received .ply or NULL
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL
);
"""

DDL_WORKERS = """
CREATE TABLE IF NOT EXISTS workers (
    id           TEXT PRIMARY KEY,           -- UUID assigned on /join
    registered_at REAL NOT NULL,
    last_seen     REAL NOT NULL
);
"""

DDL_INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);",
    "CREATE INDEX IF NOT EXISTS idx_tasks_worker ON tasks(assigned_worker);",
]


def init_db(db_path: str, reset: bool = False) -> None:
    path = Path(db_path)
    if reset and path.exists():
        path.unlink()
        log.info(f"Deleted existing database: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")   # better concurrent read/write
    conn.execute("PRAGMA foreign_keys=ON;")

    conn.execute(DDL_TASKS)
    conn.execute(DDL_WORKERS)
    for idx_sql in DDL_INDICES:
        conn.execute(idx_sql)

    conn.commit()
    conn.close()
    log.info(f"Database ready: {db_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Initialise Splat-Grid SQLite database")
    p.add_argument("--db",    type=str, default="splat_state.db",
                   help="Path to the SQLite database file (default: splat_state.db)")
    p.add_argument("--reset", action="store_true",
                   help="Delete and recreate the database from scratch")
    args = p.parse_args()
    init_db(args.db, reset=args.reset)
