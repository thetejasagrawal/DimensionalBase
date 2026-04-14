"""
Schema migration framework for DimensionalBase SQLite storage.

Each migration is a function that takes a sqlite3.Connection and applies a
schema change.  Migrations are numbered sequentially and applied automatically
on database open.

The ``schema_version`` table tracks which migrations have been applied.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Callable, Dict

logger = logging.getLogger("dimensionalbase.migrations")


# ---------------------------------------------------------------------------
# Migration registry  (version -> callable)
# ---------------------------------------------------------------------------

def _migration_001_initial_schema(conn: sqlite3.Connection) -> None:
    """Create the initial knowledge table and indexes.

    For databases that already have the table this is a no-op thanks to
    ``IF NOT EXISTS``.
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS knowledge (
            id          TEXT PRIMARY KEY,
            path        TEXT NOT NULL UNIQUE,
            value       TEXT NOT NULL,
            owner       TEXT NOT NULL,
            type        TEXT NOT NULL DEFAULT 'fact',
            confidence  REAL NOT NULL DEFAULT 1.0,
            refs        TEXT NOT NULL DEFAULT '',
            version     INTEGER NOT NULL DEFAULT 1,
            ttl         TEXT NOT NULL DEFAULT 'session',
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL,
            metadata    TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_knowledge_path ON knowledge(path);
        CREATE INDEX IF NOT EXISTS idx_knowledge_owner ON knowledge(owner);
        CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge(type);
        CREATE INDEX IF NOT EXISTS idx_knowledge_updated ON knowledge(updated_at);
    """)


MIGRATIONS: Dict[int, Callable[[sqlite3.Connection], None]] = {
    1: _migration_001_initial_schema,
    2: lambda conn: conn.executescript("""
        CREATE TABLE IF NOT EXISTS embeddings (
            path          TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            dimension     INTEGER NOT NULL,
            vector        BLOB NOT NULL,
            updated_at    REAL NOT NULL,
            FOREIGN KEY(path) REFERENCES knowledge(path) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_embeddings_provider
            ON embeddings(provider_name, dimension);
    """),
    3: lambda conn: conn.executescript("""
        CREATE TABLE IF NOT EXISTS system_state (
            state_key   TEXT PRIMARY KEY,
            value       TEXT NOT NULL,
            updated_at  REAL NOT NULL
        );
    """),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_schema_current(conn: sqlite3.Connection) -> int:
    """Check the schema version and apply any pending migrations.

    Returns the final schema version.
    """
    current = _get_schema_version(conn)
    for version in sorted(MIGRATIONS.keys()):
        if version > current:
            logger.info("Applying migration %d ...", version)
            MIGRATIONS[version](conn)
            _set_schema_version(conn, version)
            logger.info("Migration %d applied.", version)
    return _get_schema_version(conn)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_schema_version(conn: sqlite3.Connection) -> int:
    """Read the current schema version.  Creates the tracking table if needed."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version "
        "(version INTEGER NOT NULL, applied_at REAL NOT NULL)"
    )
    conn.commit()
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return row[0] or 0


def _set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
        (version, time.time()),
    )
    conn.commit()
