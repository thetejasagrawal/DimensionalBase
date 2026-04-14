"""
TextChannel — Channel 1. Always on. Universal fallback.

Uses SQLite for structured storage. Every entry is stored as text + metadata.
This channel also owns the durable embedding table so the semantic layer can
rebuild itself on startup without re-putting entries.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dimensionalbase.channels.base import Channel
from dimensionalbase.core.entry import KnowledgeEntry
from dimensionalbase.core.types import ChannelCapability, ChannelLevel, EntryType, TTL
from dimensionalbase.exceptions import StorageError
from dimensionalbase.security.encryption import EncryptionProvider, NullEncryptionProvider
from dimensionalbase.storage.migrations import ensure_schema_current

logger = logging.getLogger("dimensionalbase.channels.text")


class TextChannel(Channel):
    """Channel 1: Text storage via SQLite.

    Information loss: significant (dimensionality collapse).
    Speed: milliseconds.
    Available: always.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        encryption_provider: Optional[EncryptionProvider] = None,
    ) -> None:
        """Initialize the text channel.

        Args:
            db_path: Path to SQLite database file, or ':memory:' for in-memory.
        """
        self._db_path = db_path
        self._lock = threading.Lock()
        self._encryption = encryption_provider or NullEncryptionProvider()
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to open database at {db_path}: {exc}") from exc
        self._init_schema()
        logger.info(f"TextChannel initialized: {db_path}")

    def _init_schema(self) -> None:
        with self._lock:
            try:
                ensure_schema_current(self._conn)
            except sqlite3.Error as exc:
                raise StorageError(f"Schema initialization failed: {exc}") from exc

    def store(self, entry: KnowledgeEntry) -> None:
        """Store or update a knowledge entry."""
        metadata_json = json.dumps(entry.metadata) if entry.metadata else "{}"
        refs_str = ",".join(entry.refs) if entry.refs else ""
        encrypted_value = self._encryption.encrypt(entry.value)

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO knowledge (id, path, value, owner, type, confidence,
                                       refs, version, ttl, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    value = excluded.value,
                    owner = excluded.owner,
                    type = excluded.type,
                    confidence = excluded.confidence,
                    refs = excluded.refs,
                    version = knowledge.version + 1,
                    ttl = excluded.ttl,
                    updated_at = excluded.updated_at,
                    metadata = excluded.metadata
                """,
                (
                    entry.id, entry.path, encrypted_value, entry.owner,
                    entry.type.value, entry.confidence, refs_str,
                    entry.version, entry.ttl.value, entry.created_at,
                    entry.updated_at, metadata_json,
                ),
            )
            self._conn.commit()

    def store_embedding(
        self,
        path: str,
        vector: np.ndarray,
        provider_name: str,
        dimension: int,
        updated_at: float,
    ) -> None:
        """Persist a normalized embedding for a path."""
        vec = np.asarray(vector, dtype=np.float32).ravel()
        if len(vec) != dimension:
            raise StorageError(
                f"Embedding dimension mismatch for {path}: {len(vec)} != {dimension}"
            )

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO embeddings (path, provider_name, dimension, vector, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    provider_name = excluded.provider_name,
                    dimension = excluded.dimension,
                    vector = excluded.vector,
                    updated_at = excluded.updated_at
                """,
                (
                    path,
                    provider_name,
                    dimension,
                    sqlite3.Binary(vec.tobytes()),
                    updated_at,
                ),
            )
            self._conn.commit()

    def retrieve(self, path: str) -> Optional[KnowledgeEntry]:
        """Retrieve an entry by exact path."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM knowledge WHERE path = ?", (path,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def retrieve_embedding(
        self,
        path: str,
        provider_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Retrieve a persisted embedding by path."""
        query = "SELECT provider_name, dimension, vector FROM embeddings WHERE path = ?"
        params: List[object] = [path]
        if provider_name is not None:
            query += " AND provider_name = ?"
            params.append(provider_name)
        if dimension is not None:
            query += " AND dimension = ?"
            params.append(dimension)

        with self._lock:
            row = self._conn.execute(query, tuple(params)).fetchone()

        if row is None:
            return None

        vector = np.frombuffer(row["vector"], dtype=np.float32).copy()
        if row["dimension"] != len(vector):
            return None
        return vector

    def load_all_embeddings(
        self,
        provider_name: str,
        dimension: int,
    ) -> List[Tuple[str, np.ndarray]]:
        """Load all persisted embeddings for a provider/dimension pair."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT path, vector, dimension
                FROM embeddings
                WHERE provider_name = ? AND dimension = ?
                ORDER BY path
                """,
                (provider_name, dimension),
            ).fetchall()

        loaded: List[Tuple[str, np.ndarray]] = []
        for row in rows:
            vector = np.frombuffer(row["vector"], dtype=np.float32).copy()
            if len(vector) != row["dimension"]:
                continue
            loaded.append((row["path"], vector))
        return loaded

    def clear_embeddings(self) -> int:
        """Remove all persisted embeddings."""
        with self._lock:
            cursor = self._conn.execute("DELETE FROM embeddings")
            self._conn.commit()
            return cursor.rowcount

    def delete_embedding(self, path: str) -> bool:
        """Delete a persisted embedding."""
        with self._lock:
            cursor = self._conn.execute("DELETE FROM embeddings WHERE path = ?", (path,))
            self._conn.commit()
            return cursor.rowcount > 0

    def delete_embeddings_for_paths(self, paths: Sequence[str]) -> int:
        """Delete persisted embeddings for a list of paths."""
        if not paths:
            return 0

        with self._lock:
            cursor = self._conn.executemany(
                "DELETE FROM embeddings WHERE path = ?",
                [(path,) for path in paths],
            )
            self._conn.commit()
            return cursor.rowcount

    def embedding_count(
        self,
        provider_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> int:
        """Count persisted embeddings, optionally filtered by provider."""
        query = "SELECT COUNT(*) FROM embeddings"
        params: List[object] = []
        clauses: List[str] = []
        if provider_name is not None:
            clauses.append("provider_name = ?")
            params.append(provider_name)
        if dimension is not None:
            clauses.append("dimension = ?")
            params.append(dimension)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        with self._lock:
            row = self._conn.execute(query, tuple(params)).fetchone()
            return int(row[0])

    def save_system_state(self, state_key: str, payload: Dict[str, Any]) -> None:
        """Persist JSON-serializable subsystem state."""
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        stored = self._encryption.encrypt(raw)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO system_state (state_key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(state_key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (state_key, stored, time.time()),
            )
            self._conn.commit()

    def load_system_state(self, state_key: str) -> Optional[Dict[str, Any]]:
        """Load persisted subsystem state."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM system_state WHERE state_key = ?",
                (state_key,),
            ).fetchone()

        if row is None:
            return None

        try:
            raw = self._encryption.decrypt(row["value"])
            data = json.loads(raw)
        except Exception:
            logger.warning("Failed to decode persisted system state for key=%s", state_key)
            return None

        if not isinstance(data, dict):
            return None
        return data

    def query_by_path(self, pattern: str, limit: int = 0) -> List[KnowledgeEntry]:
        """Retrieve entries matching a glob pattern.

        Supports * (single level) and ** (multi-level) patterns.
        An optional *limit* caps the number of rows returned (0 = no limit).
        Results are always ordered by ``updated_at DESC`` (most recent first).
        """
        suffix = f" LIMIT {limit}" if limit > 0 else ""
        with self._lock:
            # For simple prefix patterns, use SQL LIKE for speed
            if pattern.endswith("/**"):
                prefix = pattern[:-3]
                rows = self._conn.execute(
                    "SELECT * FROM knowledge WHERE path = ? OR path LIKE ?"
                    f" ORDER BY updated_at DESC{suffix}",
                    (prefix, prefix + "/%"),
                ).fetchall()
            elif "*" not in pattern and "?" not in pattern:
                # Exact match
                rows = self._conn.execute(
                    "SELECT * FROM knowledge WHERE path = ?", (pattern,)
                ).fetchall()
            else:
                # Fall back to loading all and filtering with fnmatch
                rows = self._conn.execute(
                    f"SELECT * FROM knowledge ORDER BY updated_at DESC{suffix}"
                ).fetchall()
                rows = [r for r in rows if self._glob_match(pattern, r["path"])]

        return [self._row_to_entry(r) for r in rows]

    def delete(self, path: str) -> bool:
        """Delete an entry by path."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM knowledge WHERE path = ?", (path,)
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def all_entries(self) -> List[KnowledgeEntry]:
        """Return all stored entries."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM knowledge ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def count(self) -> int:
        """Number of entries stored."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()
            return row[0]

    def clear_by_ttl(self, ttl: TTL) -> int:
        """Delete all entries with a specific TTL. Returns count deleted."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM knowledge WHERE ttl = ?", (ttl.value,)
            )
            self._conn.commit()
            return cursor.rowcount

    def delete_paths_by_ttl(self, ttl: TTL) -> List[str]:
        """Delete all entries with a TTL and return the removed paths."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT path FROM knowledge WHERE ttl = ? ORDER BY path",
                (ttl.value,),
            ).fetchall()
            paths = [row["path"] for row in rows]
            if not paths:
                return []
            self._conn.executemany(
                "DELETE FROM knowledge WHERE path = ?",
                [(path,) for path in paths],
            )
            self._conn.commit()
            return paths

    def entries_by_owner(self, owner: str) -> List[KnowledgeEntry]:
        """Get all entries owned by a specific agent."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM knowledge WHERE owner = ? ORDER BY updated_at DESC",
                (owner,),
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def capability(self) -> ChannelCapability:
        return ChannelCapability(
            level=ChannelLevel.TEXT,
            available=True,
            description="Text storage via SQLite. Always available.",
            latency_estimate="milliseconds",
            information_loss="significant (dimensionality collapse)",
        )

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._lock:
            self._conn.close()
        logger.info("TextChannel closed")

    @property
    def encryption_enabled(self) -> bool:
        return not isinstance(self._encryption, NullEncryptionProvider)

    # --- Internal helpers ---

    def _row_to_entry(self, row: sqlite3.Row) -> KnowledgeEntry:
        """Convert a SQLite row to a KnowledgeEntry."""
        refs_str = row["refs"]
        refs = [r for r in refs_str.split(",") if r] if refs_str else []

        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass

        return KnowledgeEntry(
            id=row["id"],
            path=row["path"],
            value=self._encryption.decrypt(row["value"]),
            owner=row["owner"],
            type=EntryType(row["type"]),
            confidence=row["confidence"],
            refs=refs,
            version=row["version"],
            ttl=TTL(row["ttl"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=metadata,
        )

    @staticmethod
    def _glob_match(pattern: str, path: str) -> bool:
        """Match a path against a glob pattern.  Delegates to canonical DBPS matcher."""
        from dimensionalbase.core.matching import dbps_match
        return dbps_match(pattern, path)
