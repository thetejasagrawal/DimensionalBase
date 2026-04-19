"""
API key management for DimensionalBase.

Keys are stored SHA-256 hashed. Callers provide plaintext; the system
compares hashes.
"""

from __future__ import annotations

import hmac
import hashlib
import secrets
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from dimensionalbase.exceptions import DimensionalBaseError


class AuthError(DimensionalBaseError):
    """Invalid or missing API key."""


@dataclass
class APIKey:
    """Represents a registered API key."""
    key_hash: str
    agent_id: str
    created_at: float
    is_admin: bool = False
    revoked: bool = False


class APIKeyManager:
    """Manages API key lifecycle — generation, validation, revocation.

    Keys are stored in a separate SQLite table for isolation from the
    knowledge store.
    """

    def __init__(self, db_path: str = ":memory:", cache_ttl_seconds: float = 5.0) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._cache_ttl_seconds = max(0.0, cache_ttl_seconds)
        self._cache: Dict[str, Tuple[APIKey, float]] = {}

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key_hash   TEXT PRIMARY KEY,
                agent_id   TEXT NOT NULL,
                created_at REAL NOT NULL,
                is_admin   INTEGER NOT NULL DEFAULT 0,
                revoked    INTEGER NOT NULL DEFAULT 0
            )
        """)
        self._conn.commit()

    def generate_key(self, agent_id: str, is_admin: bool = False) -> str:
        """Generate a new API key for an agent. Returns the plaintext key."""
        plaintext = f"dmb_{secrets.token_urlsafe(32)}"
        self.ensure_key(plaintext, agent_id=agent_id, is_admin=is_admin)
        return plaintext

    @staticmethod
    def _hash_key(plaintext_key: str) -> str:
        return hashlib.sha256(plaintext_key.encode()).hexdigest()

    @staticmethod
    def _row_to_key(row: sqlite3.Row) -> APIKey:
        return APIKey(
            key_hash=row["key_hash"],
            agent_id=row["agent_id"],
            created_at=row["created_at"],
            is_admin=bool(row["is_admin"]),
            revoked=bool(row["revoked"]),
        )

    def _cache_set(self, key: APIKey) -> None:
        if self._cache_ttl_seconds <= 0:
            return
        self._cache[key.key_hash] = (key, time.monotonic() + self._cache_ttl_seconds)

    def _cache_get(self, key_hash: str) -> Optional[APIKey]:
        cached = self._cache.get(key_hash)
        if cached is None:
            return None
        key, expires_at = cached
        if expires_at < time.monotonic():
            self._cache.pop(key_hash, None)
            return None
        return key

    @staticmethod
    def _raise_invalid_key() -> None:
        raise AuthError("Invalid API key")

    def ensure_key(self, plaintext_key: str, agent_id: str, is_admin: bool = False) -> APIKey:
        """Create or restore a known plaintext key for an agent."""
        key_hash = self._hash_key(plaintext_key)
        created_at = time.time()
        row = self._conn.execute(
            "SELECT * FROM api_keys WHERE key_hash = ?",
            (key_hash,),
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO api_keys (key_hash, agent_id, created_at, is_admin, revoked) VALUES (?, ?, ?, ?, 0)",
                (key_hash, agent_id, created_at, int(is_admin)),
            )
        else:
            created_at = row["created_at"]
            self._conn.execute(
                """
                UPDATE api_keys
                SET agent_id = ?, is_admin = ?, revoked = 0
                WHERE key_hash = ?
                """,
                (agent_id, int(is_admin), key_hash),
            )
        self._conn.commit()
        key = APIKey(
            key_hash=key_hash, agent_id=agent_id,
            created_at=created_at, is_admin=is_admin,
        )
        self._cache_set(key)
        return key

    def validate(self, plaintext_key: str) -> APIKey:
        """Validate a key and return the associated APIKey. Raises AuthError if invalid."""
        key_hash = self._hash_key(plaintext_key)

        cached = self._cache_get(key_hash)
        if cached is not None:
            if cached.revoked or not hmac.compare_digest(cached.key_hash, key_hash):
                self._cache.pop(key_hash, None)
                self._raise_invalid_key()
            return cached

        row = self._conn.execute(
            "SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,)
        ).fetchone()
        if row is None:
            self._raise_invalid_key()

        key = self._row_to_key(row)
        if key.revoked or not hmac.compare_digest(key.key_hash, key_hash):
            self._cache.pop(key_hash, None)
            self._raise_invalid_key()

        self._cache_set(key)
        return key

    def revoke(self, plaintext_key: str) -> bool:
        """Revoke an API key."""
        key_hash = self._hash_key(plaintext_key)
        cursor = self._conn.execute(
            "UPDATE api_keys SET revoked = 1 WHERE key_hash = ?", (key_hash,)
        )
        self._conn.commit()
        self._cache.pop(key_hash, None)
        return cursor.rowcount > 0

    def list_keys(self) -> List[APIKey]:
        """List all registered keys (hashed, not plaintext)."""
        rows = self._conn.execute("SELECT * FROM api_keys").fetchall()
        return [self._row_to_key(r) for r in rows]

    def close(self) -> None:
        self._cache.clear()
        self._conn.close()
