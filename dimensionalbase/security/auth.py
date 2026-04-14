"""
API key management for DimensionalBase.

Keys are stored SHA-256 hashed. Callers provide plaintext; the system
compares hashes.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._cache: Dict[str, APIKey] = {}

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

    def ensure_key(self, plaintext_key: str, agent_id: str, is_admin: bool = False) -> APIKey:
        """Create or restore a known plaintext key for an agent."""
        key_hash = hashlib.sha256(plaintext_key.encode()).hexdigest()
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
        self._cache[key_hash] = key
        return key

    def validate(self, plaintext_key: str) -> APIKey:
        """Validate a key and return the associated APIKey. Raises AuthError if invalid."""
        key_hash = hashlib.sha256(plaintext_key.encode()).hexdigest()

        # Check cache first
        if key_hash in self._cache:
            key = self._cache[key_hash]
            if key.revoked:
                raise AuthError("API key has been revoked")
            return key

        row = self._conn.execute(
            "SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,)
        ).fetchone()
        if row is None:
            raise AuthError("Invalid API key")
        if row["revoked"]:
            raise AuthError("API key has been revoked")

        key = APIKey(
            key_hash=row["key_hash"], agent_id=row["agent_id"],
            created_at=row["created_at"], is_admin=bool(row["is_admin"]),
        )
        self._cache[key_hash] = key
        return key

    def revoke(self, plaintext_key: str) -> bool:
        """Revoke an API key."""
        key_hash = hashlib.sha256(plaintext_key.encode()).hexdigest()
        cursor = self._conn.execute(
            "UPDATE api_keys SET revoked = 1 WHERE key_hash = ?", (key_hash,)
        )
        self._conn.commit()
        if key_hash in self._cache:
            self._cache[key_hash].revoked = True
        return cursor.rowcount > 0

    def list_keys(self) -> List[APIKey]:
        """List all registered keys (hashed, not plaintext)."""
        rows = self._conn.execute("SELECT * FROM api_keys").fetchall()
        return [
            APIKey(
                key_hash=r["key_hash"], agent_id=r["agent_id"],
                created_at=r["created_at"], is_admin=bool(r["is_admin"]),
                revoked=bool(r["revoked"]),
            )
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
