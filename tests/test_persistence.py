"""Tests for SQLite persistence and schema migration."""

import numpy as np
import pytest

from dimensionalbase import DimensionalBase
from dimensionalbase.embeddings.provider import EmbeddingProvider

from dimensionalbase.security.encryption import FernetEncryptionProvider


class _PersistentEmbeddingProvider(EmbeddingProvider):
    def __init__(self, name: str = "persistent-mock", dim: int = 16):
        self._name = name
        self._dim = dim

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.RandomState(hash((self._name, text)) % (2**31))
        vec = rng.randn(self._dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-12 else vec

    def embed_batch(self, texts):
        return [self.embed(text) for text in texts]

    def dimension(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name


class TestPersistence:
    """Test that data survives close/reopen cycles."""

    def test_write_close_reopen_read(self, tmp_db_path):
        """Entries should survive close and reopen."""
        db = DimensionalBase(db_path=tmp_db_path)
        db.put("task/auth", "JWT expired", owner="agent-a", confidence=0.9)
        db.put("task/deploy", "Deploying v2.1", owner="agent-b")
        assert db.entry_count == 2
        db.close()

        db2 = DimensionalBase(db_path=tmp_db_path)
        assert db2.entry_count == 2
        entry = db2.retrieve("task/auth")
        assert entry is not None
        assert entry.value == "JWT expired"
        assert entry.owner == "agent-a"
        assert entry.confidence == 0.9
        db2.close()

    def test_version_survives(self, tmp_db_path):
        """Version increments should persist."""
        db = DimensionalBase(db_path=tmp_db_path)
        db.put("x", "v1", owner="a")
        db.put("x", "v2", owner="a")
        db.close()

        db2 = DimensionalBase(db_path=tmp_db_path)
        entry = db2.retrieve("x")
        assert entry is not None
        assert entry.version >= 2
        db2.close()

    def test_metadata_survives(self, tmp_db_path):
        """Metadata should persist."""
        db = DimensionalBase(db_path=tmp_db_path)
        db.put("x", "value", owner="a", metadata={"key1": "val1", "key2": "val2"})
        db.close()

        db2 = DimensionalBase(db_path=tmp_db_path)
        entry = db2.retrieve("x")
        assert entry is not None
        assert entry.metadata.get("key1") == "val1"
        assert entry.metadata.get("key2") == "val2"
        db2.close()

    def test_ttl_data_survives(self, tmp_db_path):
        """All TTL types should persist (TTL clearing is explicit, not automatic)."""
        db = DimensionalBase(db_path=tmp_db_path)
        db.put("a", "turn-data", owner="x", ttl="turn")
        db.put("b", "session-data", owner="x", ttl="session")
        db.put("c", "persistent-data", owner="x", ttl="persistent")
        db.close()

        db2 = DimensionalBase(db_path=tmp_db_path)
        assert db2.entry_count == 3
        db2.close()

    def test_schema_migration_creates_version_table(self, tmp_db_path):
        """Opening a new DB should create the schema_version table."""
        import sqlite3
        db = DimensionalBase(db_path=tmp_db_path)
        db.close()

        conn = sqlite3.connect(tmp_db_path)
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        assert row[0] >= 3
        conn.close()

    def test_reopen_does_not_duplicate_migration(self, tmp_db_path):
        """Reopening should not re-run migrations."""
        import sqlite3
        db = DimensionalBase(db_path=tmp_db_path)
        db.close()

        db2 = DimensionalBase(db_path=tmp_db_path)
        db2.close()

        conn = sqlite3.connect(tmp_db_path)
        rows = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()
        # Should have exactly the registered migrations, not duplicated
        assert rows[0] == 3
        conn.close()

    def test_embeddings_survive_reopen(self, tmp_db_path):
        provider = _PersistentEmbeddingProvider()
        db = DimensionalBase(db_path=tmp_db_path, embedding_provider=provider)
        db.put("task/auth", "JWT expired", owner="agent-a", confidence=0.9)
        expected = db.materialize(db.encode("JWT expired"))
        db.close()

        db2 = DimensionalBase(db_path=tmp_db_path, embedding_provider=provider)
        entry = db2.retrieve("task/auth")
        assert entry is not None
        assert entry.embedding is not None
        assert db2.status()["semantic_index_ready"] is True
        assert db2.status()["vector_entries"] == 1
        assert db2.materialize(db2.encode("JWT expired")) == expected
        db2.close()

    def test_provider_mismatch_reindexes_semantic_state(self, tmp_db_path):
        provider_a = _PersistentEmbeddingProvider(name="provider-a")
        provider_b = _PersistentEmbeddingProvider(name="provider-b")

        db = DimensionalBase(db_path=tmp_db_path, embedding_provider=provider_a)
        db.put("task/auth", "JWT expired", owner="agent-a")
        db.close()

        db2 = DimensionalBase(db_path=tmp_db_path, embedding_provider=provider_b)
        status = db2.status()
        assert status["reindexed_on_startup"] is True
        assert status["semantic_index_ready"] is True
        assert db2.retrieve("task/auth").embedding is not None
        assert db2.materialize(db2.encode("JWT expired"))[0][0] == "task/auth"
        db2.close()

    def test_clear_session_removes_semantic_state(self, tmp_db_path):
        provider = _PersistentEmbeddingProvider()
        db = DimensionalBase(db_path=tmp_db_path, embedding_provider=provider)
        db.put("turn/item", "turn state", owner="agent-a", ttl="turn")
        db.put("session/item", "session state", owner="agent-a", ttl="session")
        db.put("persistent/item", "persistent state", owner="agent-a", ttl="persistent")

        removed = db.clear_session()
        assert removed == 2
        assert db.entry_count == 1
        assert db.status()["vector_entries"] == 1
        assert [path for path, _ in db.materialize(db.encode("turn state"))] == ["persistent/item"]
        assert not db.exists("turn/item")
        assert not db.exists("session/item")
        assert db.exists("persistent/item")
        db.close()

    def test_encrypted_storage_roundtrip(self, tmp_db_path):
        import sqlite3

        try:
            provider = FernetEncryptionProvider(passphrase="hardening-release")
        except ImportError:
            pytest.skip("cryptography not installed")

        db = DimensionalBase(db_path=tmp_db_path, encryption_provider=provider)
        db.put("secret/value", "super sensitive text", owner="agent-a")
        db.close()

        conn = sqlite3.connect(tmp_db_path)
        raw_value = conn.execute(
            "SELECT value FROM knowledge WHERE path = ?",
            ("secret/value",),
        ).fetchone()[0]
        conn.close()

        assert raw_value != "super sensitive text"
        assert "sensitive" not in raw_value

        db2 = DimensionalBase(db_path=tmp_db_path, encryption_provider=provider)
        entry = db2.retrieve("secret/value")
        assert entry is not None
        assert entry.value == "super sensitive text"
        assert db2.status()["encryption_enabled"] is True
        db2.close()

    def test_reasoning_state_survives_reopen(self, tmp_db_path):
        db = DimensionalBase(db_path=tmp_db_path)
        db.put("task/auth/status", "JWT expired", owner="agent-a", confidence=0.9)
        db.put("task/auth/status", "JWT rotated", owner="agent-b", confidence=0.8)

        before_conf = db.confidence.get_confidence("task/auth/status")
        before_trust = db.agent_trust_report()
        before_lineage = db.lineage("task/auth/status")
        db.close()

        db2 = DimensionalBase(db_path=tmp_db_path)
        after_conf = db2.confidence.get_confidence("task/auth/status")
        after_trust = db2.agent_trust_report()
        after_lineage = db2.lineage("task/auth/status")

        assert after_conf == pytest.approx(before_conf, abs=1e-3)
        assert set(after_trust) == set(before_trust)
        assert len(after_lineage) == len(before_lineage)
        db2.close()
