"""Tests for embedding integration using MockEmbeddingProvider."""

import pytest
import numpy as np

from dimensionalbase import DimensionalBase


class TestEmbeddingIntegration:
    """Test DimensionalBase features that require embeddings."""

    def test_default_constructor_stays_text_only_without_explicit_provider(self, monkeypatch):
        """Environment keys should not silently enable network-backed embeddings."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        db = DimensionalBase()
        assert db.has_embeddings is False
        assert db.channel.name == "TEXT"
        db.close()

    def test_encode_returns_vector(self, db_with_embeddings):
        """encode() should return a numpy array."""
        vec = db_with_embeddings.encode("hello world")
        assert vec is not None
        assert isinstance(vec, np.ndarray)
        assert len(vec) == 64  # MockEmbeddingProvider dimension

    def test_entries_get_embeddings(self, db_with_embeddings):
        """Entries should have embeddings when provider is available."""
        db_with_embeddings.put("x", "hello world", owner="a")
        entry = db_with_embeddings.retrieve("x")
        assert entry is not None
        assert entry.has_embedding

    def test_relate_returns_metrics(self, db_with_embeddings):
        """relate() should return relationship metrics with embeddings."""
        db_with_embeddings.put("a", "machine learning is great", owner="x")
        db_with_embeddings.put("b", "deep learning is powerful", owner="x")
        rel = db_with_embeddings.relate("a", "b")
        assert rel is not None
        assert "cosine" in rel
        assert "angular_dist" in rel

    def test_compose_returns_vector(self, db_with_embeddings):
        """compose() should return a combined vector."""
        db_with_embeddings.put("a", "hello", owner="x")
        db_with_embeddings.put("b", "world", owner="x")
        vec = db_with_embeddings.compose(["a", "b"])
        assert vec is not None
        assert isinstance(vec, np.ndarray)

    def test_materialize_returns_paths(self, db_with_embeddings):
        """materialize() should return nearest paths."""
        db_with_embeddings.put("a", "hello", owner="x")
        db_with_embeddings.put("b", "world", owner="x")
        vec = db_with_embeddings.encode("hello")
        results = db_with_embeddings.materialize(vec, k=5)
        assert len(results) > 0
        assert results[0][0] in ["a", "b"]

    def test_semantic_query_boosts_relevant(self, db_with_embeddings):
        """A semantic query should influence which entries are returned."""
        db_with_embeddings.put("auth/jwt", "JWT token expired", owner="a")
        db_with_embeddings.put("deploy/status", "Deployment successful", owner="a")
        db_with_embeddings.put("auth/oauth", "OAuth token refresh needed", owner="a")

        result = db_with_embeddings.get("**", budget=5000, query="authentication tokens")
        assert len(result.entries) >= 2

    def test_knowledge_topology(self, db_with_embeddings):
        """knowledge_topology() should work with embeddings."""
        for i in range(10):
            db_with_embeddings.put(f"topic/{i}", f"Entry about topic {i}", owner="a")

        topo = db_with_embeddings.knowledge_topology()
        assert topo["available"] is True
        assert topo["total_points"] == 10

    def test_encode_without_embeddings(self):
        """encode() should return None without embeddings."""
        from dimensionalbase.embeddings.provider import NullEmbeddingProvider
        db = DimensionalBase(embedding_provider=NullEmbeddingProvider())
        assert db.encode("hello") is None
        db.close()

    def test_status_with_embeddings(self, db_with_embeddings):
        """Status should include space metrics with embeddings."""
        db_with_embeddings.put("x", "hello", owner="a")
        status = db_with_embeddings.status()
        assert "space" in status
        assert status["embeddings"] is True
