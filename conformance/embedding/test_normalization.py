"""
DBPS v1.0 Section 9 — Embedding Normalization Conformance Tests.

All embeddings in a DBPS-Embedding implementation MUST be L2-normalized
to unit length. This ensures dot(a, b) = cosine_similarity(a, b).
"""

from __future__ import annotations

import numpy as np
import pytest

from dimensionalbase.storage.vectors import VectorStore


class TestEmbeddingNormalization:
    """Verify L2 normalization invariant."""

    def test_added_vectors_are_normalized(self):
        """Every vector stored MUST have L2 norm = 1.0 (within floating point)."""
        store = VectorStore(dimension=64)
        rng = np.random.RandomState(42)

        for i in range(100):
            # Add unnormalized vectors — the store should normalize them
            vec = rng.randn(64).astype(np.float32) * (rng.rand() * 10 + 0.1)
            store.add(f"entry/{i}", vec)

        for i in range(100):
            stored = store.get(f"entry/{i}")
            assert stored is not None
            norm = np.linalg.norm(stored)
            assert abs(norm - 1.0) < 1e-5, \
                f"Vector entry/{i} has norm {norm}, expected 1.0"

    def test_dot_product_equals_cosine(self):
        """For normalized vectors, dot(a,b) MUST equal cosine_similarity(a,b)."""
        store = VectorStore(dimension=64)
        rng = np.random.RandomState(42)

        v1 = rng.randn(64).astype(np.float32)
        v2 = rng.randn(64).astype(np.float32)
        store.add("a", v1)
        store.add("b", v2)

        a = store.get("a")
        b = store.get("b")
        dot = float(np.dot(a, b))

        # Manually compute cosine
        cosine = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        assert abs(dot - cosine) < 1e-5, \
            f"dot product ({dot}) != cosine similarity ({cosine})"

    def test_zero_vector_handled(self):
        """A near-zero vector MUST NOT cause NaN after normalization."""
        store = VectorStore(dimension=64)
        tiny = np.zeros(64, dtype=np.float32)
        tiny[0] = 1e-15
        store.add("tiny", tiny)
        stored = store.get("tiny")
        assert stored is not None
        assert not np.any(np.isnan(stored))

    def test_search_returns_valid_similarities(self):
        """Search similarities MUST be in [-1.0, 1.0] for normalized vectors."""
        store = VectorStore(dimension=64)
        rng = np.random.RandomState(42)

        for i in range(50):
            store.add(f"e/{i}", rng.randn(64).astype(np.float32))

        query = rng.randn(64).astype(np.float32)
        query = query / np.linalg.norm(query)
        results = store.search(query, k=10)

        for path, sim in results:
            assert -1.0 - 1e-5 <= sim <= 1.0 + 1e-5, \
                f"Similarity {sim} out of range for {path}"
