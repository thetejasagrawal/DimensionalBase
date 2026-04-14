"""
Shared test fixtures for DimensionalBase test suite.
"""

import numpy as np
import pytest

from dimensionalbase import DimensionalBase
from dimensionalbase.embeddings.provider import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for tests.

    Generates reproducible 64-dim embeddings seeded from the input text hash.
    """

    def __init__(self, dimension: int = 64) -> None:
        self._dim = dimension
        self._cache: dict = {}

    def embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            rng = np.random.RandomState(hash(text) % (2**31))
            vec = rng.randn(self._dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                vec = vec / norm
            self._cache[text] = vec
        return self._cache[text].copy()

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    def dimension(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return "mock"


@pytest.fixture
def db():
    """In-memory DimensionalBase (text-only mode)."""
    d = DimensionalBase()
    yield d
    d.close()


@pytest.fixture
def db_with_embeddings():
    """In-memory DimensionalBase with mock embeddings."""
    d = DimensionalBase(embedding_provider=MockEmbeddingProvider())
    yield d
    d.close()


@pytest.fixture
def mock_provider():
    """A standalone MockEmbeddingProvider."""
    return MockEmbeddingProvider()


@pytest.fixture
def tmp_db_path(tmp_path):
    """Temporary file path for file-based SQLite tests."""
    return str(tmp_path / "test.db")
