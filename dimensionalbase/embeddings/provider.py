"""
Embedding providers.

Default startup is intentionally conservative:
  1. Explicitly requested local model (`prefer="local"`)
  2. Explicitly requested OpenAI provider (`prefer="openai"` or `openai_api_key=...`)
  3. None (text-only mode, still works)

This avoids slow or network-adjacent implicit model initialization during
ordinary `DimensionalBase()` construction.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger("dimensionalbase.embeddings")


class EmbeddingProvider(ABC):
    """Abstract base for all embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string into a vector."""
        ...

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts. Override for providers that support batching."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimensionality."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
        ...

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class LocalEmbeddingProvider(EmbeddingProvider):
    """Uses sentence-transformers for local, free, no-network embeddings.

    Model: all-MiniLM-L6-v2 (~22MB, 384 dimensions)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install dimensionalbase[embeddings-local]"
            )
        logger.info(f"Loading local embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"Local embedding model loaded: dim={self._dim}")

    def embed(self, text: str) -> np.ndarray:
        return self._model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [embeddings[i] for i in range(len(texts))]

    def dimension(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"local:{self._model_name}"


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Uses OpenAI's text-embedding-3-small API.

    Cost: ~$0.02 per million tokens. 1536 dimensions.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
    ):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai not installed. "
                "Run: pip install dimensionalbase[embeddings-openai]"
            )
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self._client = openai.OpenAI(api_key=self._api_key)
        self._model = model
        self._dim = 1536  # text-embedding-3-small default
        logger.info(f"OpenAI embedding provider initialized: model={model}")

    def embed(self, text: str) -> np.ndarray:
        response = self._client.embeddings.create(
            input=text,
            model=self._model,
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        response = self._client.embeddings.create(
            input=texts,
            model=self._model,
        )
        return [
            np.array(item.embedding, dtype=np.float32)
            for item in sorted(response.data, key=lambda x: x.index)
        ]

    def dimension(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"openai:{self._model}"


class NullEmbeddingProvider(EmbeddingProvider):
    """Fallback when no embedding model is available.

    Text-only mode. DimensionalBase still works — you just lose
    semantic similarity search (Channel 2). Channel 1 (text) is
    always available.
    """

    def embed(self, text: str) -> np.ndarray:
        raise RuntimeError(
            "No embedding provider available. "
            "Install sentence-transformers or set OPENAI_API_KEY."
        )

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        raise RuntimeError(
            "No embedding provider available. "
            "Install sentence-transformers or set OPENAI_API_KEY."
        )

    def dimension(self) -> int:
        return 0

    @property
    def name(self) -> str:
        return "none"


def auto_detect_provider(
    prefer: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> EmbeddingProvider:
    """Auto-detect the best available embedding provider.

    Priority order (unless overridden by `prefer`):
      1. Explicit local sentence-transformers
      2. Explicit OpenAI API
      3. Null provider (text-only mode)

    Args:
        prefer: Force a specific provider ('local', 'openai', or None for auto).
        openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var).

    Returns:
        The best available EmbeddingProvider.
    """
    if prefer == "local":
        return LocalEmbeddingProvider()

    if prefer == "openai":
        return OpenAIEmbeddingProvider(api_key=openai_api_key)

    # Auto-detect: only initialize providers that are cheap and explicit.
    # Local model bootstrap can import large stacks and attempt model resolution,
    # and OpenAI embeddings introduce network-backed writes. Both are opt-in.
    # 1. Try OpenAI only when the caller explicitly passed a key.
    if openai_api_key:
        try:
            provider = OpenAIEmbeddingProvider(api_key=openai_api_key)
            logger.info(f"Auto-detected embedding provider: {provider.name}")
            return provider
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI embeddings: {e}")

    # 2. Fall back to null
    logger.info(
        "No explicit embedding provider configured. Running in text-only mode. "
        "Pass prefer='local', prefer='openai', openai_api_key, or a custom "
        "embedding provider to enable semantic search."
    )
    return NullEmbeddingProvider()
