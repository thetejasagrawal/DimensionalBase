"""
EmbeddingChannel — Channel 2. Near-lossless semantic communication.

v0.3: Uses the unified VectorStore instead of a separate EmbeddingIndex.
All vectors pre-normalized on write. dot(a,b) = cosine similarity everywhere.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from dimensionalbase.channels.base import Channel
from dimensionalbase.core.entry import KnowledgeEntry
from dimensionalbase.core.types import ChannelCapability, ChannelLevel
from dimensionalbase.embeddings.circuit_breaker import CircuitBreaker
from dimensionalbase.embeddings.provider import EmbeddingProvider
from dimensionalbase.storage.vectors import VectorStore

logger = logging.getLogger("dimensionalbase.channels.embedding")


class EmbeddingChannel(Channel):
    """Channel 2: Embedding storage with semantic similarity search.

    Wraps a TextChannel (for persistence) and a shared VectorStore
    (for vector search). The embedding IS the primary data.
    """

    def __init__(
        self,
        text_channel,
        provider: EmbeddingProvider,
        vector_store: VectorStore,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        self._text = text_channel
        self._provider = provider
        self._store = vector_store
        self._provider_name = provider.name
        self._dimension = provider.dimension()
        self._semantic_index_ready = False
        self._reindexed_on_startup = False
        self._breaker = circuit_breaker or CircuitBreaker()
        logger.info(
            f"EmbeddingChannel initialized: provider={provider.name}, "
            f"dim={provider.dimension()}"
        )
        self._initialize_index()

    def store(self, entry: KnowledgeEntry) -> None:
        """Store entry. Generate + pre-normalize embedding if missing."""
        if not entry.has_embedding:
            if self._breaker.is_open:
                logger.debug("Circuit breaker OPEN — skipping embedding for %s", entry.path)
            else:
                try:
                    entry.embedding = self._normalize(self._provider.embed(entry.value))
                    self._breaker.record_success()
                except Exception as e:
                    self._breaker.record_failure()
                    logger.warning(f"Failed to embed {entry.path}: {e}")

        self._text.store(entry)

        if entry.has_embedding:
            self._store.add(entry.path, entry.embedding)
            self._text.store_embedding(
                path=entry.path,
                vector=entry.embedding,
                provider_name=self._provider_name,
                dimension=self._dimension,
                updated_at=entry.updated_at,
            )
            self._semantic_index_ready = True

    def retrieve(self, path: str) -> Optional[KnowledgeEntry]:
        entry = self._text.retrieve(path)
        if entry is not None:
            emb = self._store.get(path)
            if emb is not None:
                entry.embedding = emb
        return entry

    def query_by_path(self, pattern: str) -> List[KnowledgeEntry]:
        entries = self._text.query_by_path(pattern)
        for entry in entries:
            emb = self._store.get(entry.path)
            if emb is not None:
                entry.embedding = emb
        return entries

    def query_by_similarity(
        self,
        query: str,
        k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[KnowledgeEntry, float]]:
        """Semantic similarity search. Falls back to path search if provider is down."""
        if self._breaker.is_open:
            logger.debug("Circuit breaker OPEN — falling back to path search for query")
            return []

        try:
            raw = self._provider.embed(query)
            self._breaker.record_success()
        except Exception as e:
            self._breaker.record_failure()
            logger.warning("Embedding provider failed during query, falling back: %s", e)
            return []

        # VectorStore normalizes internally
        results = self._store.search(raw, k=k)

        entries_with_scores = []
        for path, score in results:
            if score < threshold:
                continue
            entry = self._text.retrieve(path)
            if entry:
                emb = self._store.get(path)
                if emb is not None:
                    entry.embedding = emb
                entry._score = score
                entries_with_scores.append((entry, score))
        return entries_with_scores

    def delete(self, path: str) -> bool:
        self._store.remove(path)
        return self._text.delete(path)

    def all_entries(self) -> List[KnowledgeEntry]:
        entries = self._text.all_entries()
        for entry in entries:
            emb = self._store.get(entry.path)
            if emb is not None:
                entry.embedding = emb
        return entries

    def count(self) -> int:
        return self._text.count()

    def capability(self) -> ChannelCapability:
        return ChannelCapability(
            level=ChannelLevel.EMBEDDING,
            available=True,
            description=f"Embedding search via {self._provider.name}. Near-lossless.",
            latency_estimate="milliseconds",
            information_loss="minimal (embedding compression)",
        )

    @property
    def provider(self) -> EmbeddingProvider:
        return self._provider

    @property
    def vector_store(self) -> VectorStore:
        return self._store

    def close(self) -> None:
        logger.info("EmbeddingChannel closed")

    @property
    def semantic_index_ready(self) -> bool:
        return self._semantic_index_ready

    @property
    def reindexed_on_startup(self) -> bool:
        return self._reindexed_on_startup

    def _initialize_index(self) -> None:
        """Populate the in-memory index from durable storage."""
        entries = self._text.all_entries()
        if not entries:
            self._text.clear_embeddings()
            self._semantic_index_ready = True
            return

        persisted = {
            path: vector
            for path, vector in self._text.load_all_embeddings(
                provider_name=self._provider_name,
                dimension=self._dimension,
            )
        }

        if len(persisted) == len(entries) and all(entry.path in persisted for entry in entries):
            for entry in entries:
                self._store.add(entry.path, persisted[entry.path])
            self._semantic_index_ready = self._store.count == len(entries)
            return

        self._reindex_entries(entries)

    def _reindex_entries(self, entries: List[KnowledgeEntry]) -> None:
        """Re-embed all durable entries for the current provider."""
        logger.info(
            "Rebuilding semantic index for %d entries using provider=%s",
            len(entries),
            self._provider_name,
        )
        try:
            raw_vectors = self._provider.embed_batch([entry.value for entry in entries])
        except Exception:
            raw_vectors = [self._provider.embed(entry.value) for entry in entries]

        self._text.clear_embeddings()
        for entry, raw in zip(entries, raw_vectors):
            embedding = self._normalize(raw)
            entry.embedding = embedding
            self._store.add(entry.path, embedding)
            self._text.store_embedding(
                path=entry.path,
                vector=embedding,
                provider_name=self._provider_name,
                dimension=self._dimension,
                updated_at=entry.updated_at,
            )

        self._reindexed_on_startup = True
        self._semantic_index_ready = self._store.count == len(entries)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        raw = np.asarray(vector, dtype=np.float32).ravel()
        norm = np.linalg.norm(raw)
        if norm > 1e-12:
            raw = raw / norm
        return raw
