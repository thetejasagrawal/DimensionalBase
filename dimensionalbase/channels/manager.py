"""
ChannelManager — auto-detects and orchestrates available channels.

v0.3: Creates a shared VectorStore that EmbeddingChannel and
DimensionalSpace both use. Single source of truth for vectors.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from dimensionalbase.channels.base import Channel
from dimensionalbase.channels.text import TextChannel
from dimensionalbase.channels.embedding import EmbeddingChannel
from dimensionalbase.channels.tensor import TensorChannel
from dimensionalbase.core.entry import KnowledgeEntry
from dimensionalbase.core.types import ChannelCapability, ChannelLevel
from dimensionalbase.embeddings.provider import (
    EmbeddingProvider,
    NullEmbeddingProvider,
    auto_detect_provider,
)
from dimensionalbase.security.encryption import (
    EncryptionProvider,
    FernetEncryptionProvider,
)
from dimensionalbase.storage.vectors import VectorStore

logger = logging.getLogger("dimensionalbase.channels")


class ChannelManager:
    """Manages and auto-detects available communication channels.

    Creates a shared VectorStore that the EmbeddingChannel and
    DimensionalSpace both reference — no double storage.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        embedding_provider: Optional[EmbeddingProvider] = None,
        prefer_embedding: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        encryption_provider: Optional[EncryptionProvider] = None,
        encryption_key: Optional[str] = None,
        encryption_passphrase: Optional[str] = None,
    ):
        if encryption_provider is not None:
            resolved_encryption = encryption_provider
        elif encryption_key or encryption_passphrase:
            resolved_encryption = FernetEncryptionProvider(
                key=encryption_key,
                passphrase=encryption_passphrase,
            )
        else:
            resolved_encryption = None

        # Channel 1: TEXT — always on
        self._text = TextChannel(
            db_path=db_path,
            encryption_provider=resolved_encryption,
        )

        # Detect embedding provider
        self._provider: Optional[EmbeddingProvider] = None
        if embedding_provider is not None:
            self._provider = embedding_provider
        else:
            self._provider = auto_detect_provider(
                prefer=prefer_embedding,
                openai_api_key=openai_api_key,
            )

        # Shared VectorStore + Channel 2
        self._vector_store: Optional[VectorStore] = None
        self._embedding: Optional[EmbeddingChannel] = None

        if not isinstance(self._provider, NullEmbeddingProvider):
            self._vector_store = VectorStore(
                dimension=self._provider.dimension(),
                initial_capacity=2048,
            )
            self._embedding = EmbeddingChannel(
                text_channel=self._text,
                provider=self._provider,
                vector_store=self._vector_store,
            )

        # Channel 3: TENSOR — detect only
        self._tensor = TensorChannel()
        self._reindexed_on_startup = (
            self._embedding.reindexed_on_startup if self._embedding is not None else False
        )

        caps = self.capabilities()
        active = [c for c in caps if c.available]
        logger.info(
            f"ChannelManager: {len(active)}/{len(caps)} channels "
            f"[{', '.join(c.level.name for c in active)}]"
        )

    @property
    def best_channel_level(self) -> ChannelLevel:
        if self._tensor.capability().available:
            return ChannelLevel.TENSOR
        if self._embedding is not None:
            return ChannelLevel.EMBEDDING
        return ChannelLevel.TEXT

    @property
    def has_embeddings(self) -> bool:
        return self._embedding is not None

    @property
    def text_channel(self) -> TextChannel:
        return self._text

    @property
    def embedding_channel(self) -> Optional[EmbeddingChannel]:
        return self._embedding

    @property
    def embedding_provider(self) -> Optional[EmbeddingProvider]:
        return self._provider

    @property
    def vector_store(self) -> Optional[VectorStore]:
        return self._vector_store

    @property
    def semantic_index_ready(self) -> bool:
        if self._embedding is None:
            return False
        return self._embedding.semantic_index_ready

    @property
    def reindexed_on_startup(self) -> bool:
        return self._reindexed_on_startup

    @property
    def encryption_enabled(self) -> bool:
        return self._text.encryption_enabled

    def store(self, entry: KnowledgeEntry) -> ChannelLevel:
        if self._embedding is not None:
            self._embedding.store(entry)
            return ChannelLevel.EMBEDDING
        else:
            self._text.store(entry)
            return ChannelLevel.TEXT

    def retrieve(self, path: str) -> Optional[KnowledgeEntry]:
        if self._embedding is not None:
            return self._embedding.retrieve(path)
        return self._text.retrieve(path)

    def query_by_path(self, pattern: str) -> List[KnowledgeEntry]:
        if self._embedding is not None:
            return self._embedding.query_by_path(pattern)
        return self._text.query_by_path(pattern)

    def query_by_similarity(
        self, query: str, k: int = 10, threshold: float = 0.0,
    ) -> List[Tuple[KnowledgeEntry, float]]:
        if self._embedding is None:
            return []
        return self._embedding.query_by_similarity(query, k=k, threshold=threshold)

    def delete(self, path: str) -> bool:
        if self._embedding is not None:
            return self._embedding.delete(path)
        return self._text.delete(path)

    def count(self) -> int:
        return self._text.count()

    def all_entries(self) -> List[KnowledgeEntry]:
        if self._embedding is not None:
            return self._embedding.all_entries()
        return self._text.all_entries()

    def capabilities(self) -> List[ChannelCapability]:
        return [
            self._text.capability(),
            self._embedding.capability() if self._embedding else ChannelCapability(
                level=ChannelLevel.EMBEDDING, available=False,
                description="No embedding provider available",
                latency_estimate="milliseconds", information_loss="minimal",
            ),
            self._tensor.capability(),
        ]

    def close(self) -> None:
        if self._embedding is not None:
            self._embedding.close()
        self._text.close()
        self._tensor.close()
        logger.info("ChannelManager closed")
