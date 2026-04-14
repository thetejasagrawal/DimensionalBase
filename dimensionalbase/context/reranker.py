"""
Re-ranking pass for retrieved knowledge entries.

After the fast embedding retrieval (top ~50 candidates), a cross-encoder
scores each (query, chunk) pair jointly.  This distinguishes "related to
the question" from "actually answers the question" — a distinction that
bi-encoder embeddings fundamentally cannot make.

Two implementations:
  - CrossEncoderReranker  — local, free, ~50-100 ms for 50 candidates
  - NullReranker          — passthrough (when reranking is disabled)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

logger = logging.getLogger("dimensionalbase.context.reranker")


class Reranker(ABC):
    """Abstract re-ranking interface."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        entries: List[Tuple[Any, float]],
        top_k: int = 10,
    ) -> List[Tuple[Any, float]]:
        """Re-score entries and return the top_k by the new scores.

        Args:
            query:   The natural-language query.
            entries: List of (KnowledgeEntry, embedding_score) tuples.
            top_k:   How many to keep after re-ranking.

        Returns:
            Re-ranked list of (entry, new_score) tuples, length ≤ top_k.
        """


class NullReranker(Reranker):
    """Passthrough — returns entries unchanged."""

    def rerank(self, query, entries, top_k=10):
        return entries[:top_k]


class CrossEncoderReranker(Reranker):
    """Re-rank using a local cross-encoder model (sentence-transformers).

    The model is lazy-loaded on first call so construction is free.
    Default model: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (~22 MB).
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None  # lazy

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading cross-encoder model: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
            logger.info("Cross-encoder ready")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Install with: pip install dimensionalbase[embeddings-local]"
            )

    def rerank(
        self,
        query: str,
        entries: List[Tuple[Any, float]],
        top_k: int = 10,
    ) -> List[Tuple[Any, float]]:
        if not entries:
            return []

        self._ensure_model()

        # Build (query, chunk_text) pairs for the cross-encoder
        pairs = [(query, entry.value[:1500]) for entry, _ in entries]

        # Batch score — one forward pass through the cross-encoder
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Pair entries with new scores
        reranked = [
            (entry, float(score))
            for (entry, _old_score), score in zip(entries, scores)
        ]

        # Sort by cross-encoder score descending, take top_k
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
