"""
Naive RAG baseline for benchmark comparison.

Chunks documents, computes embeddings, returns top-k by cosine similarity.
No confidence, no trust, no budget packing — just raw vector retrieval.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import numpy as np


class NaiveRAGBaseline:
    """Simple chunk-and-retrieve RAG baseline.

    Uses random embeddings for reproducibility (no external model needed).
    The point is to compare retrieval STRATEGY, not embedding quality.
    """

    def __init__(self, chunk_size: int = 500, top_k: int = 5) -> None:
        self.chunk_size = chunk_size
        self.top_k = top_k
        self._chunks: List[str] = []
        self._embeddings: List[np.ndarray] = []
        self._dim = 64

    def _embed(self, text: str) -> np.ndarray:
        """Deterministic pseudo-embedding from text hash."""
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-12)

    def ingest(self, text: str) -> int:
        """Split text into chunks and embed."""
        words = text.split()
        total_tokens = 0
        for i in range(0, len(words), self.chunk_size):
            chunk = " ".join(words[i:i + self.chunk_size])
            self._chunks.append(chunk)
            self._embeddings.append(self._embed(chunk))
            total_tokens += len(chunk) // 4
        return total_tokens

    def query(self, question: str, budget: int = 2000) -> str:
        """Retrieve top-k chunks by cosine similarity."""
        if not self._chunks:
            return ""

        q_emb = self._embed(question)
        emb_matrix = np.stack(self._embeddings)
        sims = emb_matrix @ q_emb

        top_indices = np.argsort(sims)[::-1][:self.top_k]

        # Pack within budget (simple: concatenate until budget)
        result_parts = []
        tokens_used = 0
        for idx in top_indices:
            chunk = self._chunks[idx]
            chunk_tokens = len(chunk) // 4
            if tokens_used + chunk_tokens > budget:
                break
            result_parts.append(chunk)
            tokens_used += chunk_tokens

        return "\n\n".join(result_parts)

    def reset(self) -> None:
        self._chunks = []
        self._embeddings = []
