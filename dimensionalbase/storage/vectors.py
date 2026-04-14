"""
VectorStore — the single source of truth for all embeddings.

Replaces both EmbeddingIndex (channels/embedding.py) and
DimensionalSpace._vectors (algebra/space.py).

Design:
  - Contiguous float32 numpy array (enables single BLAS call for batch ops)
  - All vectors L2-normalized on insert: dot(a, b) = cosine similarity
  - Swap-on-delete keeps array contiguous (no gaps)
  - Thread-safe with RLock (read-reentrant)

Performance characteristics:
  - add():                O(1) amortized
  - remove():             O(1) (swap with last)
  - get():                O(1) dict lookup
  - search(k):            O(n) BLAS + O(n log k) partial sort
  - all_similarities():   O(n) single BLAS call
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

import numpy as np


class VectorStore:
    """Unified, contiguous float32 vector storage with pre-normalization.

    All vectors are L2-normalized on insert. After that:
      np.dot(a, b) == cosine_similarity(a, b)
    No norm computation needed anywhere in the hot path.
    """

    __slots__ = (
        "_dim", "_capacity", "_matrix", "_paths", "_path_to_idx",
        "_count", "_lock",
    )

    def __init__(self, dimension: int, initial_capacity: int = 1024):
        self._dim = dimension
        self._capacity = initial_capacity
        self._matrix = np.zeros((initial_capacity, dimension), dtype=np.float32)
        self._paths: List[str] = []
        self._path_to_idx: Dict[str, int] = {}
        self._count = 0
        self._lock = threading.RLock()

    # ── Core CRUD ─────────────────────────────────────────────

    def add(self, path: str, vector: np.ndarray) -> int:
        """Add or update a normalized vector. Returns slot index."""
        vec = self._normalize(vector)

        with self._lock:
            if path in self._path_to_idx:
                idx = self._path_to_idx[path]
                self._matrix[idx] = vec
                return idx

            if self._count >= self._capacity:
                self._grow()

            idx = self._count
            self._matrix[idx] = vec
            self._paths.append(path)
            self._path_to_idx[path] = idx
            self._count += 1
            return idx

    def remove(self, path: str) -> bool:
        """Remove a vector. O(1) via swap-with-last."""
        with self._lock:
            if path not in self._path_to_idx:
                return False

            idx = self._path_to_idx.pop(path)
            last_idx = self._count - 1

            if idx != last_idx:
                last_path = self._paths[last_idx]
                self._matrix[idx] = self._matrix[last_idx]
                self._paths[idx] = last_path
                self._path_to_idx[last_path] = idx

            self._paths.pop()
            self._count -= 1
            return True

    def get(self, path: str) -> Optional[np.ndarray]:
        """Get a vector by path. Returns a copy (safe to mutate)."""
        with self._lock:
            if path not in self._path_to_idx:
                return None
            return self._matrix[self._path_to_idx[path]].copy()

    def contains(self, path: str) -> bool:
        with self._lock:
            return path in self._path_to_idx

    # ── Batch Operations (BLAS-accelerated) ───────────────────

    def search(self, query: np.ndarray, k: int = 10,
               exclude: Optional[set] = None) -> List[Tuple[str, float]]:
        """Top-k similarity search. Single BLAS matmul."""
        q = self._normalize(query)

        with self._lock:
            if self._count == 0:
                return []

            sims = self._matrix[:self._count] @ q  # ONE BLAS call

            if exclude:
                for path, idx in self._path_to_idx.items():
                    if path in exclude and idx < self._count:
                        sims[idx] = -2.0

            top_k = min(k, self._count)
            if top_k >= self._count:
                indices = np.argsort(sims)[::-1][:top_k]
            else:
                indices = np.argpartition(sims, -top_k)[-top_k:]
                indices = indices[np.argsort(sims[indices])[::-1]]

            return [(self._paths[i], float(sims[i])) for i in indices
                    if sims[i] > -1.5]

    def all_similarities(self, query: np.ndarray) -> np.ndarray:
        """Similarity of query against ALL stored vectors. Single BLAS call.

        Returns ndarray of shape (count,). Index i corresponds to self._paths[i].
        """
        q = self._normalize(query)
        with self._lock:
            if self._count == 0:
                return np.array([], dtype=np.float32)
            return (self._matrix[:self._count] @ q).copy()

    def pairwise_similarity(self, path_a: str, path_b: str) -> Optional[float]:
        """Cosine similarity between two stored vectors."""
        with self._lock:
            if path_a not in self._path_to_idx or path_b not in self._path_to_idx:
                return None
            a = self._matrix[self._path_to_idx[path_a]]
            b = self._matrix[self._path_to_idx[path_b]]
            return float(np.dot(a, b))

    def get_active_data(self) -> Tuple[np.ndarray, List[str]]:
        """Return (matrix_copy, paths_copy) for batch analytics.

        Returns copies — safe to use outside the lock.
        """
        with self._lock:
            return (
                self._matrix[:self._count].copy(),
                list(self._paths),
            )

    def get_vectors_for_paths(self, paths: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Get a stacked matrix for specific paths. Skips missing paths."""
        with self._lock:
            found_vecs = []
            found_paths = []
            for p in paths:
                if p in self._path_to_idx:
                    found_vecs.append(self._matrix[self._path_to_idx[p]])
                    found_paths.append(p)
            if not found_vecs:
                return np.empty((0, self._dim), dtype=np.float32), []
            return np.stack(found_vecs), found_paths

    # ── Properties ────────────────────────────────────────────

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def paths(self) -> List[str]:
        with self._lock:
            return list(self._paths)

    @property
    def memory_bytes(self) -> int:
        """Approximate memory usage in bytes."""
        return self._matrix.nbytes + self._count * 64  # matrix + path overhead

    # ── Internal ──────────────────────────────────────────────

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """L2-normalize to float32. After this, dot product = cosine similarity."""
        v = np.asarray(vector, dtype=np.float32).ravel()
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            v = v / norm
        return v

    def _grow(self):
        """Double capacity."""
        new_cap = self._capacity * 2
        new_matrix = np.zeros((new_cap, self._dim), dtype=np.float32)
        new_matrix[:self._count] = self._matrix[:self._count]
        self._matrix = new_matrix
        self._capacity = new_cap
