"""
SemanticFingerprint — locality-sensitive hashing for sub-linear similarity.

The problem: cosine similarity is O(d) per pair, O(n*d) per query.
When you have 100k entries and need to check for contradictions on
every write, you need something faster.

SemanticFingerprint uses multiple locality-sensitive hash functions
to create compact binary fingerprints. Two vectors with high cosine
similarity will have similar fingerprints with high probability.

This gives us:
  - O(1) approximate similarity checks (Hamming distance on fingerprints)
  - O(b/64) where b = number of hash bits (typically 256-1024)
  - Configurable precision/recall tradeoff via number of bits
  - Bloom filter integration for "have I seen something like this before?"

The math: each bit is a random hyperplane hash (SimHash).
Probability that two vectors share a bit = 1 - arccos(sim) / pi.
With 256 bits, we can distinguish similarities to ~0.05 resolution.
"""

from __future__ import annotations

import hashlib
import struct
import threading
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


class SemanticFingerprint:
    """Locality-sensitive hashing for fast approximate similarity.

    Usage:
        fp = SemanticFingerprint(dimension=384, n_bits=256)

        # Create fingerprints
        fp_a = fp.hash(vector_a)
        fp_b = fp.hash(vector_b)

        # Fast approximate similarity
        approx_sim = fp.approximate_similarity(fp_a, fp_b)

        # Index for fast lookup
        fp.index("path/a", vector_a)
        candidates = fp.query(vector_q, threshold=0.7)
    """

    def __init__(
        self,
        dimension: int,
        n_bits: int = 256,
        n_tables: int = 4,
        seed: int = 42,
    ):
        """Initialize the fingerprint system.

        Args:
            dimension: Embedding dimensionality.
            n_bits:    Number of hash bits per fingerprint (more = more precise).
            n_tables:  Number of hash tables for LSH index (more = better recall).
            seed:      Random seed for reproducible hyperplanes.
        """
        self._dim = dimension
        self._n_bits = n_bits
        self._n_tables = n_tables
        self._lock = threading.Lock()

        # Generate random hyperplanes for SimHash
        rng = np.random.RandomState(seed)
        self._hyperplanes = rng.randn(n_bits, dimension).astype(np.float32)
        # Normalize hyperplanes for stability
        norms = np.linalg.norm(self._hyperplanes, axis=1, keepdims=True)
        self._hyperplanes /= (norms + 1e-12)

        # LSH tables: each table uses a different subset of bits
        bits_per_table = max(8, n_bits // n_tables)
        self._table_bit_ranges = []
        for i in range(n_tables):
            start = (i * bits_per_table) % n_bits
            end = min(start + bits_per_table, n_bits)
            self._table_bit_ranges.append((start, end))

        # Index storage: table_id -> hash_key -> set of paths
        self._tables: List[Dict[int, Set[str]]] = [
            {} for _ in range(n_tables)
        ]
        # Path -> fingerprint mapping
        self._fingerprints: Dict[str, np.ndarray] = {}

    def hash(self, vector: np.ndarray) -> np.ndarray:
        """Compute the binary fingerprint of a vector.

        Returns a packed binary array where each bit represents
        which side of a random hyperplane the vector falls on.
        """
        vec = vector.astype(np.float32)
        # Dot product with all hyperplanes simultaneously
        projections = self._hyperplanes @ vec
        # Convert to binary: 1 if positive, 0 if negative
        return (projections > 0).astype(np.uint8)

    def approximate_similarity(
        self,
        fp_a: np.ndarray,
        fp_b: np.ndarray,
    ) -> float:
        """Estimate cosine similarity from two fingerprints.

        Uses the relationship: P(same bit) = 1 - arccos(sim) / pi
        Inverted: sim ≈ cos(pi * (1 - fraction_same))

        This is O(n_bits/64) instead of O(dimension).
        """
        # Hamming agreement: fraction of matching bits
        agreement = float(np.mean(fp_a == fp_b))
        # Convert to estimated cosine similarity
        estimated_sim = np.cos(np.pi * (1.0 - agreement))
        return float(np.clip(estimated_sim, -1.0, 1.0))

    def hamming_distance(self, fp_a: np.ndarray, fp_b: np.ndarray) -> int:
        """Hamming distance between two fingerprints."""
        return int(np.sum(fp_a != fp_b))

    def index(self, path: str, vector: np.ndarray) -> np.ndarray:
        """Add a vector to the LSH index. Returns its fingerprint."""
        fp = self.hash(vector)

        with self._lock:
            # Remove old entry if exists
            self._remove_from_tables(path)

            # Store fingerprint
            self._fingerprints[path] = fp

            # Insert into all LSH tables
            for table_id, (start, end) in enumerate(self._table_bit_ranges):
                key = self._bits_to_key(fp[start:end])
                if key not in self._tables[table_id]:
                    self._tables[table_id][key] = set()
                self._tables[table_id][key].add(path)

        return fp

    def remove(self, path: str) -> bool:
        """Remove a path from the index."""
        with self._lock:
            if path not in self._fingerprints:
                return False
            self._remove_from_tables(path)
            del self._fingerprints[path]
        return True

    def query(
        self,
        vector: np.ndarray,
        threshold: float = 0.5,
        max_results: int = 50,
    ) -> List[Tuple[str, float]]:
        """Find paths with approximate similarity above threshold.

        Uses LSH tables for candidate generation, then re-ranks
        by fingerprint similarity. Much faster than brute force.

        Returns list of (path, estimated_similarity) tuples.
        """
        query_fp = self.hash(vector)

        with self._lock:
            # Candidate generation: union of matches across all tables
            candidates: Set[str] = set()
            for table_id, (start, end) in enumerate(self._table_bit_ranges):
                key = self._bits_to_key(query_fp[start:end])
                if key in self._tables[table_id]:
                    candidates.update(self._tables[table_id][key])

                # Also check neighboring buckets (1-bit flips) for recall
                for bit_offset in range(min(3, end - start)):
                    flipped = query_fp[start:end].copy()
                    flipped[bit_offset] = 1 - flipped[bit_offset]
                    neighbor_key = self._bits_to_key(flipped)
                    if neighbor_key in self._tables[table_id]:
                        candidates.update(self._tables[table_id][neighbor_key])

            # Re-rank by fingerprint similarity
            results = []
            for path in candidates:
                if path not in self._fingerprints:
                    continue
                sim = self.approximate_similarity(query_fp, self._fingerprints[path])
                if sim >= threshold:
                    results.append((path, sim))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]

    def find_near_duplicates(self, threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        """Find pairs of entries that are near-duplicates.

        Returns list of (path_a, path_b, estimated_similarity) tuples.
        """
        with self._lock:
            paths = list(self._fingerprints.keys())
            fps = [self._fingerprints[p] for p in paths]

        duplicates = []
        n = len(paths)
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.approximate_similarity(fps[i], fps[j])
                if sim >= threshold:
                    duplicates.append((paths[i], paths[j], sim))

        return duplicates

    def get_fingerprint(self, path: str) -> Optional[np.ndarray]:
        """Get the stored fingerprint for a path."""
        with self._lock:
            return self._fingerprints.get(path)

    @property
    def indexed_count(self) -> int:
        with self._lock:
            return len(self._fingerprints)

    @property
    def n_bits(self) -> int:
        return self._n_bits

    # --- Internal ---

    def _remove_from_tables(self, path: str):
        """Remove a path from all LSH tables (must hold lock)."""
        if path in self._fingerprints:
            fp = self._fingerprints[path]
            for table_id, (start, end) in enumerate(self._table_bit_ranges):
                key = self._bits_to_key(fp[start:end])
                if key in self._tables[table_id]:
                    self._tables[table_id][key].discard(path)

    @staticmethod
    def _bits_to_key(bits: np.ndarray) -> int:
        """Convert a bit array to an integer key."""
        # Pack bits into an integer (up to 64 bits for dict key efficiency)
        key = 0
        for i, b in enumerate(bits[:64]):
            if b:
                key |= (1 << i)
        return key


class BloomFilter:
    """Approximate set membership for semantic concepts.

    "Have we seen knowledge about this topic before?"
    O(1) check with configurable false positive rate.
    Used by the reasoning layer for fast duplicate/novelty detection.
    """

    def __init__(self, capacity: int = 10000, fp_rate: float = 0.01):
        self._capacity = capacity
        self._fp_rate = fp_rate

        # Calculate optimal bit array size and hash count
        # m = -n*ln(p) / (ln2)^2
        self._m = max(64, int(-capacity * np.log(fp_rate) / (np.log(2) ** 2)))
        # k = (m/n) * ln2
        self._k = max(1, int((self._m / capacity) * np.log(2)))

        self._bits = np.zeros(self._m, dtype=np.uint8)
        self._count = 0
        self._lock = threading.Lock()

    def add(self, fingerprint: np.ndarray) -> None:
        """Add a semantic fingerprint to the bloom filter."""
        with self._lock:
            for idx in self._get_indices(fingerprint):
                self._bits[idx] = 1
            self._count += 1

    def might_contain(self, fingerprint: np.ndarray) -> bool:
        """Check if we might have seen this semantic fingerprint before.

        False positives possible. False negatives impossible.
        """
        with self._lock:
            return all(self._bits[idx] == 1 for idx in self._get_indices(fingerprint))

    def _get_indices(self, fingerprint: np.ndarray) -> List[int]:
        """Generate k hash indices from a fingerprint."""
        # Use the fingerprint bits directly as hash input
        data = fingerprint.tobytes()
        indices = []
        for i in range(self._k):
            h = hashlib.md5(data + struct.pack('I', i)).digest()
            idx = int.from_bytes(h[:4], 'little') % self._m
            indices.append(idx)
        return indices

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def fill_ratio(self) -> float:
        """Fraction of bits set to 1."""
        with self._lock:
            return float(np.mean(self._bits))
