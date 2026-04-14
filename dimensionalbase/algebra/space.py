"""
DimensionalSpace — analytics layer on top of VectorStore.

v0.3: No longer stores vectors itself. Delegates to the shared VectorStore.
Keeps only: running statistics (Welford), covariance tracking, clustering,
manifold analysis. All in float64 for precision.

The VectorStore handles storage (float32, contiguous, BLAS-fast).
The DimensionalSpace handles understanding (float64, statistical, geometric).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from dimensionalbase.storage.vectors import VectorStore

logger = logging.getLogger("dimensionalbase.algebra.space")


@dataclass
class RegionStats:
    center: np.ndarray
    radius: float
    density: float
    count: int
    variance: float
    intrinsic_dim: float
    paths: List[str] = field(default_factory=list)


@dataclass
class SpaceMetrics:
    total_points: int
    ambient_dimension: int
    intrinsic_dimension_estimate: float
    mean_pairwise_similarity: float
    coverage: float
    cluster_count: int
    isolation_score: float
    last_updated: float = field(default_factory=time.time)


class DimensionalSpace:
    """Analytics layer over the shared VectorStore.

    Tracks manifold geometry, detects clusters, estimates intrinsic
    dimensionality, computes novelty. All vectors live in VectorStore.
    """

    def __init__(self, dimension: int = 0, vector_store: Optional[VectorStore] = None,
                 max_history: int = 10000, merge_threshold: float = 0.7):
        # Accept either a shared VectorStore or create one internally
        if vector_store is not None:
            self._store = vector_store
            self._dim = vector_store.dimension
        elif dimension > 0:
            self._store = VectorStore(dimension=dimension)
            self._dim = dimension
        else:
            raise ValueError("Either dimension or vector_store must be provided")

        self._lock = threading.RLock()

        # Running statistics (float64 for precision)
        self._mean = np.zeros(self._dim, dtype=np.float64)
        self._M2 = np.zeros(self._dim, dtype=np.float64)
        self._n = 0
        self._cov_sum = np.zeros((self._dim, self._dim), dtype=np.float64)

        self._merge_threshold = merge_threshold

        # Cluster tracking
        self._cluster_dirty = True
        self._last_cluster_update = 0.0
        self._cluster_update_interval = 5.0
        self._cached_clusters: List[RegionStats] = []
        self._cluster_assignments: Dict[str, int] = {}

        if self._store.count:
            self.refresh_from_store()

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def count(self) -> int:
        return self._store.count

    # ── Write Operations ──────────────────────────────────────

    def add(self, path: str, vector: np.ndarray) -> Dict[str, float]:
        """Add a point and compute novelty metrics.

        The vector is stored in VectorStore (float32, normalized).
        Stats are updated here in float64.
        """
        if self._store.contains(path):
            self._store.add(path, vector)
            self.refresh_from_store()
            return self._compute_novelty_metrics(VectorStore._normalize(vector))

        vec_f64 = np.asarray(vector, dtype=np.float64).ravel()
        norm = np.linalg.norm(vec_f64)
        if norm > 1e-12:
            vec_f64 = vec_f64 / norm

        # Novelty BEFORE adding
        metrics = self._compute_novelty_metrics(vec_f64.astype(np.float32))

        # Store in VectorStore (handles float32 normalization)
        self._store.add(path, vector)

        # Update running stats
        with self._lock:
            self._n += 1
            delta = vec_f64 - self._mean
            self._mean += delta / self._n
            delta2 = vec_f64 - self._mean
            self._M2 += delta * delta2
            if self._n > 1:
                self._cov_sum += np.outer(delta, delta2)
            self._cluster_dirty = True

        return metrics

    def add_fast(self, path: str, vector: np.ndarray) -> None:
        """Stats-only update. Skips novelty computation.

        Used when bloom filter indicates this is a known topic.
        """
        if self._store.contains(path):
            self._store.add(path, vector)
            self.refresh_from_store()
            return

        vec_f64 = np.asarray(vector, dtype=np.float64).ravel()
        norm = np.linalg.norm(vec_f64)
        if norm > 1e-12:
            vec_f64 = vec_f64 / norm

        self._store.add(path, vector)

        with self._lock:
            self._n += 1
            delta = vec_f64 - self._mean
            self._mean += delta / self._n
            delta2 = vec_f64 - self._mean
            self._M2 += delta * delta2
            if self._n > 1:
                self._cov_sum += np.outer(delta, delta2)
            self._cluster_dirty = True

    def remove(self, path: str) -> bool:
        removed = self._store.remove(path)
        if removed:
            self.refresh_from_store()
        return removed

    def get(self, path: str) -> Optional[np.ndarray]:
        return self._store.get(path)

    # ── Similarity ────────────────────────────────────────────

    def similarity(self, path_a: str, path_b: str) -> Optional[float]:
        """Pre-normalized dot product = cosine similarity."""
        return self._store.pairwise_similarity(path_a, path_b)

    def relationship_type(self, path_a: str, path_b: str) -> Optional[Dict[str, float]]:
        a = self._store.get(path_a)
        b = self._store.get(path_b)
        if a is None or b is None:
            return None

        a64 = a.astype(np.float64)
        b64 = b.astype(np.float64)
        cos_sim = float(np.dot(a64, b64))
        diff = a64 - b64

        similarity = max(0.0, cos_sim)
        contradictory = max(0.0, -cos_sim)
        complementary = min(1.0, float(np.linalg.norm(diff)) / (math.sqrt(2) + 1e-10)) * (1 - abs(cos_sim))
        proj = cos_sim
        residual_a = float(np.linalg.norm(a64 - proj * b64))
        residual_b = float(np.linalg.norm(b64 - proj * a64))
        hierarchical = abs(residual_a - residual_b) * similarity
        independent = max(0.0, 1.0 - abs(cos_sim) * 3.0)

        return {
            "similarity": round(similarity, 4),
            "complementary": round(complementary, 4),
            "hierarchical": round(hierarchical, 4),
            "contradictory": round(contradictory, 4),
            "independent": round(independent, 4),
        }

    # ── Search ────────────────────────────────────────────────

    def search(self, query: np.ndarray, k: int = 10,
               diversity_factor: float = 0.0) -> List[Tuple[str, float]]:
        if diversity_factor <= 0:
            return self._store.search(query, k=k)

        # MMR re-ranking for diversity
        candidates = self._store.search(query, k=min(k * 3, self._store.count))
        if len(candidates) <= k:
            return candidates

        q_norm = VectorStore._normalize(query)
        selected: List[Tuple[str, float]] = []
        remaining = list(candidates)

        for _ in range(min(k, len(candidates))):
            best_score = -float('inf')
            best_idx = 0
            for i, (path, rel_score) in enumerate(remaining):
                if selected:
                    max_sim = max(
                        self._store.pairwise_similarity(path, sp) or 0.0
                        for sp, _ in selected
                    )
                else:
                    max_sim = 0.0
                mmr = (1 - diversity_factor) * rel_score - diversity_factor * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
            selected.append(remaining.pop(best_idx))

        return selected

    # ── Manifold Analysis ─────────────────────────────────────

    def estimate_intrinsic_dimension(self) -> float:
        matrix, paths = self._store.get_active_data()
        n = len(paths)
        if n < 10:
            return float(self._dim)

        k = min(5, n - 1)
        sims = matrix @ matrix.T
        np.fill_diagonal(sims, -2)
        distances = np.arccos(np.clip(sims, -1, 1))

        dims = []
        for i in range(n):
            dists_i = np.sort(distances[i])[:k + 1]
            dists_i = dists_i[dists_i > 1e-10]
            if len(dists_i) < 2:
                continue
            T_k = dists_i[-1]
            if T_k < 1e-10:
                continue
            log_sum = sum(math.log(T_k / d) for d in dists_i[:-1])
            if log_sum > 0:
                dims.append((len(dists_i) - 1) / log_sum)
        return float(np.median(dims)) if dims else float(self._dim)

    def get_principal_components(self, n_components: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if self._n < 2:
                return np.eye(min(n_components, self._dim)), np.ones(min(n_components, self._dim))
            cov = self._cov_sum / (self._n - 1)
        n_components = min(n_components, self._dim)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1][:n_components]
        components = eigenvectors[:, idx].T
        total_var = np.sum(eigenvalues)
        ratios = eigenvalues[idx] / total_var if total_var > 0 else np.ones(n_components) / n_components
        return components, ratios

    def detect_clusters(self, min_cluster_size: int = 3) -> List[RegionStats]:
        now = time.time()
        with self._lock:
            if not self._cluster_dirty and (now - self._last_cluster_update < self._cluster_update_interval):
                return self._cached_clusters

        matrix, paths = self._store.get_active_data()
        n = len(paths)
        if n < min_cluster_size:
            return []

        sims = matrix @ matrix.T
        labels = list(range(n))
        cluster_members: Dict[int, List[int]] = {i: [i] for i in range(n)}
        merge_threshold = self._merge_threshold

        for _ in range(n - 1):
            best_sim = -1.0
            best_i, best_j = -1, -1
            active = sorted(cluster_members.keys())
            for ci_idx, ci in enumerate(active):
                for cj in active[ci_idx + 1:]:
                    total_sim = sum(sims[mi, mj] for mi in cluster_members[ci] for mj in cluster_members[cj])
                    count = len(cluster_members[ci]) * len(cluster_members[cj])
                    avg = total_sim / count if count else 0
                    if avg > best_sim:
                        best_sim = avg
                        best_i, best_j = ci, cj
            if best_sim < merge_threshold:
                break
            cluster_members[best_i].extend(cluster_members[best_j])
            for idx in cluster_members[best_j]:
                labels[idx] = best_i
            del cluster_members[best_j]

        regions = []
        for cluster_id, members in cluster_members.items():
            if len(members) < min_cluster_size:
                continue
            member_vecs = matrix[members]
            center = np.mean(member_vecs, axis=0)
            cnorm = np.linalg.norm(center)
            if cnorm > 1e-12:
                center /= cnorm
            dists = np.arccos(np.clip(member_vecs @ center, -1, 1))
            regions.append(RegionStats(
                center=center, radius=float(np.max(dists)),
                density=len(members) / (float(np.max(dists)) + 1e-10),
                count=len(members),
                variance=float(np.mean(np.var(member_vecs, axis=0))),
                intrinsic_dim=self._local_dim(member_vecs),
                paths=[paths[m] for m in members],
            ))

        with self._lock:
            self._cached_clusters = regions
            self._cluster_dirty = False
            self._last_cluster_update = now
            self._cluster_assignments = {paths[i]: labels[i] for i in range(n)}
        return regions

    def find_voids(self, n_probes: int = 50) -> List[np.ndarray]:
        matrix, _ = self._store.get_active_data()
        if len(matrix) < 3:
            return []
        probes = np.random.randn(n_probes, self._dim).astype(np.float32)
        probes /= np.linalg.norm(probes, axis=1, keepdims=True)
        max_sims = np.max(np.abs(matrix @ probes.T), axis=0)
        return [probes[i] for i in np.where(max_sims < 0.3)[0]]

    def metrics(self) -> SpaceMetrics:
        n = self._store.count
        if n == 0:
            return SpaceMetrics(0, self._dim, 0, 0, 0, 0, 0)

        matrix, _ = self._store.get_active_data()
        sims = matrix @ matrix.T
        np.fill_diagonal(sims, 0)
        mean_sim = float(np.sum(sims)) / max(1, n * (n - 1))

        coverage = 0.0
        if n > 1:
            _, ratios = self.get_principal_components(min(10, self._dim))
            entropy = -float(np.sum(ratios * np.log(ratios + 1e-12)))
            max_entropy = math.log(len(ratios))
            coverage = entropy / max_entropy if max_entropy > 0 else 0

        isolation = 1.0
        if n > 1:
            np.fill_diagonal(sims, -2)
            max_sims = np.max(sims, axis=1)
            isolation = float(np.mean(max_sims < 0.3))

        clusters = self.detect_clusters()
        return SpaceMetrics(
            total_points=n, ambient_dimension=self._dim,
            intrinsic_dimension_estimate=self.estimate_intrinsic_dimension() if n >= 10 else float(self._dim),
            mean_pairwise_similarity=mean_sim, coverage=coverage,
            cluster_count=len(clusters), isolation_score=isolation,
        )

    def refresh_from_store(self) -> None:
        """Recompute statistics from the shared vector store."""
        matrix, _ = self._store.get_active_data()
        with self._lock:
            self._n = len(matrix)
            self._mean = np.zeros(self._dim, dtype=np.float64)
            self._M2 = np.zeros(self._dim, dtype=np.float64)
            self._cov_sum = np.zeros((self._dim, self._dim), dtype=np.float64)
            self._cluster_dirty = True
            self._cached_clusters = []
            self._cluster_assignments = {}
            self._last_cluster_update = 0.0

            if self._n == 0:
                return

            matrix64 = matrix.astype(np.float64)
            self._mean = np.mean(matrix64, axis=0)
            if self._n > 1:
                centered = matrix64 - self._mean
                self._M2 = np.sum(centered * centered, axis=0)
                self._cov_sum = centered.T @ centered

    # ── Novelty ───────────────────────────────────────────────

    def _compute_novelty_metrics(self, vec: np.ndarray) -> Dict[str, float]:
        if self._store.count == 0:
            return {"novelty": 1.0, "density": 0.0, "nearest_distance": float('inf'), "information_gain": 1.0}

        # Single BLAS call via VectorStore
        sims = self._store.all_similarities(vec)
        max_sim = float(np.max(sims))
        nearest_dist = float(np.arccos(np.clip(max_sim, -1, 1)))
        novelty = 1.0 - max(0.0, max_sim)
        close_count = int(np.sum(sims > 0.7))
        density = close_count / max(1, len(sims))
        info_gain = novelty * (1.0 - density)

        return {
            "novelty": round(novelty, 4), "density": round(density, 4),
            "nearest_distance": round(nearest_dist, 4), "information_gain": round(info_gain, 4),
        }

    @staticmethod
    def _local_dim(vectors: np.ndarray) -> float:
        if len(vectors) < 3:
            return float(vectors.shape[1])
        centered = vectors - np.mean(vectors, axis=0)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        s = s[s > 1e-10]
        if len(s) == 0:
            return 0.0
        s_sq = s ** 2
        return float((np.sum(s_sq) ** 2) / (np.sum(s_sq ** 2) + 1e-12))

    def geodesic_distance(self, path_a: str, path_b: str, k: int = 5) -> Optional[float]:
        a = self._store.get(path_a)
        b = self._store.get(path_b)
        if a is None or b is None:
            return None

        matrix, paths_list = self._store.get_active_data()
        n = len(paths_list)
        if n < 3:
            return float(np.arccos(np.clip(float(np.dot(a, b)), -1, 1)))

        sims = matrix @ matrix.T
        idx_a = paths_list.index(path_a)
        idx_b = paths_list.index(path_b)
        k = min(k, n - 1)
        INF = float('inf')
        dist = [INF] * n
        dist[idx_a] = 0.0
        visited = [False] * n

        for _ in range(n):
            u = -1
            for i in range(n):
                if not visited[i] and (u == -1 or dist[i] < dist[u]):
                    u = i
            if u == -1 or dist[u] == INF:
                break
            visited[u] = True
            if u == idx_b:
                break
            neighbors = np.argpartition(sims[u], -k - 1)[-k - 1:]
            for v in neighbors:
                v = int(v)
                if v == u or visited[v]:
                    continue
                ang = float(np.arccos(np.clip(sims[u, v], -1.0, 1.0)))
                if dist[u] + ang < dist[v]:
                    dist[v] = dist[u] + ang

        return dist[idx_b] if dist[idx_b] != INF else None
