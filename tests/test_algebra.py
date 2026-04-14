"""
Tests for the dimensional algebra engine.

These test the mathematical core that makes DimensionalBase
fundamentally different from a vector database.
"""

import numpy as np
import pytest

from dimensionalbase.algebra.space import DimensionalSpace
from dimensionalbase.algebra.operations import (
    compose, relate, project, interpolate, decompose,
    centroid, orthogonal_complement, analogy, subspace_alignment,
)
from dimensionalbase.algebra.fingerprint import SemanticFingerprint, BloomFilter


class TestDimensionalSpace:
    """Test the shared meaning space."""

    def test_add_and_retrieve(self):
        space = DimensionalSpace(dimension=64)
        vec = np.random.randn(64).astype(np.float32)
        metrics = space.add("test/a", vec)

        assert "novelty" in metrics
        assert "information_gain" in metrics
        assert space.count == 1

        retrieved = space.get("test/a")
        assert retrieved is not None

    def test_novelty_decreases_with_similar_vectors(self):
        space = DimensionalSpace(dimension=64)
        base = np.random.randn(64).astype(np.float32)

        m1 = space.add("a", base)
        # Nearly identical vector
        similar = base + np.random.randn(64).astype(np.float32) * 0.01
        m2 = space.add("b", similar)

        assert m2["novelty"] < m1["novelty"]

    def test_novelty_high_for_orthogonal(self):
        space = DimensionalSpace(dimension=64)
        v1 = np.zeros(64, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(64, dtype=np.float32)
        v2[32] = 1.0

        space.add("a", v1)
        metrics = space.add("b", v2)
        assert metrics["novelty"] > 0.5

    def test_similarity(self):
        space = DimensionalSpace(dimension=64)
        v1 = np.random.randn(64).astype(np.float32)
        v2 = v1 + np.random.randn(64).astype(np.float32) * 0.1

        space.add("a", v1)
        space.add("b", v2)

        sim = space.similarity("a", "b")
        assert sim is not None
        assert sim > 0.8

    def test_search_with_diversity(self):
        space = DimensionalSpace(dimension=64)
        # Add 3 clusters
        for i in range(10):
            v = np.zeros(64, dtype=np.float32)
            v[0] = 1.0
            v += np.random.randn(64).astype(np.float32) * 0.05
            space.add(f"cluster1/{i}", v)

        for i in range(10):
            v = np.zeros(64, dtype=np.float32)
            v[32] = 1.0
            v += np.random.randn(64).astype(np.float32) * 0.05
            space.add(f"cluster2/{i}", v)

        query = np.ones(64, dtype=np.float32) / np.sqrt(64)
        results_no_div = space.search(query, k=5, diversity_factor=0.0)
        results_div = space.search(query, k=5, diversity_factor=0.8)

        assert len(results_no_div) == 5
        assert len(results_div) == 5

    def test_relationship_type(self):
        space = DimensionalSpace(dimension=64)
        v1 = np.random.randn(64).astype(np.float32)
        v2 = v1.copy()  # Same direction
        v3 = -v1  # Opposite direction
        v4 = np.random.randn(64).astype(np.float32)  # Random

        space.add("same", v1)
        space.add("similar", v2)
        space.add("opposite", v3)
        space.add("random", v4)

        rel_similar = space.relationship_type("same", "similar")
        rel_opposite = space.relationship_type("same", "opposite")

        assert rel_similar["similarity"] > 0.9
        assert rel_opposite["contradictory"] > 0.9

    def test_detect_clusters(self):
        space = DimensionalSpace(dimension=32)

        # Create two distinct clusters
        for i in range(5):
            v = np.zeros(32, dtype=np.float32)
            v[0] = 1.0
            v += np.random.randn(32).astype(np.float32) * 0.02
            space.add(f"A/{i}", v)

        for i in range(5):
            v = np.zeros(32, dtype=np.float32)
            v[16] = 1.0
            v += np.random.randn(32).astype(np.float32) * 0.02
            space.add(f"B/{i}", v)

        clusters = space.detect_clusters(min_cluster_size=3)
        assert len(clusters) >= 2

    def test_metrics(self):
        space = DimensionalSpace(dimension=32)
        for i in range(20):
            space.add(f"p/{i}", np.random.randn(32).astype(np.float32))

        metrics = space.metrics()
        assert metrics.total_points == 20
        assert metrics.ambient_dimension == 32
        assert 0 <= metrics.coverage <= 1
        assert 0 <= metrics.isolation_score <= 1

    def test_remove(self):
        space = DimensionalSpace(dimension=16)
        space.add("a", np.random.randn(16).astype(np.float32))
        assert space.count == 1
        assert space.remove("a")
        assert space.count == 0
        assert not space.remove("nonexistent")


class TestOperations:
    """Test the dimensional algebra operations."""

    def test_compose_weighted_mean(self):
        v1 = np.array([1, 0, 0], dtype=np.float64)
        v2 = np.array([0, 1, 0], dtype=np.float64)
        result = compose([v1, v2], mode="weighted_mean")
        assert result.shape == (3,)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_compose_attentive(self):
        # Three vectors where two agree and one disagrees
        v1 = np.array([1, 0, 0], dtype=np.float64)
        v2 = np.array([0.9, 0.1, 0], dtype=np.float64)
        v3 = np.array([0, 0, 1], dtype=np.float64)

        result = compose([v1, v2, v3], mode="attentive")
        # Attentive mode should weight toward the agreeing pair
        assert result[0] > result[2]

    def test_compose_grassmann(self):
        v1 = np.random.randn(32)
        v2 = np.random.randn(32)
        result = compose([v1, v2], mode="grassmann")
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_relate(self):
        v1 = np.array([1, 0, 0], dtype=np.float64)
        v2 = np.array([0.9, 0.1, 0], dtype=np.float64)

        rel = relate(v1, v2)
        assert "cosine" in rel
        assert "angular_dist" in rel
        assert "parallelism" in rel
        assert "opposition" in rel
        assert "independence" in rel
        assert rel["parallelism"] > 0.5

    def test_relate_opposite(self):
        v1 = np.array([1, 0, 0], dtype=np.float64)
        v2 = np.array([-1, 0, 0], dtype=np.float64)
        rel = relate(v1, v2)
        assert rel["opposition"] > 0.9
        assert rel["parallelism"] < 0.1

    def test_project(self):
        v = np.array([1, 1, 1], dtype=np.float64)
        basis = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)

        projected, info_retained = project(v, basis)
        assert 0 < info_retained < 1.0  # Lost the z-component
        assert projected.shape == (3,)

    def test_interpolate_slerp(self):
        v1 = np.array([1, 0, 0], dtype=np.float64)
        v2 = np.array([0, 1, 0], dtype=np.float64)

        mid = interpolate(v1, v2, t=0.5, mode="slerp")
        assert abs(np.linalg.norm(mid) - 1.0) < 1e-6
        # Should be equidistant from both
        sim1 = float(np.dot(mid, v1 / np.linalg.norm(v1)))
        sim2 = float(np.dot(mid, v2 / np.linalg.norm(v2)))
        assert abs(sim1 - sim2) < 0.01

    def test_interpolate_endpoints(self):
        v1 = np.random.randn(32)
        v2 = np.random.randn(32)

        start = interpolate(v1, v2, t=0.0)
        end = interpolate(v1, v2, t=1.0)

        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        assert np.allclose(start, v1_norm, atol=1e-6)
        assert np.allclose(end, v2_norm, atol=1e-6)

    def test_decompose(self):
        code_dir = np.array([1, 0, 0, 0], dtype=np.float64)
        biz_dir = np.array([0, 1, 0, 0], dtype=np.float64)
        sec_dir = np.array([0, 0, 1, 0], dtype=np.float64)

        # A vector that's mostly code with some security
        v = np.array([0.9, 0.1, 0.4, 0.0], dtype=np.float64)

        components = decompose(v, {"code": code_dir, "business": biz_dir, "security": sec_dir})
        assert components["code"] > components["business"]
        assert components["code"] > components["security"]
        assert abs(sum(components.values()) - 1.0) < 0.01

    def test_centroid(self):
        vectors = [np.random.randn(16) for _ in range(10)]
        c = centroid(vectors)
        assert c.shape == (16,)
        assert abs(np.linalg.norm(c) - 1.0) < 1e-6

    def test_orthogonal_complement(self):
        v1 = np.array([1, 0, 0], dtype=np.float64)
        query = np.array([1, 1, 0], dtype=np.float64)

        complement = orthogonal_complement([v1], query)
        # Should be orthogonal to v1
        assert abs(np.dot(complement, v1 / np.linalg.norm(v1))) < 0.1

    def test_analogy(self):
        a = np.array([1, 0, 0], dtype=np.float64)
        b = np.array([1, 1, 0], dtype=np.float64)
        c = np.array([0, 0, 1], dtype=np.float64)

        result = analogy(a, b, c)
        assert result.shape == (3,)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_subspace_alignment(self):
        # Same subspace
        v1 = [np.random.randn(16) for _ in range(5)]
        score = subspace_alignment(v1, v1)
        assert score > 0.9

        # Different subspaces
        va = [np.zeros(16) for _ in range(5)]
        vb = [np.zeros(16) for _ in range(5)]
        for i, v in enumerate(va):
            v[i] = 1.0
        for i, v in enumerate(vb):
            v[8 + i] = 1.0
        score = subspace_alignment(va, vb)
        assert score < 0.5


class TestFingerprint:
    """Test semantic fingerprinting (LSH)."""

    def test_similar_vectors_similar_fingerprints(self):
        fp = SemanticFingerprint(dimension=64, n_bits=256)
        v1 = np.random.randn(64).astype(np.float32)
        v2 = v1 + np.random.randn(64).astype(np.float32) * 0.1

        fp1 = fp.hash(v1)
        fp2 = fp.hash(v2)

        sim = fp.approximate_similarity(fp1, fp2)
        assert sim > 0.5  # Should be reasonably similar

    def test_orthogonal_vectors_low_similarity(self):
        fp = SemanticFingerprint(dimension=64, n_bits=256)
        v1 = np.zeros(64, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(64, dtype=np.float32)
        v2[32] = 1.0

        fp1 = fp.hash(v1)
        fp2 = fp.hash(v2)

        sim = fp.approximate_similarity(fp1, fp2)
        assert sim < 0.5

    def test_index_and_query(self):
        fp = SemanticFingerprint(dimension=64, n_bits=128, n_tables=4)

        # Index some vectors
        vectors = {}
        for i in range(20):
            v = np.random.randn(64).astype(np.float32)
            vectors[f"path/{i}"] = v
            fp.index(f"path/{i}", v)

        assert fp.indexed_count == 20

        # Query with one of the indexed vectors
        results = fp.query(vectors["path/5"], threshold=0.3)
        paths = [r[0] for r in results]
        assert "path/5" in paths

    def test_remove_from_index(self):
        fp = SemanticFingerprint(dimension=32, n_bits=64)
        v = np.random.randn(32).astype(np.float32)
        fp.index("test", v)
        assert fp.indexed_count == 1
        assert fp.remove("test")
        assert fp.indexed_count == 0

    def test_near_duplicate_detection(self):
        fp = SemanticFingerprint(dimension=64, n_bits=256)
        base = np.random.randn(64).astype(np.float32)

        fp.index("orig", base)
        fp.index("dup", base + np.random.randn(64).astype(np.float32) * 0.01)
        fp.index("different", np.random.randn(64).astype(np.float32))

        dups = fp.find_near_duplicates(threshold=0.8)
        dup_pairs = [(a, b) for a, b, _ in dups]
        assert any("orig" in pair and "dup" in pair for pair in [(a, b) for a, b, _ in dups])


class TestBloomFilter:
    """Test the semantic bloom filter."""

    def test_add_and_check(self):
        bf = BloomFilter(capacity=1000, fp_rate=0.01)
        fp = np.random.randint(0, 2, size=256).astype(np.uint8)

        bf.add(fp)
        assert bf.might_contain(fp)
        assert bf.count == 1

    def test_false_negative_impossible(self):
        bf = BloomFilter(capacity=1000)
        fps = [np.random.randint(0, 2, size=256).astype(np.uint8) for _ in range(100)]

        for fp in fps:
            bf.add(fp)

        for fp in fps:
            assert bf.might_contain(fp)  # Must be True

    def test_fill_ratio(self):
        bf = BloomFilter(capacity=100)
        for _ in range(50):
            bf.add(np.random.randint(0, 2, size=256).astype(np.uint8))
        assert 0 < bf.fill_ratio < 1
