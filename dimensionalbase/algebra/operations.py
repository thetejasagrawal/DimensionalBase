"""
Dimensional algebra operations — the mathematical core of DimensionalBase.

These are the five operations from the vision doc, plus deeper primitives
that make the system genuinely hard to replicate:

  COMPOSE:    Merge multiple agents' knowledge into a unified representation
  RELATE:     Discover the mathematical relationship between any two points
  PROJECT:    Map a point into a lower-dimensional subspace
  INTERPOLATE: Find semantic midpoints between concepts
  DECOMPOSE:  Factor a point into interpretable components

Plus:
  centroid:              Weighted geometric mean of multiple points
  orthogonal_complement: Find what's NOT covered by a set of vectors
  analogy:               Vector arithmetic for semantic analogies
  subspace_alignment:    Measure how aligned two sets of knowledge are
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def compose(
    vectors: Sequence[np.ndarray],
    weights: Optional[Sequence[float]] = None,
    mode: str = "weighted_mean",
) -> np.ndarray:
    """Merge multiple knowledge vectors into a unified representation.

    Modes:
      - 'weighted_mean': Weighted average (default). Simple, fast, works.
      - 'principal':     Project onto the first principal component. Captures
                         the dominant shared direction. Good for finding consensus.
      - 'grassmann':     Grassmann mean on the unit sphere. Preserves angular
                         geometry better than Euclidean mean for normalized vectors.
      - 'attentive':     Soft attention over vectors using mutual similarity
                         as attention weights. Up-weights vectors that agree.

    Args:
        vectors: Sequence of embedding vectors to compose.
        weights: Optional per-vector weights (default: uniform).
        mode:    Composition strategy.

    Returns:
        Composed vector (normalized to unit length).
    """
    if len(vectors) == 0:
        raise ValueError("Cannot compose empty sequence")
    if len(vectors) == 1:
        v = vectors[0].astype(np.float64)
        return v / (np.linalg.norm(v) + 1e-12)

    matrix = np.stack([v.astype(np.float64) for v in vectors])

    if weights is not None:
        w = np.array(weights, dtype=np.float64)
        w = w / (np.sum(w) + 1e-12)
    else:
        w = np.ones(len(vectors)) / len(vectors)

    if mode == "weighted_mean":
        result = np.sum(matrix * w[:, np.newaxis], axis=0)

    elif mode == "principal":
        # Weighted PCA: first principal component of weighted data
        centered = matrix - np.mean(matrix * w[:, np.newaxis], axis=0)
        weighted = centered * np.sqrt(w[:, np.newaxis])
        _, _, Vt = np.linalg.svd(weighted, full_matrices=False)
        # Sign: align with the weighted mean direction
        mean_dir = np.sum(matrix * w[:, np.newaxis], axis=0)
        if np.dot(Vt[0], mean_dir) < 0:
            Vt[0] = -Vt[0]
        result = Vt[0]

    elif mode == "grassmann":
        # Iterative Grassmann mean (Karcher mean on the sphere)
        result = matrix[0].copy()
        for _ in range(10):  # iterations
            # Log map: project tangent vectors
            tangents = np.zeros_like(result)
            for i, v in enumerate(matrix):
                cos_d = np.clip(np.dot(result, v), -1, 1)
                if abs(cos_d) > 0.9999:
                    continue
                angle = np.arccos(cos_d)
                direction = v - cos_d * result
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-10:
                    tangents += w[i] * angle * direction / direction_norm

            # Exp map: move along the geodesic
            tangent_norm = np.linalg.norm(tangents)
            if tangent_norm < 1e-10:
                break
            result = result * np.cos(tangent_norm) + (tangents / tangent_norm) * np.sin(tangent_norm)

    elif mode == "attentive":
        # Mutual attention: weight each vector by how much others agree with it
        sims = matrix @ matrix.T  # pairwise similarities
        np.fill_diagonal(sims, 0)
        attention = np.sum(sims, axis=1)  # total agreement score
        attention = np.exp(attention - np.max(attention))  # softmax
        attention = attention / (np.sum(attention) + 1e-12)
        # Combine base weights with attention
        combined_w = w * attention
        combined_w = combined_w / (np.sum(combined_w) + 1e-12)
        result = np.sum(matrix * combined_w[:, np.newaxis], axis=0)

    else:
        raise ValueError(f"Unknown composition mode: {mode}")

    norm = np.linalg.norm(result)
    if norm < 1e-12:
        return matrix[0] / (np.linalg.norm(matrix[0]) + 1e-12)
    return result / norm


def relate(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Discover the mathematical relationship between two points.

    Returns a rich characterization, not just a single similarity number:
      - cosine:       Standard cosine similarity
      - angular_dist: Angular distance in radians
      - projection:   How much of b is "explained by" a
      - residual:     How much of b is orthogonal to a
      - parallelism:  Are they pointing the same direction? (0-1)
      - opposition:   Are they pointing opposite directions? (0-1)
      - independence: Are they orthogonal? (0-1)
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < 1e-12 or b_norm < 1e-12:
        return {k: 0.0 for k in
                ["cosine", "angular_dist", "projection", "residual",
                 "parallelism", "opposition", "independence"]}

    a_hat = a / a_norm
    b_hat = b / b_norm
    cos = float(np.dot(a_hat, b_hat))
    cos_clipped = np.clip(cos, -1, 1)

    # Projection of b onto a
    proj_scalar = float(np.dot(b, a_hat))
    proj_vec = proj_scalar * a_hat
    residual_vec = b - proj_vec
    residual_mag = float(np.linalg.norm(residual_vec)) / (b_norm + 1e-12)

    return {
        "cosine": round(cos, 6),
        "angular_dist": round(float(np.arccos(cos_clipped)), 6),
        "projection": round(abs(proj_scalar) / (b_norm + 1e-12), 6),
        "residual": round(residual_mag, 6),
        "parallelism": round(max(0, cos), 6),
        "opposition": round(max(0, -cos), 6),
        "independence": round(1.0 - abs(cos), 6),
    }


def project(
    vector: np.ndarray,
    subspace_basis: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Project a vector into a subspace defined by an orthonormal basis.

    Returns (projected_vector, information_retained).

    The information_retained value (0-1) tells you how much of the
    original meaning survives the projection. If it's 0.3, you're
    losing 70% of the information by restricting to this subspace.

    Args:
        vector:          The vector to project.
        subspace_basis:  Matrix where each row is a basis vector.

    Returns:
        (projected_vector, information_retained)
    """
    v = vector.astype(np.float64)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return np.zeros_like(v), 0.0

    basis = subspace_basis.astype(np.float64)

    # Orthonormalize the basis (Gram-Schmidt)
    Q, _ = np.linalg.qr(basis.T)  # columns of Q are orthonormal basis
    k = min(basis.shape[0], Q.shape[1])
    Q = Q[:, :k]

    # Project
    coefficients = Q.T @ v
    projected = Q @ coefficients

    # Information retained: ratio of norms
    proj_norm = np.linalg.norm(projected)
    info_retained = (proj_norm / v_norm) ** 2  # squared for energy

    # Normalize
    if proj_norm > 1e-12:
        projected = projected / proj_norm

    return projected, float(info_retained)


def interpolate(
    a: np.ndarray,
    b: np.ndarray,
    t: float = 0.5,
    mode: str = "slerp",
) -> np.ndarray:
    """Find a point between two vectors in the dimensional space.

    Modes:
      - 'slerp':  Spherical linear interpolation. Follows the great circle
                   on the unit sphere. This is the geometrically correct
                   interpolation for normalized vectors.
      - 'lerp':   Linear interpolation + renormalize. Faster but
                   doesn't preserve angular distances.

    At t=0 returns a, at t=1 returns b, at t=0.5 returns the midpoint.

    Args:
        a:    Start vector.
        b:    End vector.
        t:    Interpolation parameter (0-1).
        mode: 'slerp' or 'lerp'.

    Returns:
        Interpolated vector (normalized).
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)

    if mode == "slerp":
        dot = np.clip(float(np.dot(a, b)), -1.0, 1.0)

        # Handle near-parallel vectors
        if abs(dot) > 0.9999:
            result = a + t * (b - a)
            return result / (np.linalg.norm(result) + 1e-12)

        theta = np.arccos(dot)
        sin_theta = np.sin(theta)

        result = (np.sin((1 - t) * theta) / sin_theta) * a + (np.sin(t * theta) / sin_theta) * b

    elif mode == "lerp":
        result = (1 - t) * a + t * b

    else:
        raise ValueError(f"Unknown interpolation mode: {mode}")

    norm = np.linalg.norm(result)
    if norm < 1e-12:
        return a
    return result / norm


def decompose(
    vector: np.ndarray,
    basis_vectors: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Decompose a vector into interpretable components.

    Given a set of named basis directions (e.g., {"code": ..., "business": ...,
    "security": ...}), compute how much of the input vector aligns with each.

    This is like spectral decomposition but in a semantic space.
    It tells you: "this knowledge is 60% about code, 25% about security,
    and 15% about business."

    Args:
        vector:        The vector to decompose.
        basis_vectors: Named direction vectors.

    Returns:
        Dict of component_name -> magnitude (sums to ~1.0 if basis spans space).
    """
    v = vector.astype(np.float64)
    v = v / (np.linalg.norm(v) + 1e-12)

    components = {}
    total = 0.0

    for name, basis in basis_vectors.items():
        b = basis.astype(np.float64)
        b = b / (np.linalg.norm(b) + 1e-12)
        score = float(np.dot(v, b)) ** 2  # Squared projection for non-negative
        components[name] = score
        total += score

    # Normalize so components sum to 1 (approximately)
    if total > 0:
        for name in components:
            components[name] = round(components[name] / total, 4)

    return components


def centroid(
    vectors: Sequence[np.ndarray],
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Compute the weighted centroid of a set of vectors on the unit sphere.

    This is the "average meaning" of a set of knowledge entries.
    Uses Euclidean centroid + renormalization (fast and good enough
    for most cases; use compose(mode='grassmann') for precision).
    """
    if len(vectors) == 0:
        raise ValueError("Cannot compute centroid of empty set")

    matrix = np.stack([v.astype(np.float64) for v in vectors])

    if weights is not None:
        w = np.array(weights, dtype=np.float64)
        w = w / (np.sum(w) + 1e-12)
        result = np.sum(matrix * w[:, np.newaxis], axis=0)
    else:
        result = np.mean(matrix, axis=0)

    norm = np.linalg.norm(result)
    if norm < 1e-12:
        return matrix[0] / (np.linalg.norm(matrix[0]) + 1e-12)
    return result / norm


def orthogonal_complement(
    vectors: Sequence[np.ndarray],
    query: np.ndarray,
) -> np.ndarray:
    """Find the component of query that's NOT explained by the given vectors.

    This answers: "What is this knowledge about that the existing
    knowledge doesn't cover?"

    Returns the residual vector after projecting out all given directions.
    A large residual means the query brings genuinely new information.
    """
    q = query.astype(np.float64)

    if len(vectors) == 0:
        return q / (np.linalg.norm(q) + 1e-12)

    # Build orthonormal basis from the given vectors
    matrix = np.stack([v.astype(np.float64) for v in vectors])
    Q, _ = np.linalg.qr(matrix.T)
    k = min(matrix.shape[0], Q.shape[1])
    Q = Q[:, :k]

    # Project out
    projection = Q @ (Q.T @ q)
    residual = q - projection

    norm = np.linalg.norm(residual)
    if norm < 1e-12:
        return np.zeros_like(q)
    return residual / norm


def analogy(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """Semantic analogy: a is to b as c is to ???

    Classic vector arithmetic: result = c + (b - a)
    Normalized to unit sphere.

    Example: if a=king, b=queen, c=man, result≈woman
    In DimensionalBase: if a=auth_error, b=auth_fix,
    c=deploy_error, result≈deploy_fix
    """
    a, b, c = [x.astype(np.float64) for x in (a, b, c)]
    a, b, c = [x / (np.linalg.norm(x) + 1e-12) for x in (a, b, c)]

    result = c + (b - a)
    norm = np.linalg.norm(result)
    if norm < 1e-12:
        return c
    return result / norm


def subspace_alignment(
    vectors_a: Sequence[np.ndarray],
    vectors_b: Sequence[np.ndarray],
    n_components: int = 5,
) -> float:
    """Measure how aligned two sets of knowledge are.

    Returns a score from 0 (completely different topics) to 1
    (covering the same semantic subspace).

    Uses principal angle analysis between the two subspaces.

    Args:
        vectors_a:    First set of vectors (e.g., Agent A's knowledge).
        vectors_b:    Second set (e.g., Agent B's knowledge).
        n_components: Number of principal components to compare.

    Returns:
        Alignment score (0-1).
    """
    if len(vectors_a) < 2 or len(vectors_b) < 2:
        return 0.0

    mat_a = np.stack([v.astype(np.float64) for v in vectors_a])
    mat_b = np.stack([v.astype(np.float64) for v in vectors_b])

    k = min(n_components, mat_a.shape[0], mat_b.shape[0], mat_a.shape[1])

    # PCA of each set
    _, _, Vt_a = np.linalg.svd(mat_a - np.mean(mat_a, axis=0), full_matrices=False)
    _, _, Vt_b = np.linalg.svd(mat_b - np.mean(mat_b, axis=0), full_matrices=False)

    # Principal angles via SVD of cross-product
    cross = Vt_a[:k] @ Vt_b[:k].T
    _, sigmas, _ = np.linalg.svd(cross)

    # Alignment = mean cosine of principal angles
    alignment = float(np.mean(np.clip(sigmas[:k], 0, 1)))
    return round(alignment, 4)
