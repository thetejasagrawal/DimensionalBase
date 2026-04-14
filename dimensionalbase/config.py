"""
Central configuration for DimensionalBase.

All tuneable thresholds live here as a single dataclass.  Pass an instance to
``DimensionalBase(config=...)``; anything you don't override keeps the default
(which matches the prior hard-coded behaviour — zero behaviour change).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DimensionalBaseConfig:
    """Global configuration knobs for a DimensionalBase instance."""

    # ── Active reasoning ───────────────────────────────
    contradiction_threshold: float = 0.75
    staleness_seconds: float = 3600.0
    summary_threshold: int = 10

    # ── Trust engine ───────────────────────────────────
    default_trust: float = 0.5
    trust_k_factor: float = 32.0
    trust_decay_half_life: float = 86400.0  # 24 h
    min_interactions_reliable: int = 5

    # ── Confidence engine ──────────────────────────────
    confidence_decay_half_life: float = 7200.0  # 2 h
    confirmation_weight: float = 1.0
    contradiction_weight: float = 2.0
    propagation_depth: int = 3

    # ── Dimensional space ──────────────────────────────
    cluster_merge_threshold: float = 0.7

    # ── Novelty filter ─────────────────────────────────
    bloom_capacity: int = 50_000

    # ── PageRank ───────────────────────────────────────
    pagerank_max_iterations: int = 20
    pagerank_epsilon: float = 1e-6

    # ── Event bus ──────────────────────────────────────
    event_history_max: int = 1000

    # ── Circuit breaker ────────────────────────────────
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_seconds: float = 60.0

    # ── Re-ranking ─────────────────────────────────────
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
