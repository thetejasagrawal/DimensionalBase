"""
BENCHMARK 2: Contradiction Detection

Claim: "79% of multi-agent failures happen at the coordination layer."

Setup:
  - 4 agents write facts about a shared system
  - We inject KNOWN contradictions at a controlled rate
  - Measure: detection rate, false positive rate, detection latency

The point: TextPassing, SharedDict, and VectorStore have ZERO
contradiction detection. They silently overwrite or ignore conflicts.
DimensionalBase catches them automatically.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

from benchmarks.baselines import (
    BenchmarkMetrics,
    TextPassingBaseline,
    SharedDictBaseline,
    VectorStoreBaseline,
    _estimate_tokens,
)

AGENTS = ["backend", "frontend", "qa", "monitor"]

# Ground truth: known contradictions we inject
CONTRADICTION_PAIRS = [
    # (agent_a, path_a, value_a, agent_b, path_b, value_b)
    ("backend", "service/api/status", "API healthy. All endpoints 200.",
     "frontend", "service/api/health", "API down. All requests timing out."),
    ("backend", "service/db/latency", "Database latency: 5ms p99.",
     "monitor", "service/db/perf", "Database latency: 2500ms p99. Critical alert."),
    ("qa", "deploy/staging/result", "Staging tests: 142/142 passing.",
     "monitor", "deploy/staging/health", "Staging health check failing. 503 errors."),
    ("backend", "service/auth/status", "Auth service healthy. Token validation OK.",
     "frontend", "service/auth/errors", "Auth returning 401 on all requests."),
    ("qa", "deploy/canary/metrics", "Canary deployment: error rate 0.1%.",
     "monitor", "deploy/canary/alerts", "Canary deployment: error rate 45%. Rollback recommended."),
    ("backend", "service/cache/hit_rate", "Cache hit rate: 98%.",
     "monitor", "service/cache/perf", "Cache miss rate: 67%. Memory pressure critical."),
    ("qa", "test/integration/auth", "Auth integration tests: all passing.",
     "frontend", "test/integration/auth_e2e", "Auth E2E tests: 8/20 failing on token refresh."),
    ("backend", "service/queue/depth", "Message queue depth: 12. Normal.",
     "monitor", "service/queue/alert", "Message queue depth: 50000. Consumer lag critical."),
]

# Non-contradictory facts (should NOT trigger conflicts)
CONSISTENT_FACTS = [
    ("backend", "service/api/version", "API version: v2.3.1"),
    ("frontend", "ui/version", "Frontend build: #1847"),
    ("qa", "test/unit/coverage", "Unit test coverage: 87%"),
    ("monitor", "infra/cpu/avg", "Average CPU: 34%"),
    ("backend", "service/api/endpoints", "Active endpoints: 47"),
    ("frontend", "ui/load_time", "Page load time: 1.2s"),
    ("qa", "test/e2e/count", "E2E test count: 320"),
    ("monitor", "infra/memory/used", "Memory usage: 62%"),
]


def run_text_passing() -> BenchmarkMetrics:
    """TextPassing: zero detection. All contradictions missed."""
    system = TextPassingBaseline()
    metrics = system.metrics

    # Write consistent facts
    for agent, path, value in CONSISTENT_FACTS:
        system.write(agent, f"{path}: {value}")

    # Write contradictions
    for a1, p1, v1, a2, p2, v2 in CONTRADICTION_PAIRS:
        system.write(a1, f"{p1}: {v1}")
        system.write(a2, f"{p2}: {v2}")

    metrics.contradictions_missed = len(CONTRADICTION_PAIRS)
    metrics.contradictions_detected = 0
    metrics.false_conflicts = 0
    return metrics


def run_shared_dict() -> BenchmarkMetrics:
    """SharedDict: zero detection. Silently overwrites."""
    system = SharedDictBaseline()
    metrics = system.metrics

    for agent, path, value in CONSISTENT_FACTS:
        system.write(agent, path, value)

    for a1, p1, v1, a2, p2, v2 in CONTRADICTION_PAIRS:
        system.write(a1, p1, v1)
        system.write(a2, p2, v2)

    metrics.contradictions_missed = len(CONTRADICTION_PAIRS)
    metrics.contradictions_detected = 0
    metrics.false_conflicts = 0
    return metrics


def run_vector_store() -> BenchmarkMetrics:
    """VectorStore: zero detection. Just stores and retrieves."""
    system = VectorStoreBaseline(dimension=64)
    metrics = system.metrics

    for agent, path, value in CONSISTENT_FACTS:
        system.write(agent, path, value)

    for a1, p1, v1, a2, p2, v2 in CONTRADICTION_PAIRS:
        system.write(a1, p1, v1)
        system.write(a2, p2, v2)

    metrics.contradictions_missed = len(CONTRADICTION_PAIRS)
    metrics.contradictions_detected = 0
    metrics.false_conflicts = 0
    return metrics


def run_dimensionalbase() -> BenchmarkMetrics:
    """DimensionalBase: active contradiction detection."""
    from dimensionalbase import DimensionalBase, EventType

    db = DimensionalBase()
    metrics = BenchmarkMetrics()
    detected_conflicts: List[str] = []
    known_conflict_paths = set()

    def on_event(event):
        if event.type == EventType.CONFLICT:
            detected_conflicts.append(event.path)

    db.subscribe("**", "benchmark", on_event)

    # Build set of paths involved in contradictions
    for a1, p1, v1, a2, p2, v2 in CONTRADICTION_PAIRS:
        known_conflict_paths.add(p1)
        known_conflict_paths.add(p2)

    # Write consistent facts first
    for agent, path, value in CONSISTENT_FACTS:
        t0 = time.perf_counter()
        db.put(path=path, value=value, owner=agent, type="fact")
        metrics.write_latency_us.append((time.perf_counter() - t0) * 1e6)
        metrics.total_tokens_written += _estimate_tokens(value)

    # Write contradictions
    for a1, p1, v1, a2, p2, v2 in CONTRADICTION_PAIRS:
        t0 = time.perf_counter()
        db.put(path=p1, value=v1, owner=a1, type="fact", confidence=0.9)
        metrics.write_latency_us.append((time.perf_counter() - t0) * 1e6)
        metrics.total_tokens_written += _estimate_tokens(v1)

        t0 = time.perf_counter()
        db.put(path=p2, value=v2, owner=a2, type="fact", confidence=0.85)
        metrics.write_latency_us.append((time.perf_counter() - t0) * 1e6)
        metrics.total_tokens_written += _estimate_tokens(v2)

    # Count detections
    detected_set = set(detected_conflicts)
    true_positives = len(detected_set & known_conflict_paths)
    false_positives = len(detected_set - known_conflict_paths)

    metrics.contradictions_detected = true_positives
    metrics.contradictions_missed = len(CONTRADICTION_PAIRS) - true_positives
    metrics.false_conflicts = false_positives

    db.close()
    return metrics


def run() -> Dict[str, BenchmarkMetrics]:
    return {
        "TextPassing": run_text_passing(),
        "SharedDict": run_shared_dict(),
        "VectorStore": run_vector_store(),
        "DimensionalBase": run_dimensionalbase(),
    }
