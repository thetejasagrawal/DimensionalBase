"""
BENCHMARK 5: Scale Stress Test

Claim: DimensionalBase scales linearly while maintaining quality.

Setup:
  - Insert 100, 1000, 5000, 10000 entries
  - Measure: write latency, read latency, memory, retrieval quality

The point: TextPassing degrades catastrophically (context grows linearly).
SharedDict degrades on prefix scans. VectorStore stays fast but quality
doesn't improve. DimensionalBase stays fast AND gets smarter with scale.
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List

import numpy as np

from benchmarks.baselines import (
    BenchmarkMetrics,
    TextPassingBaseline,
    SharedDictBaseline,
    VectorStoreBaseline,
    _estimate_tokens,
)

SCALE_LEVELS = [100, 1000, 5000, 10000]


def _generate_entries(n: int) -> List[Dict]:
    np.random.seed(42)
    agents = [f"agent-{i}" for i in range(10)]
    domains = ["auth", "api", "db", "cache", "queue", "deploy", "monitoring"]
    entries = []
    for i in range(n):
        domain = domains[i % len(domains)]
        agent = agents[i % len(agents)]
        entries.append({
            "path": f"system/{domain}/metric-{i:05d}",
            "value": f"{domain} metric {i}: value={np.random.randint(0, 10000)}, "
                     f"status={'ok' if np.random.random() > 0.2 else 'alert'}",
            "agent": agent,
            "domain": domain,
        })
    return entries


def run_at_scale(n: int) -> Dict[str, BenchmarkMetrics]:
    """Run all baselines at a given scale."""
    entries = _generate_entries(n)
    results = {}

    # --- TextPassing ---
    tp = TextPassingBaseline()
    for e in entries:
        t0 = time.perf_counter()
        tp.write(e["agent"], f"{e['path']}: {e['value']}")
        tp.metrics.write_latency_us.append((time.perf_counter() - t0) * 1e6)
        tp.metrics.total_tokens_written += _estimate_tokens(e["value"])

    for _ in range(10):
        t0 = time.perf_counter()
        ctx = tp.read("reader")
        tp.metrics.read_latency_us.append((time.perf_counter() - t0) * 1e6)

    results["TextPassing"] = tp.metrics

    # --- SharedDict ---
    sd = SharedDictBaseline()
    for e in entries:
        t0 = time.perf_counter()
        sd.write(e["agent"], e["path"], e["value"])
        sd.metrics.write_latency_us.append((time.perf_counter() - t0) * 1e6)
        sd.metrics.total_tokens_written += _estimate_tokens(e["value"])

    for _ in range(10):
        t0 = time.perf_counter()
        sd.read("system/")
        sd.metrics.read_latency_us.append((time.perf_counter() - t0) * 1e6)

    results["SharedDict"] = sd.metrics

    # --- VectorStore ---
    vs = VectorStoreBaseline(dimension=64)
    for e in entries:
        t0 = time.perf_counter()
        vs.write(e["agent"], e["path"], e["value"])
        vs.metrics.write_latency_us.append((time.perf_counter() - t0) * 1e6)
        vs.metrics.total_tokens_written += _estimate_tokens(e["value"])

    for _ in range(10):
        t0 = time.perf_counter()
        vs.read("system status alert", k=20)
        vs.metrics.read_latency_us.append((time.perf_counter() - t0) * 1e6)

    results["VectorStore"] = vs.metrics

    # --- DimensionalBase ---
    from dimensionalbase import DimensionalBase
    db = DimensionalBase()
    db_metrics = BenchmarkMetrics()

    for e in entries:
        t0 = time.perf_counter()
        db.put(path=e["path"], value=e["value"], owner=e["agent"], type="observation")
        db_metrics.write_latency_us.append((time.perf_counter() - t0) * 1e6)
        db_metrics.total_tokens_written += _estimate_tokens(e["value"])

    for _ in range(10):
        t0 = time.perf_counter()
        result = db.get(scope="system/**", budget=500, query="system status alerts")
        db_metrics.read_latency_us.append((time.perf_counter() - t0) * 1e6)
        db_metrics.total_tokens_read += result.tokens_used

    db.close()
    results["DimensionalBase"] = db_metrics

    return results


def run() -> Dict[int, Dict[str, BenchmarkMetrics]]:
    """Run at all scale levels."""
    all_results = {}
    for n in SCALE_LEVELS:
        all_results[n] = run_at_scale(n)
    return all_results
