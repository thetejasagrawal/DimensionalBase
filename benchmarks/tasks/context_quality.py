"""
BENCHMARK 4: Context Retrieval Quality

Claim: DimensionalBase returns higher-quality context per token than
any alternative.

Setup:
  - 200 knowledge entries across 10 domains
  - 20 queries with KNOWN relevant entries (ground truth)
  - Measure: precision@k, recall@k, F1, tokens wasted

The point: TextPassing returns everything (high recall, zero precision).
SharedDict returns prefix matches (structured but no ranking).
VectorStore returns top-k by similarity (good precision, no budget control).
DimensionalBase: scored, budget-fitted, deduplicated, trust-weighted.
"""

from __future__ import annotations

import time
from typing import Dict, List, Set, Tuple

import numpy as np

from benchmarks.baselines import (
    BenchmarkMetrics,
    TextPassingBaseline,
    SharedDictBaseline,
    VectorStoreBaseline,
    _estimate_tokens,
)

DOMAINS = ["auth", "api", "database", "cache", "queue",
           "deploy", "monitoring", "security", "testing", "performance"]

# Generate 200 knowledge entries
def _generate_knowledge_base() -> List[Dict]:
    """Generate a realistic knowledge base with 200 entries."""
    np.random.seed(42)
    entries = []
    agents = ["backend", "frontend", "qa", "devops", "sre"]

    for domain in DOMAINS:
        for i in range(20):
            agent = agents[i % len(agents)]
            path = f"system/{domain}/metric-{i:03d}"
            value = (
                f"{domain.title()} metric {i}: "
                f"value={np.random.randint(0, 1000)}, "
                f"status={'healthy' if np.random.random() > 0.2 else 'degraded'}, "
                f"last_check={np.random.randint(1, 60)}s ago"
            )
            entries.append({
                "path": path,
                "value": value,
                "agent": agent,
                "domain": domain,
                "confidence": round(0.5 + np.random.random() * 0.5, 2),
            })
    return entries


# Define queries with known relevant entries
def _generate_queries(entries: List[Dict]) -> List[Dict]:
    """Generate queries with ground truth relevant paths."""
    queries = []
    for domain in DOMAINS[:5]:  # 5 domains × 4 queries each = 20 queries
        domain_entries = [e for e in entries if e["domain"] == domain]
        relevant_paths = {e["path"] for e in domain_entries}

        queries.append({
            "query": f"What is the status of {domain}?",
            "scope": f"system/{domain}/**",
            "relevant_paths": relevant_paths,
            "domain": domain,
        })
        queries.append({
            "query": f"Are there any {domain} issues?",
            "scope": "system/**",
            "relevant_paths": relevant_paths,
            "domain": domain,
        })
        queries.append({
            "query": f"Show me {domain} degraded metrics",
            "scope": "system/**",
            "relevant_paths": {
                e["path"] for e in domain_entries
                if "degraded" in e["value"]
            } or relevant_paths,
            "domain": domain,
        })
        queries.append({
            "query": f"Latest {domain} health check results",
            "scope": f"system/{domain}/**",
            "relevant_paths": relevant_paths,
            "domain": domain,
        })

    return queries


def _precision_recall(retrieved_paths: Set[str], relevant_paths: Set[str]) -> Tuple[float, float]:
    """Compute precision and recall."""
    if not retrieved_paths:
        return 0.0, 0.0
    if not relevant_paths:
        return 1.0, 1.0

    true_positives = len(retrieved_paths & relevant_paths)
    precision = true_positives / len(retrieved_paths) if retrieved_paths else 0
    recall = true_positives / len(relevant_paths) if relevant_paths else 0
    return precision, recall


def run_text_passing() -> BenchmarkMetrics:
    system = TextPassingBaseline()
    metrics = system.metrics
    entries = _generate_knowledge_base()
    queries = _generate_queries(entries)

    for e in entries:
        system.write(e["agent"], f"{e['path']}: {e['value']}")

    precisions, recalls = [], []
    for q in queries:
        context = system.read("query-agent")
        retrieved = set()
        for line in context.split("\n"):
            for e in entries:
                if e["path"] in line:
                    retrieved.add(e["path"])

        p, r = _precision_recall(retrieved, q["relevant_paths"])
        precisions.append(p)
        recalls.append(r)

        tokens = _estimate_tokens(context)
        relevant_tokens = int(tokens * p) if p > 0 else 0
        metrics.useful_tokens += relevant_tokens
        metrics.redundant_tokens += tokens - relevant_tokens

    metrics.context_precision = np.mean(precisions)
    metrics.context_recall = np.mean(recalls)
    return metrics


def run_shared_dict() -> BenchmarkMetrics:
    system = SharedDictBaseline()
    metrics = system.metrics
    entries = _generate_knowledge_base()
    queries = _generate_queries(entries)

    for e in entries:
        system.write(e["agent"], e["path"], e["value"])

    precisions, recalls = [], []
    for q in queries:
        context = system.read(f"system/{q['domain']}")
        retrieved = set()
        for key in system.store:
            if key.startswith(f"system/{q['domain']}"):
                retrieved.add(key)

        p, r = _precision_recall(retrieved, q["relevant_paths"])
        precisions.append(p)
        recalls.append(r)

        tokens = _estimate_tokens(context)
        relevant_tokens = int(tokens * p) if p > 0 else 0
        metrics.useful_tokens += relevant_tokens
        metrics.redundant_tokens += tokens - relevant_tokens

    metrics.context_precision = np.mean(precisions)
    metrics.context_recall = np.mean(recalls)
    return metrics


def run_vector_store() -> BenchmarkMetrics:
    system = VectorStoreBaseline(dimension=64)
    metrics = system.metrics
    entries = _generate_knowledge_base()
    queries = _generate_queries(entries)

    for e in entries:
        system.write(e["agent"], e["path"], e["value"])

    precisions, recalls = [], []
    for q in queries:
        context = system.read(q["query"], k=20)
        retrieved = set()
        for line in context.split("\n"):
            for e in entries:
                if e["path"] in line:
                    retrieved.add(e["path"])

        p, r = _precision_recall(retrieved, q["relevant_paths"])
        precisions.append(p)
        recalls.append(r)

        tokens = _estimate_tokens(context)
        relevant_tokens = int(tokens * p) if p > 0 else 0
        metrics.useful_tokens += relevant_tokens
        metrics.redundant_tokens += tokens - relevant_tokens

    metrics.context_precision = np.mean(precisions)
    metrics.context_recall = np.mean(recalls)
    return metrics


def run_dimensionalbase() -> BenchmarkMetrics:
    from dimensionalbase import DimensionalBase

    db = DimensionalBase()
    metrics = BenchmarkMetrics()
    entries = _generate_knowledge_base()
    queries = _generate_queries(entries)

    for e in entries:
        t0 = time.perf_counter()
        db.put(
            path=e["path"], value=e["value"], owner=e["agent"],
            type="observation", confidence=e["confidence"],
        )
        metrics.write_latency_us.append((time.perf_counter() - t0) * 1e6)
        metrics.total_tokens_written += _estimate_tokens(e["value"])

    precisions, recalls = [], []
    for q in queries:
        t0 = time.perf_counter()
        result = db.get(
            scope=q["scope"],
            budget=300,
            query=q["query"],
        )
        metrics.read_latency_us.append((time.perf_counter() - t0) * 1e6)

        retrieved = {e.path for e in result.entries}
        p, r = _precision_recall(retrieved, q["relevant_paths"])
        precisions.append(p)
        recalls.append(r)

        metrics.total_tokens_read += result.tokens_used
        relevant_tokens = int(result.tokens_used * p) if p > 0 else 0
        metrics.useful_tokens += relevant_tokens
        metrics.redundant_tokens += result.tokens_used - relevant_tokens

    metrics.context_precision = np.mean(precisions)
    metrics.context_recall = np.mean(recalls)
    db.close()
    return metrics


def run() -> Dict[str, BenchmarkMetrics]:
    return {
        "TextPassing": run_text_passing(),
        "SharedDict": run_shared_dict(),
        "VectorStore": run_vector_store(),
        "DimensionalBase": run_dimensionalbase(),
    }
