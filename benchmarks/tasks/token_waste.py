"""
BENCHMARK 1: Token Waste

Claim: "72% of all tokens in multi-agent systems are wasted on redundant context."

Setup:
  - 5 agents writing knowledge over 20 rounds
  - Each round, each agent writes 1-3 entries and reads context
  - We track: total tokens read vs. tokens that were actually NEW/RELEVANT

This measures the fundamental inefficiency of text-based coordination.
TextPassing dumps everything every time. SharedDict returns all matching keys.
VectorStore returns top-k without budget awareness. DimensionalBase returns
scored, budget-fitted, deduplicated context.
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

# Ground truth: what information actually exists at each round
AGENTS = ["planner", "backend", "frontend", "qa", "devops"]
DOMAINS = ["auth", "api", "deploy", "monitoring", "database"]


def _generate_facts(round_num: int, agent: str) -> List[Dict[str, str]]:
    """Generate realistic knowledge entries for a round."""
    np.random.seed(hash(f"{round_num}_{agent}") % (2**31))
    domain = DOMAINS[hash(agent) % len(DOMAINS)]
    n_facts = np.random.randint(1, 4)
    facts = []
    for i in range(n_facts):
        path = f"task/{domain}/round{round_num}/item{i}"
        # Mix of new info and updates to old paths
        if np.random.random() < 0.3 and round_num > 0:
            path = f"task/{domain}/round{round_num - 1}/item0"  # Update
        value = f"[R{round_num}] {agent}: {domain} status update #{i}. " \
                f"Metric={np.random.randint(0, 100)}, healthy={np.random.random() > 0.3}"
        facts.append({"path": path, "value": value, "domain": domain})
    return facts


def run_text_passing(rounds: int = 20) -> BenchmarkMetrics:
    """Run the token waste benchmark with TextPassing baseline."""
    system = TextPassingBaseline()
    relevant_per_read: List[float] = []

    for r in range(rounds):
        for agent in AGENTS:
            facts = _generate_facts(r, agent)
            for f in facts:
                system.write(agent, f"{f['path']}: {f['value']}")

            # Agent reads context
            context = system.read(agent)
            total_tokens = _estimate_tokens(context)

            # How much of this context is relevant to this agent?
            domain = DOMAINS[hash(agent) % len(DOMAINS)]
            relevant_lines = [
                line for line in context.split("\n")
                if domain in line or f"round{r}" in line
            ]
            relevant_tokens = sum(_estimate_tokens(l) for l in relevant_lines)

            system.metrics.useful_tokens += relevant_tokens
            system.metrics.redundant_tokens += (total_tokens - relevant_tokens)

    return system.metrics


def run_shared_dict(rounds: int = 20) -> BenchmarkMetrics:
    """Run the token waste benchmark with SharedDict baseline."""
    system = SharedDictBaseline()

    for r in range(rounds):
        for agent in AGENTS:
            facts = _generate_facts(r, agent)
            for f in facts:
                system.write(agent, f["path"], f["value"])

            # Agent reads all task entries (no budget limit, no scoring)
            context = system.read("task/")
            total_tokens = _estimate_tokens(context)

            domain = DOMAINS[hash(agent) % len(DOMAINS)]
            relevant_lines = [
                line for line in context.split("\n")
                if domain in line or f"round{r}" in line
            ]
            relevant_tokens = sum(_estimate_tokens(l) for l in relevant_lines)

            system.metrics.useful_tokens += relevant_tokens
            system.metrics.redundant_tokens += (total_tokens - relevant_tokens)

    return system.metrics


def run_vector_store(rounds: int = 20) -> BenchmarkMetrics:
    """Run the token waste benchmark with VectorStore baseline."""
    system = VectorStoreBaseline(dimension=64)

    for r in range(rounds):
        for agent in AGENTS:
            facts = _generate_facts(r, agent)
            for f in facts:
                system.write(agent, f["path"], f["value"])

            # Agent reads top-10 by similarity (no budget awareness)
            domain = DOMAINS[hash(agent) % len(DOMAINS)]
            context = system.read(f"task {domain} round{r}", k=10)
            total_tokens = _estimate_tokens(context) if context else 0

            relevant_lines = [
                line for line in context.split("\n")
                if domain in line or f"round{r}" in line
            ] if context else []
            relevant_tokens = sum(_estimate_tokens(l) for l in relevant_lines)

            system.metrics.useful_tokens += relevant_tokens
            system.metrics.redundant_tokens += (total_tokens - relevant_tokens)

    return system.metrics


def run_dimensionalbase(rounds: int = 20) -> BenchmarkMetrics:
    """Run the token waste benchmark with DimensionalBase."""
    from dimensionalbase import DimensionalBase

    db = DimensionalBase()
    metrics = BenchmarkMetrics()

    for r in range(rounds):
        for agent in AGENTS:
            facts = _generate_facts(r, agent)
            for f in facts:
                t0 = time.perf_counter()
                db.put(
                    path=f["path"],
                    value=f["value"],
                    owner=agent,
                    type="observation",
                    confidence=0.7 + np.random.random() * 0.3,
                )
                metrics.write_latency_us.append((time.perf_counter() - t0) * 1e6)
                metrics.total_tokens_written += _estimate_tokens(f["value"])

            # Budget-aware read with query
            domain = DOMAINS[hash(agent) % len(DOMAINS)]
            t0 = time.perf_counter()
            result = db.get(
                scope="task/**",
                budget=200,  # Fixed budget — DB manages what fits
                query=f"What's happening with {domain}?",
                reader=agent,
            )
            metrics.read_latency_us.append((time.perf_counter() - t0) * 1e6)

            total_tokens = result.tokens_used
            metrics.total_tokens_read += total_tokens

            # Measure relevance
            relevant = sum(
                1 for e in result.entries
                if domain in e.path or f"round{r}" in e.value
            )
            total = len(result.entries) or 1
            relevant_tokens = int(total_tokens * (relevant / total))

            metrics.useful_tokens += relevant_tokens
            metrics.redundant_tokens += (total_tokens - relevant_tokens)

    db.close()
    return metrics


def run() -> Dict[str, BenchmarkMetrics]:
    """Run all baselines and return metrics."""
    return {
        "TextPassing": run_text_passing(),
        "SharedDict": run_shared_dict(),
        "VectorStore": run_vector_store(),
        "DimensionalBase": run_dimensionalbase(),
    }
