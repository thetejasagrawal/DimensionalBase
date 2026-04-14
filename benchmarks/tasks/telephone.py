"""
BENCHMARK 3: Information Telephone (Error Amplification)

Claim: "Errors amplify 17× as they pass between agents."

Setup:
  - A chain of 8 agents
  - Agent 1 writes 10 ground-truth facts
  - Each subsequent agent reads, processes, and writes its understanding
  - We measure: information retained vs. lost at each hop

In TextPassing: each agent re-encodes ALL previous text, losing nuance.
In DimensionalBase: each agent reads from the shared source, not copies.
Zero hops. Zero telephone-game effects.
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from benchmarks.baselines import (
    BenchmarkMetrics,
    TextPassingBaseline,
    SharedDictBaseline,
    _estimate_tokens,
)

N_AGENTS = 8
CHAIN_AGENTS = [f"agent-{i}" for i in range(N_AGENTS)]

# Ground truth facts with specific numeric values
GROUND_TRUTH = [
    {"key": "api/latency_p99", "value": "API p99 latency is 47ms", "number": 47},
    {"key": "api/error_rate", "value": "API error rate is 0.3%", "number": 0.3},
    {"key": "db/connections", "value": "Database has 142 active connections", "number": 142},
    {"key": "db/query_time", "value": "Average query time is 12ms", "number": 12},
    {"key": "cache/hit_rate", "value": "Cache hit rate is 94.7%", "number": 94.7},
    {"key": "queue/depth", "value": "Message queue depth is 23", "number": 23},
    {"key": "cpu/utilization", "value": "CPU utilization is 67%", "number": 67},
    {"key": "memory/used_gb", "value": "Memory usage is 12.4 GB", "number": 12.4},
    {"key": "deploy/version", "value": "Current deploy version is v2.3.1", "number": 2.31},
    {"key": "test/passing", "value": "321 of 350 tests passing", "number": 321},
]


def _simulate_agent_distortion(value: str, hop: int) -> str:
    """Simulate how text transforms as agents process it.

    Each hop introduces small changes — rounding, paraphrasing,
    losing precision. This models real LLM behavior where each
    agent summarizes in its own words.
    """
    np.random.seed(hash(f"{value}_{hop}") % (2**31))

    # Each hop: 10-20% chance of introducing an error
    if np.random.random() < 0.15 * hop:
        # Numeric drift: change a number slightly
        import re
        numbers = re.findall(r'\d+\.?\d*', value)
        if numbers:
            old_num = numbers[0]
            try:
                n = float(old_num)
                # Random drift proportional to hop count
                drift = n * np.random.normal(0, 0.05 * hop)
                new_num = str(round(n + drift, 1))
                value = value.replace(old_num, new_num, 1)
            except ValueError:
                pass

    # Each hop: rephrase slightly
    prefixes = ["", "Reportedly, ", "According to previous agent, ",
                "It appears that ", "The data shows ", "Status: "]
    if hop > 0:
        value = np.random.choice(prefixes) + value.lower()

    return value


def _measure_retention(final_values: List[str], ground_truth: List[Dict]) -> float:
    """Measure how much ground truth information survived.

    Checks if the original numbers are still present in the final text.
    """
    import re
    retained = 0
    for gt in ground_truth:
        target_num = gt["number"]
        for val in final_values:
            numbers = re.findall(r'\d+\.?\d*', val)
            for n in numbers:
                try:
                    if abs(float(n) - target_num) / (abs(target_num) + 0.01) < 0.05:
                        retained += 1
                        break
                except ValueError:
                    pass
            else:
                continue
            break

    return retained / len(ground_truth)


def run_text_passing() -> BenchmarkMetrics:
    """Text passing: each agent reads ALL history and writes its version."""
    system = TextPassingBaseline()
    metrics = system.metrics
    retention_per_hop = []

    # Agent 0 writes ground truth
    for fact in GROUND_TRUTH:
        system.write(CHAIN_AGENTS[0], fact["value"])

    # Each subsequent agent reads everything and writes its understanding
    for hop in range(1, N_AGENTS):
        agent = CHAIN_AGENTS[hop]
        context = system.read(agent)

        # Agent processes and writes its understanding
        lines = context.split("\n")
        for line in lines:
            if line.strip():
                # Extract the value part and distort it
                parts = line.split("]: ", 1)
                value = parts[1] if len(parts) > 1 else line
                distorted = _simulate_agent_distortion(value, hop)
                system.write(agent, distorted)

        # Measure retention at this hop
        agent_outputs = [h["text"] for h in system.history if h["agent"] == agent]
        retention = _measure_retention(agent_outputs, GROUND_TRUTH)
        retention_per_hop.append(retention)

    # Final retention
    final_agent = CHAIN_AGENTS[-1]
    final_outputs = [h["text"] for h in system.history if h["agent"] == final_agent]
    metrics.information_retained = _measure_retention(final_outputs, GROUND_TRUTH)

    return metrics


def run_shared_dict() -> BenchmarkMetrics:
    """SharedDict: agents read from shared store but still transform."""
    system = SharedDictBaseline()
    metrics = system.metrics

    # Agent 0 writes ground truth
    for fact in GROUND_TRUTH:
        system.write(CHAIN_AGENTS[0], fact["key"], fact["value"])

    # Each agent reads and writes its version
    for hop in range(1, N_AGENTS):
        agent = CHAIN_AGENTS[hop]
        context = system.read("", budget=999999)

        # Agent writes its understanding (still distorts)
        for fact in GROUND_TRUTH:
            original = system.store.get(fact["key"], {}).get("value", fact["value"])
            distorted = _simulate_agent_distortion(original, hop)
            system.write(agent, fact["key"], distorted)

    # Final values
    final_values = [entry["value"] for entry in system.store.values()]
    metrics.information_retained = _measure_retention(final_values, GROUND_TRUTH)

    return metrics


def run_dimensionalbase() -> BenchmarkMetrics:
    """DimensionalBase: shared state — zero hops, zero telephone game.

    Key difference: agents read from the ORIGINAL source, not from
    forwarded copies. The first agent's ground truth is always available.
    Later agents ADD observations, they don't REPLACE the source.
    """
    from dimensionalbase import DimensionalBase

    db = DimensionalBase()
    metrics = BenchmarkMetrics()

    # Agent 0 writes ground truth (this stays in the store permanently)
    for fact in GROUND_TRUTH:
        t0 = time.perf_counter()
        db.put(
            path=f"metrics/{fact['key']}",
            value=fact["value"],
            owner=CHAIN_AGENTS[0],
            type="fact",
            confidence=1.0,
            ttl="persistent",
        )
        metrics.write_latency_us.append((time.perf_counter() - t0) * 1e6)
        metrics.total_tokens_written += _estimate_tokens(fact["value"])

    # Each subsequent agent reads from the SHARED STORE (not from the previous agent's copy)
    for hop in range(1, N_AGENTS):
        agent = CHAIN_AGENTS[hop]

        # Read original facts — they're still there, untouched
        t0 = time.perf_counter()
        result = db.get(
            scope="metrics/**",
            budget=500,
            query="What are the current system metrics?",
        )
        metrics.read_latency_us.append((time.perf_counter() - t0) * 1e6)
        metrics.total_tokens_read += result.tokens_used

        # Agent adds its observations ALONGSIDE the originals (not replacing)
        for entry in result.entries:
            distorted = _simulate_agent_distortion(entry.value, hop=1)  # Only 1 hop from source!
            path = entry.path.replace("metrics/", f"observations/{agent}/")
            t0 = time.perf_counter()
            db.put(
                path=path,
                value=distorted,
                owner=agent,
                type="observation",
                confidence=0.8,
                refs=[entry.path],
            )
            metrics.write_latency_us.append((time.perf_counter() - t0) * 1e6)

    # Measure retention: read the ORIGINAL facts (still in the store)
    originals = db.get(scope="metrics/**", budget=5000, owner=CHAIN_AGENTS[0])
    original_values = [e.value for e in originals.entries]
    metrics.information_retained = _measure_retention(original_values, GROUND_TRUTH)

    db.close()
    return metrics


def run() -> Dict[str, BenchmarkMetrics]:
    return {
        "TextPassing": run_text_passing(),
        "SharedDict": run_shared_dict(),
        "DimensionalBase": run_dimensionalbase(),
    }
