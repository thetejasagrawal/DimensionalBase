"""
Baselines — faithful implementations of how multi-agent systems actually
communicate today. Not strawmen. These represent real production patterns.

Baseline 1: TextPassing
    How: Each agent gets the FULL text history of all previous agents.
    Reality: This is how LangChain chains, CrewAI sequential tasks, and
    most multi-agent systems actually work. Agent N receives a growing
    context dump of everything agents 1..N-1 said.

Baseline 2: SharedDict
    How: Agents read/write to a shared Python dict.
    Reality: This is the "improved" version people build when text passing
    gets too expensive. Structured keys, but no scoring, no conflict
    detection, no budget limits.

Baseline 3: VectorStore
    How: Agents write to a vector store with embedding similarity search.
    Reality: This is what you get with a basic RAG setup. Semantic
    retrieval, but no active reasoning, no trust, no provenance,
    no budget-aware packing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# METRICS — what we measure for every baseline
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkMetrics:
    """Standardized metrics for fair comparison."""
    # Token usage
    total_tokens_written: int = 0       # Tokens consumed by all writes
    total_tokens_read: int = 0          # Tokens returned by all reads
    redundant_tokens: int = 0           # Tokens that were duplicates/irrelevant
    useful_tokens: int = 0              # Tokens that were actually relevant

    # Quality
    contradictions_detected: int = 0    # Conflicts caught by the system
    contradictions_missed: int = 0      # Conflicts that slipped through
    false_conflicts: int = 0            # False positives
    information_retained: float = 0.0   # 0-1: how much ground truth survives
    context_precision: float = 0.0      # Retrieved relevant / retrieved total
    context_recall: float = 0.0         # Retrieved relevant / total relevant

    # Performance
    write_latency_us: List[float] = field(default_factory=list)
    read_latency_us: List[float] = field(default_factory=list)
    peak_memory_bytes: int = 0

    # Derived
    @property
    def token_waste_ratio(self) -> float:
        """What fraction of read tokens were wasted (redundant/irrelevant)."""
        if self.total_tokens_read == 0:
            return 0.0
        return self.redundant_tokens / self.total_tokens_read

    @property
    def token_efficiency(self) -> float:
        """Useful tokens / total tokens read. Higher is better."""
        if self.total_tokens_read == 0:
            return 0.0
        return self.useful_tokens / self.total_tokens_read

    @property
    def contradiction_detection_rate(self) -> float:
        total = self.contradictions_detected + self.contradictions_missed
        if total == 0:
            return 0.0
        return self.contradictions_detected / total

    @property
    def f1_score(self) -> float:
        if self.context_precision + self.context_recall == 0:
            return 0.0
        return 2 * (self.context_precision * self.context_recall) / \
            (self.context_precision + self.context_recall)

    @property
    def p50_write_us(self) -> float:
        if not self.write_latency_us:
            return 0
        return float(np.percentile(self.write_latency_us, 50))

    @property
    def p95_write_us(self) -> float:
        if not self.write_latency_us:
            return 0
        return float(np.percentile(self.write_latency_us, 95))

    @property
    def p50_read_us(self) -> float:
        if not self.read_latency_us:
            return 0
        return float(np.percentile(self.read_latency_us, 50))

    @property
    def p95_read_us(self) -> float:
        if not self.read_latency_us:
            return 0
        return float(np.percentile(self.read_latency_us, 95))


def _estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars per token)."""
    return max(1, len(text) // 4)


# ═══════════════════════════════════════════════════════════════════
# BASELINE 1: TEXT PASSING
# ═══════════════════════════════════════════════════════════════════

class TextPassingBaseline:
    """How most multi-agent systems actually work.

    Each agent receives the FULL concatenated text output of all
    previous agents. Context grows linearly. No filtering. No scoring.
    Every agent re-reads everything.

    This is: LangChain sequential chains, CrewAI sequential tasks,
    AutoGen group chat, most ChatGPT plugin chains.
    """

    def __init__(self):
        self.history: List[Dict[str, str]] = []  # [{agent, text}]
        self.metrics = BenchmarkMetrics()

    def write(self, agent: str, text: str) -> None:
        t0 = time.perf_counter()
        self.history.append({"agent": agent, "text": text})
        tokens = _estimate_tokens(text)
        self.metrics.total_tokens_written += tokens
        self.metrics.write_latency_us.append((time.perf_counter() - t0) * 1e6)

    def read(self, agent: str, budget: int = 999999) -> str:
        """Read ALL history — no filtering, no scoring, no budget awareness.

        This is the reality: agents get everything, whether relevant or not.
        """
        t0 = time.perf_counter()
        # Concatenate all history
        full_context = "\n".join(
            f"[{h['agent']}]: {h['text']}" for h in self.history
        )
        tokens = _estimate_tokens(full_context)
        self.metrics.total_tokens_read += tokens
        self.metrics.read_latency_us.append((time.perf_counter() - t0) * 1e6)
        return full_context

    def detect_contradictions(self) -> int:
        """Text passing has NO contradiction detection. Always 0."""
        return 0


# ═══════════════════════════════════════════════════════════════════
# BASELINE 2: SHARED DICT
# ═══════════════════════════════════════════════════════════════════

class SharedDictBaseline:
    """The 'improved' version: a shared key-value store.

    Better than text passing (structured keys, overwrite instead of append).
    But: no scoring, no conflict detection, no budget management,
    no semantic search, just exact-key lookups and prefix scans.

    This is: Redis/dict-based shared state, simple databases,
    what people build when text passing gets too expensive.
    """

    def __init__(self):
        self.store: Dict[str, Dict[str, Any]] = {}
        self.metrics = BenchmarkMetrics()

    def write(self, agent: str, key: str, value: str, **kwargs) -> None:
        t0 = time.perf_counter()
        tokens = _estimate_tokens(value)
        self.store[key] = {
            "agent": agent,
            "value": value,
            "updated_at": time.time(),
        }
        self.metrics.total_tokens_written += tokens
        self.metrics.write_latency_us.append((time.perf_counter() - t0) * 1e6)

    def read(self, prefix: str, budget: int = 999999) -> str:
        """Read all entries matching a prefix. No scoring, no budget control."""
        t0 = time.perf_counter()
        results = []
        for key, entry in self.store.items():
            if key.startswith(prefix) or prefix == "**":
                results.append(f"[{key}] ({entry['agent']}): {entry['value']}")

        context = "\n".join(results)
        tokens = _estimate_tokens(context)
        self.metrics.total_tokens_read += tokens
        self.metrics.read_latency_us.append((time.perf_counter() - t0) * 1e6)
        return context

    def detect_contradictions(self) -> int:
        """Shared dict has NO contradiction detection."""
        return 0


# ═══════════════════════════════════════════════════════════════════
# BASELINE 3: VECTOR STORE
# ═══════════════════════════════════════════════════════════════════

class VectorStoreBaseline:
    """RAG-style vector store with embedding similarity search.

    Has semantic search (good), but:
    - No active reasoning (no contradiction detection)
    - No budget-aware packing
    - No trust model
    - No provenance
    - No compositional operations
    - Returns top-k by similarity only

    This is: Pinecone, Weaviate, Chroma, basic RAG setups.
    """

    def __init__(self, dimension: int = 64):
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.vectors: Dict[str, np.ndarray] = {}
        self.dimension = dimension
        self.metrics = BenchmarkMetrics()

    def write(self, agent: str, key: str, value: str,
              embedding: Optional[np.ndarray] = None, **kwargs) -> None:
        t0 = time.perf_counter()
        if embedding is None:
            embedding = self._fake_embed(value)

        self.entries[key] = {
            "agent": agent,
            "value": value,
            "updated_at": time.time(),
        }
        self.vectors[key] = embedding / (np.linalg.norm(embedding) + 1e-12)

        tokens = _estimate_tokens(value)
        self.metrics.total_tokens_written += tokens
        self.metrics.write_latency_us.append((time.perf_counter() - t0) * 1e6)

    def read(self, query: str, k: int = 10, budget: int = 999999,
             query_embedding: Optional[np.ndarray] = None) -> str:
        """Top-k similarity search. No scoring beyond similarity. No budget."""
        t0 = time.perf_counter()

        if query_embedding is None:
            query_embedding = self._fake_embed(query)
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)

        if not self.vectors:
            self.metrics.read_latency_us.append((time.perf_counter() - t0) * 1e6)
            return ""

        # Brute force similarity search
        scores = {}
        for key, vec in self.vectors.items():
            scores[key] = float(np.dot(query_norm, vec))

        top_keys = sorted(scores, key=scores.get, reverse=True)[:k]
        results = []
        for key in top_keys:
            entry = self.entries[key]
            results.append(f"[{key}] ({entry['agent']}): {entry['value']}")

        context = "\n".join(results)
        tokens = _estimate_tokens(context)
        self.metrics.total_tokens_read += tokens
        self.metrics.read_latency_us.append((time.perf_counter() - t0) * 1e6)
        return context

    def detect_contradictions(self) -> int:
        """Vector stores have NO contradiction detection."""
        return 0

    def _fake_embed(self, text: str) -> np.ndarray:
        """Deterministic pseudo-embedding for benchmarking."""
        np.random.seed(hash(text) % (2**31))
        return np.random.randn(self.dimension).astype(np.float32)
