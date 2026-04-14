"""
Semantic Compression — minimize tokens while maximizing information.

The context window is the bottleneck. Every token counts. This module
provides three layers of compression:

  1. Delta encoding:      Only transmit what changed since last read
  2. Deduplication:       Detect and merge near-duplicate entries
  3. Information density:  Score each entry by bits of NEW info per token
                           and prioritize high-density entries

The goal: fit the most useful information into the smallest token budget.
This is the difference between "here's 2000 tokens of context" and
"here's 800 tokens with the same effective information."
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("dimensionalbase.context.compression")


@dataclass
class DeltaState:
    """Tracks what an agent has already seen (for delta encoding)."""
    agent_id: str
    seen_paths: Dict[str, int] = field(default_factory=dict)   # path -> version
    seen_hashes: Set[str] = field(default_factory=set)          # value hashes
    last_read_time: float = field(default_factory=time.time)
    total_tokens_served: int = 0


@dataclass
class CompressionResult:
    """Result of compressing a set of entries."""
    entries: List        # The filtered/compressed entries
    original_count: int
    compressed_count: int
    duplicates_removed: int
    deltas_applied: int
    estimated_token_savings: int
    compression_ratio: float  # 0-1, lower = more compression


class SemanticCompressor:
    """Multi-strategy compression engine for context windows.

    Combines delta encoding, deduplication, and information density
    scoring to minimize token usage while maximizing information.
    """

    def __init__(
        self,
        dedup_threshold: float = 0.92,
        min_information_density: float = 0.1,
    ):
        self._dedup_threshold = dedup_threshold
        self._min_density = min_information_density
        self._delta_states: Dict[str, DeltaState] = {}
        self._lock = threading.Lock()

        # Cache for value hashes
        self._hash_cache: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Delta Encoding — only send what's new
    # ------------------------------------------------------------------

    def register_reader(self, agent_id: str) -> DeltaState:
        """Register an agent as a reader for delta encoding."""
        with self._lock:
            if agent_id not in self._delta_states:
                self._delta_states[agent_id] = DeltaState(agent_id=agent_id)
            return self._delta_states[agent_id]

    def compute_delta(
        self,
        entries: List,
        reader_agent: str,
    ) -> Tuple[List, List, List]:
        """Partition entries into new, updated, and unchanged.

        Returns (new_entries, updated_entries, unchanged_entries).
        The caller can then prioritize new > updated > unchanged.
        """
        with self._lock:
            state = self._delta_states.get(reader_agent)

        if state is None:
            return entries, [], []

        new_entries = []
        updated_entries = []
        unchanged_entries = []

        for entry in entries:
            path = entry.path
            version = entry.version
            value_hash = self._hash_value(entry.value)

            if path not in state.seen_paths:
                new_entries.append(entry)
            elif state.seen_paths[path] < version:
                updated_entries.append(entry)
            elif value_hash not in state.seen_hashes:
                updated_entries.append(entry)
            else:
                unchanged_entries.append(entry)

        return new_entries, updated_entries, unchanged_entries

    def mark_as_seen(self, reader_agent: str, entries: List) -> None:
        """Mark entries as seen by a reader (for next delta computation)."""
        with self._lock:
            state = self._delta_states.get(reader_agent)
            if state is None:
                state = DeltaState(agent_id=reader_agent)
                self._delta_states[reader_agent] = state

            for entry in entries:
                state.seen_paths[entry.path] = entry.version
                state.seen_hashes.add(self._hash_value(entry.value))
                state.total_tokens_served += entry.token_estimate

            state.last_read_time = time.time()

    # ------------------------------------------------------------------
    # Deduplication — merge near-identical entries
    # ------------------------------------------------------------------

    def deduplicate(
        self,
        entries: List,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[List, int]:
        """Remove near-duplicate entries.

        Uses embedding similarity if available, falls back to
        text hash comparison.

        Returns (deduplicated_entries, count_removed).
        """
        if len(entries) <= 1:
            return entries, 0

        if embeddings and len(embeddings) >= 2:
            return self._deduplicate_by_embedding(entries, embeddings)
        return self._deduplicate_by_text(entries)

    def _deduplicate_by_embedding(
        self,
        entries: List,
        embeddings: Dict[str, np.ndarray],
    ) -> Tuple[List, int]:
        """Deduplicate using embedding similarity."""
        keep = []
        removed = 0
        seen_embeddings: List[np.ndarray] = []

        for entry in entries:
            emb = embeddings.get(entry.path)
            if emb is None:
                keep.append(entry)
                continue

            is_dup = False
            for seen in seen_embeddings:
                sim = float(np.dot(emb, seen) / (
                    np.linalg.norm(emb) * np.linalg.norm(seen) + 1e-12
                ))
                if sim > self._dedup_threshold:
                    is_dup = True
                    break

            if is_dup:
                removed += 1
            else:
                keep.append(entry)
                seen_embeddings.append(emb)

        return keep, removed

    def _deduplicate_by_text(self, entries: List) -> Tuple[List, int]:
        """Deduplicate using text hash comparison."""
        seen_hashes: Set[str] = set()
        keep = []
        removed = 0

        for entry in entries:
            h = self._hash_value(entry.value)
            if h in seen_hashes:
                removed += 1
            else:
                keep.append(entry)
                seen_hashes.add(h)

        return keep, removed

    # ------------------------------------------------------------------
    # Information Density — bits of new info per token
    # ------------------------------------------------------------------

    def score_information_density(
        self,
        entries: List,
        existing_context: Optional[List] = None,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Tuple[float, object]]:
        """Score each entry by information density (new info / tokens).

        High density = this entry adds a lot of new information per token.
        Low density = this entry is mostly redundant with existing context.

        Returns list of (density_score, entry) tuples, sorted by density DESC.
        """
        if not entries:
            return []

        scored = []
        context_hashes = set()
        context_embeddings: List[np.ndarray] = []

        if existing_context:
            for e in existing_context:
                context_hashes.add(self._hash_value(e.value))
                if embeddings and e.path in embeddings:
                    context_embeddings.append(embeddings[e.path])

        for entry in entries:
            tokens = max(1, entry.token_estimate)

            # Base novelty: is the text new?
            text_novelty = 0.0 if self._hash_value(entry.value) in context_hashes else 1.0

            # Semantic novelty: how different from existing context?
            semantic_novelty = 1.0
            if embeddings and entry.path in embeddings and context_embeddings:
                emb = embeddings[entry.path]
                max_sim = max(
                    float(np.dot(emb, ce) / (np.linalg.norm(emb) * np.linalg.norm(ce) + 1e-12))
                    for ce in context_embeddings
                )
                semantic_novelty = max(0.0, 1.0 - max_sim)

            # Confidence factor: higher confidence = more informative
            confidence_factor = entry.confidence

            # Information density: new info per token
            raw_density = (0.5 * text_novelty + 0.3 * semantic_novelty + 0.2 * confidence_factor)
            density = raw_density / tokens * 100  # Normalize to readable range

            scored.append((density, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Combined compression
    # ------------------------------------------------------------------

    def compress(
        self,
        entries: List,
        reader_agent: Optional[str] = None,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        budget: int = 2000,
    ) -> CompressionResult:
        """Apply all compression strategies and return optimized entries.

        Pipeline:
          1. Delta encoding (remove already-seen entries)
          2. Deduplication (merge near-duplicates)
          3. Information density ranking (prioritize high-info entries)
          4. Budget packing (fit within token budget)
        """
        original_count = len(entries)

        # 1. Delta encoding
        deltas_applied = 0
        if reader_agent:
            new, updated, unchanged = self.compute_delta(entries, reader_agent)
            # Prioritize: new first, then updated, then unchanged
            entries = new + updated + unchanged
            deltas_applied = len(unchanged)

        # 2. Deduplication
        entries, dups_removed = self.deduplicate(entries, embeddings)

        # 3. Information density ranking
        if embeddings:
            scored = self.score_information_density(entries, embeddings=embeddings)
            entries = [entry for _, entry in scored]

        # 4. Budget packing
        packed = []
        tokens_used = 0
        for entry in entries:
            est = entry.token_estimate
            if tokens_used + est <= budget:
                packed.append(entry)
                tokens_used += est
            else:
                # Try compact representation
                compact_est = entry.compact_token_estimate
                if tokens_used + compact_est <= budget:
                    packed.append(entry)
                    tokens_used += compact_est

        # Mark as seen for next delta
        if reader_agent:
            self.mark_as_seen(reader_agent, packed)

        original_tokens = sum(e.token_estimate for e in entries[:original_count]) if entries else 0
        ratio = tokens_used / max(1, original_tokens)

        return CompressionResult(
            entries=packed,
            original_count=original_count,
            compressed_count=len(packed),
            duplicates_removed=dups_removed,
            deltas_applied=deltas_applied,
            estimated_token_savings=original_tokens - tokens_used,
            compression_ratio=round(ratio, 4),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_value(value: str) -> str:
        """Fast hash for dedup comparison."""
        return hashlib.md5(value.encode()).hexdigest()[:16]
