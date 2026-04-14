"""
ActiveReasoning — the fourth participant in any multi-agent system.

v0.3 optimizations:
  - SINGLE prefetch: one query_by_path() call shared across all checks
    (was: 3 separate identical queries per write)
  - LSH-accelerated contradiction detection: fingerprint.query() narrows
    candidates from O(n) to O(k) before full embedding comparison
  - Pre-normalized embeddings: dot product = cosine, no norm calls
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from dimensionalbase.channels.manager import ChannelManager
from dimensionalbase.core.entry import KnowledgeEntry
from dimensionalbase.core.types import EntryType, Event, EventType
from dimensionalbase.events.bus import EventBus

logger = logging.getLogger("dimensionalbase.reasoning")

CONTRADICTION_SIMILARITY_THRESHOLD = 0.75
STALENESS_THRESHOLD_SECONDS = 3600
SUMMARY_ENTRY_THRESHOLD = 10


class ActiveReasoning:
    """Active reasoning with optimized single-prefetch and LSH filtering."""

    def __init__(
        self,
        channel_manager: ChannelManager,
        event_bus: EventBus,
        staleness_threshold: float = STALENESS_THRESHOLD_SECONDS,
        summary_threshold: int = SUMMARY_ENTRY_THRESHOLD,
        contradiction_threshold: float = CONTRADICTION_SIMILARITY_THRESHOLD,
        fingerprint=None,
    ):
        self._channels = channel_manager
        self._bus = event_bus
        self._staleness_threshold = staleness_threshold
        self._summary_threshold = summary_threshold
        self._contradiction_threshold = contradiction_threshold
        self._fingerprint = fingerprint  # SemanticFingerprint for O(1) candidate filtering
        self._prefix_counts: Dict[str, int] = {}

    def on_write(self, entry: KnowledgeEntry) -> List[Event]:
        """Run all reasoning checks. SINGLE prefetch, shared across checks."""
        events: List[Event] = []
        prefix = self._get_prefix(entry.path)

        # ── SINGLE PREFETCH ─────────────────────────────────
        # One query, used by contradiction + staleness + summarization
        prefetched = self._channels.query_by_path(prefix + "/**") if prefix else []

        # 1. Contradiction (uses prefetch + LSH filter)
        events.extend(self._check_contradictions(entry, prefetched))

        # 2. Gap detection (plan entries only, queries refs)
        if entry.type == EntryType.PLAN:
            events.extend(self._check_gaps(entry))

        # 3. Staleness (reuses same prefetch)
        events.extend(self._check_staleness(entry, prefetched))

        # 4. Auto-summarization
        summary = self._check_summarization(entry, prefetched)
        if summary:
            events.append(summary)

        for event in events:
            self._bus.emit(event)
        return events

    def _check_contradictions(self, new_entry: KnowledgeEntry,
                               prefetched: List[KnowledgeEntry]) -> List[Event]:
        """Detect contradictions. Uses LSH for fast candidate filtering."""
        events = []

        # Filter to different owners from prefetched (already loaded)
        candidates = [
            e for e in prefetched
            if e.owner != new_entry.owner and e.path != new_entry.path
        ]
        if not candidates:
            return events

        # ── LSH FAST PATH ───────────────────────────────────
        # Only use LSH when there are many candidates (> 50).
        # For small knowledge bases the O(n) full scan is negligible
        # and LSH's approximate hashing can miss close entries.
        if (self._fingerprint and new_entry.has_embedding
                and self._channels.has_embeddings
                and len(candidates) > 50):
            lsh_results = self._fingerprint.query(
                new_entry.embedding,
                threshold=self._contradiction_threshold - 0.15,
                max_results=30,
            )
            lsh_paths = {path for path, _ in lsh_results}
            # Intersect LSH candidates with prefix-matched candidates
            candidates = [e for e in candidates if e.path in lsh_paths]

        for existing in candidates:
            is_contradiction = False
            similarity = 0.0

            if (self._channels.has_embeddings
                    and new_entry.has_embedding and existing.has_embedding):
                # Pre-normalized: dot product = cosine similarity
                sim = float(np.dot(new_entry.embedding, existing.embedding))
                similarity = sim
                if sim >= self._contradiction_threshold:
                    if new_entry.value.strip().lower() != existing.value.strip().lower():
                        is_contradiction = True
            else:
                # Text-only heuristic
                if (new_entry.type == existing.type
                        and self._paths_overlap(new_entry.path, existing.path)
                        and new_entry.value.strip().lower() != existing.value.strip().lower()):
                    is_contradiction = True
                    similarity = 0.5

            if is_contradiction:
                events.append(Event(
                    type=EventType.CONFLICT, path=new_entry.path,
                    data={
                        "new_entry_path": new_entry.path,
                        "new_entry_owner": new_entry.owner,
                        "new_entry_value": new_entry.value[:200],
                        "existing_entry_path": existing.path,
                        "existing_entry_owner": existing.owner,
                        "existing_entry_value": existing.value[:200],
                        "similarity": similarity,
                        "new_confidence": new_entry.confidence,
                        "existing_confidence": existing.confidence,
                    },
                    source_owner=new_entry.owner, timestamp=time.time(),
                ))
                logger.warning(
                    f"CONFLICT: {new_entry.owner}@{new_entry.path} "
                    f"vs {existing.owner}@{existing.path} (sim={similarity:.3f})"
                )
        return events

    def _check_gaps(self, plan_entry: KnowledgeEntry) -> List[Event]:
        events = []
        for ref_path in plan_entry.refs:
            related = self._channels.query_by_path(ref_path)
            observations = [e for e in related if e.type == EntryType.OBSERVATION]
            if not observations:
                events.append(Event(
                    type=EventType.GAP, path=ref_path,
                    data={
                        "plan_path": plan_entry.path,
                        "plan_owner": plan_entry.owner,
                        "missing_observation_for": ref_path,
                        "plan_value": plan_entry.value[:200],
                    },
                    source_owner=plan_entry.owner, timestamp=time.time(),
                ))
        return events

    def _check_staleness(self, new_entry: KnowledgeEntry,
                          prefetched: List[KnowledgeEntry]) -> List[Event]:
        """Check staleness using prefetched results (no extra query)."""
        events = []
        now = time.time()
        for entry in prefetched:
            if entry.path == new_entry.path:
                continue
            age = now - entry.updated_at
            if age > self._staleness_threshold:
                events.append(Event(
                    type=EventType.STALE, path=entry.path,
                    data={
                        "stale_entry_path": entry.path,
                        "stale_entry_owner": entry.owner,
                        "age_seconds": age,
                        "age_human": self._format_age(age),
                        "triggered_by": new_entry.path,
                    },
                    source_owner=entry.owner, timestamp=now,
                ))
        return events

    def _check_summarization(self, entry: KnowledgeEntry,
                              prefetched: List[KnowledgeEntry]) -> Optional[Event]:
        prefix = self._get_prefix(entry.path)
        if not prefix:
            return None
        self._prefix_counts[prefix] = self._prefix_counts.get(prefix, 0) + 1
        if self._prefix_counts[prefix] >= self._summary_threshold:
            self._prefix_counts[prefix] = 0
            summary = self._generate_summary(prefetched)
            return Event(
                type=EventType.SUMMARY, path=prefix,
                data={"prefix": prefix, "entry_count": len(prefetched), "summary": summary},
                timestamp=time.time(),
            )
        return None

    def check_all_staleness(self) -> List[Event]:
        events = []
        now = time.time()
        for entry in self._channels.all_entries():
            age = now - entry.updated_at
            if age > self._staleness_threshold:
                event = Event(
                    type=EventType.STALE, path=entry.path,
                    data={"stale_entry_path": entry.path, "stale_entry_owner": entry.owner,
                          "age_seconds": age, "age_human": self._format_age(age)},
                    source_owner=entry.owner, timestamp=now,
                )
                events.append(event)
                self._bus.emit(event)
        return events

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _get_prefix(path: str) -> str:
        parts = path.rsplit("/", 1)
        return parts[0] if len(parts) > 1 else ""

    @staticmethod
    def _paths_overlap(a: str, b: str) -> bool:
        parts_a = a.split("/")
        parts_b = b.split("/")
        return any(pa == pb for pa, pb in zip(parts_a, parts_b))

    @staticmethod
    def _generate_summary(entries: List[KnowledgeEntry]) -> str:
        if not entries:
            return "No entries."
        owners = set(e.owner for e in entries)
        types: Dict[str, int] = {}
        for e in entries:
            types[e.type.value] = types.get(e.type.value, 0) + 1
        latest = max(entries, key=lambda e: e.updated_at)
        highest_conf = max(entries, key=lambda e: e.confidence)
        return (
            f"{len(entries)} entries by {len(owners)} agent(s). "
            f"Types: {', '.join(f'{t}={c}' for t, c in types.items())}. "
            f"Latest: [{latest.path}] {latest.value[:80]} "
            f"Highest confidence: [{highest_conf.path}] (conf={highest_conf.confidence:.2f})"
        )

    @staticmethod
    def _format_age(seconds: float) -> str:
        if seconds < 60: return f"{seconds:.0f}s"
        if seconds < 3600: return f"{seconds / 60:.1f}m"
        if seconds < 86400: return f"{seconds / 3600:.1f}h"
        return f"{seconds / 86400:.1f}d"
