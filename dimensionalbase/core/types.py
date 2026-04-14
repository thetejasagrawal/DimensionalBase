"""
DimensionalBase core type definitions.

Every type in the system lives here. Single source of truth.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EntryType(str, enum.Enum):
    """What kind of knowledge this entry represents."""
    FACT = "fact"
    DECISION = "decision"
    PLAN = "plan"
    OBSERVATION = "observation"


class TTL(str, enum.Enum):
    """How long this entry should live."""
    TURN = "turn"           # Dies after one agent turn
    SESSION = "session"     # Dies when the session ends
    PERSISTENT = "persistent"  # Lives forever (until explicitly deleted)


class ChannelLevel(int, enum.Enum):
    """Communication fidelity levels — higher is richer."""
    TEXT = 1        # Lossy, universal fallback
    EMBEDDING = 2   # Near-lossless, requires shared embedding model
    TENSOR = 3      # Lossless, requires shared GPU hardware


class EventType(str, enum.Enum):
    """Events the active reasoning layer can fire."""
    CHANGE = "change"           # An entry was created or updated
    DELETE = "delete"           # An entry was deleted
    CONFLICT = "conflict"       # Two agents contradict each other
    GAP = "gap"                 # A plan step has no corresponding observation
    STALE = "stale"             # An entry is older than its staleness threshold
    SUMMARY = "summary"         # A prefix was auto-summarized


# ---------------------------------------------------------------------------
# Scoring weights for the context engine
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoringWeights:
    """Weights for ranking knowledge entries during retrieval.

    When ``adaptive`` is True (the default), the context engine adjusts weights
    based on whether a semantic query is provided: queries boost similarity,
    no-query boosts recency.  Set ``adaptive=False`` to use exact weights.
    """
    recency: float = 0.30
    confidence: float = 0.20
    similarity: float = 0.30
    reference_distance: float = 0.20
    adaptive: bool = True

    def __post_init__(self) -> None:
        total = self.recency + self.confidence + self.similarity + self.reference_distance
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total}")


# ---------------------------------------------------------------------------
# Event payload
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """An event fired by the system."""
    type: EventType
    path: str
    data: Dict[str, Any] = field(default_factory=dict)
    source_owner: Optional[str] = None
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Subscription handle
# ---------------------------------------------------------------------------

@dataclass
class Subscription:
    """A handle to an active subscription."""
    id: str
    pattern: str
    subscriber: str
    callback: Callable[[Event], None]
    active: bool = True


# ---------------------------------------------------------------------------
# Query result
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """What db.get() returns."""
    entries: List[Any]  # List[KnowledgeEntry] — forward ref avoids circular import
    total_matched: int
    tokens_used: int
    budget_remaining: int
    channel_used: ChannelLevel
    summary: Optional[str] = None

    @property
    def text(self) -> str:
        """Concatenated text of all returned entries, with metadata prefix."""
        parts = []
        for entry in self.entries:
            parts.append(f"[{entry.path}] ({entry.owner}, confidence={entry.confidence:.2f}): {entry.value}")
        return "\n".join(parts)

    @property
    def raw_text(self) -> str:
        """Clean text without metadata — maximum signal for LLM context windows."""
        return "\n\n".join(entry.value for entry in self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return len(self.entries) > 0


# ---------------------------------------------------------------------------
# Channel capability descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelCapability:
    """Describes what a channel can do."""
    level: ChannelLevel
    available: bool
    description: str
    latency_estimate: str  # e.g., "microseconds", "milliseconds"
    information_loss: str  # e.g., "zero", "minimal", "significant"
