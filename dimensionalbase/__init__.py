"""
DimensionalBase — The protocol and database for AI communication.

    pip install dimensionalbase

    from dimensionalbase import DimensionalBase

    db = DimensionalBase()
    db.put("task/auth/status", "JWT expired", owner="backend-agent")
    context = db.get("task/**", budget=500, query="What's blocking deploy?")

Four methods. Three channels. Deterministic startup.
Under the hood: dimensional algebra, Bayesian confidence, provenance
tracking, agent trust, semantic fingerprinting, adaptive compression.

DB, evolved.
"""

__version__ = "0.5.0"

from dimensionalbase.config import DimensionalBaseConfig
from dimensionalbase.db import DimensionalBase
from dimensionalbase.core.types import (
    ChannelLevel,
    EntryType,
    Event,
    EventType,
    QueryResult,
    ScoringWeights,
    Subscription,
    TTL,
)
from dimensionalbase.core.entry import KnowledgeEntry
from dimensionalbase.exceptions import (
    BudgetExhaustedError,
    ChannelError,
    ConflictError,
    DimensionalBaseError,
    EmbeddingError,
    EntryValidationError,
    RateLimitError,
    SchemaVersionError,
    StorageError,
)

__all__ = [
    "DimensionalBase",
    "DimensionalBaseConfig",
    "KnowledgeEntry",
    "ChannelLevel",
    "EntryType",
    "Event",
    "EventType",
    "QueryResult",
    "ScoringWeights",
    "Subscription",
    "TTL",
    "DimensionalBaseError",
    "EntryValidationError",
    "StorageError",
    "ChannelError",
    "EmbeddingError",
    "SchemaVersionError",
    "BudgetExhaustedError",
    "ConflictError",
    "RateLimitError",
]
