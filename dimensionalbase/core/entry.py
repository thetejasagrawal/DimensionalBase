"""
KnowledgeEntry — the atomic unit of shared knowledge in DimensionalBase.

Every piece of data an agent writes into the system becomes one of these.
The embedding IS the primary data. The text is a human-readable shadow.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from dimensionalbase.core.types import EntryType, TTL
from dimensionalbase.exceptions import EntryValidationError


@dataclass
class KnowledgeEntry:
    """A single piece of knowledge in the shared dimensional space.

    Attributes:
        path:       Hierarchical address (e.g., 'task/auth/status').
        value:      Human-readable text content.
        owner:      Which agent wrote this (single source of truth).
        type:       fact | decision | plan | observation.
        confidence: 0.0–1.0, self-reported by the writing agent.
        refs:       Paths to related entries (this IS the graph).
        embedding:  High-dimensional vector (the real data).
        version:    Incremented on every update.
        ttl:        turn | session | persistent.
        id:         Unique identifier for this entry.
        created_at: Unix timestamp when first written.
        updated_at: Unix timestamp of last update.
        metadata:   Arbitrary key-value pairs for extensibility.
    """

    # === Required fields ===
    path: str
    value: str
    owner: str

    # === Optional fields with smart defaults ===
    type: EntryType = EntryType.FACT
    confidence: float = 1.0
    refs: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    version: int = 1
    ttl: TTL = TTL.SESSION
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, str] = field(default_factory=dict)

    # === Computed at query time, not stored ===
    _raw_score: float = field(default=0.0, repr=False, compare=False)
    _score: float = field(default=0.0, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate entry on creation."""
        if not self.path:
            raise EntryValidationError("path cannot be empty")
        if not self.value:
            raise EntryValidationError("value cannot be empty")
        if not self.owner:
            raise EntryValidationError("owner cannot be empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise EntryValidationError(f"confidence must be 0.0–1.0, got {self.confidence}")
        if isinstance(self.type, str):
            self.type = EntryType(self.type)
        if isinstance(self.ttl, str):
            self.ttl = TTL(self.ttl)

    @property
    def has_embedding(self) -> bool:
        """Whether this entry has a dimensional representation."""
        return self.embedding is not None

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding, or 0 if none."""
        if self.embedding is None:
            return 0
        return len(self.embedding)

    @property
    def token_estimate(self) -> int:
        """Rough token count for budget calculations (~4 chars per token)."""
        base = len(f"[{self.path}] ({self.owner}, confidence={self.confidence:.2f}): {self.value}")
        return max(1, base // 4)

    @property
    def compact_token_estimate(self) -> int:
        """Token count for compressed representation."""
        base = len(f"[{self.path}]: {self.value[:100]}")
        return max(1, base // 4)

    @property
    def path_only_token_estimate(self) -> int:
        """Token count for path-only representation."""
        return max(1, len(self.path) // 4)

    def update(self, value: str, confidence: Optional[float] = None,
               refs: Optional[List[str]] = None, embedding: Optional[np.ndarray] = None,
               metadata: Optional[Dict[str, str]] = None) -> None:
        """Update this entry in place, bumping the version."""
        self.value = value
        self.version += 1
        self.updated_at = time.time()
        if confidence is not None:
            if not (0.0 <= confidence <= 1.0):
                raise EntryValidationError(f"confidence must be 0.0–1.0, got {confidence}")
            self.confidence = confidence
        if refs is not None:
            self.refs = refs
        if embedding is not None:
            self.embedding = embedding
        if metadata is not None:
            self.metadata.update(metadata)

    def to_full_text(self) -> str:
        """Full text representation for LLM context windows."""
        return f"[{self.path}] ({self.owner}, confidence={self.confidence:.2f}): {self.value}"

    def to_compact_text(self) -> str:
        """Compressed representation when budget is tight."""
        truncated = self.value[:100] + "..." if len(self.value) > 100 else self.value
        return f"[{self.path}]: {truncated}"

    def to_path_only(self) -> str:
        """Minimal representation — just the path."""
        return self.path

    def to_dict(self) -> Dict:
        """Serialize to dictionary (for SQLite storage)."""
        d = {
            "id": self.id,
            "path": self.path,
            "value": self.value,
            "owner": self.owner,
            "type": self.type.value,
            "confidence": self.confidence,
            "refs": ",".join(self.refs) if self.refs else "",
            "version": self.version,
            "ttl": self.ttl.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        # Metadata stored as key=value pairs
        for k, v in self.metadata.items():
            d[f"meta_{k}"] = v
        return d

    @classmethod
    def from_dict(cls, d: Dict, embedding: Optional[np.ndarray] = None) -> KnowledgeEntry:
        """Deserialize from dictionary (from SQLite storage)."""
        refs_str = d.get("refs", "")
        refs = [r for r in refs_str.split(",") if r] if refs_str else []

        metadata = {}
        for k, v in d.items():
            if k.startswith("meta_"):
                metadata[k[5:]] = v

        return cls(
            id=d["id"],
            path=d["path"],
            value=d["value"],
            owner=d["owner"],
            type=EntryType(d["type"]),
            confidence=d["confidence"],
            refs=refs,
            embedding=embedding,
            version=d["version"],
            ttl=TTL(d["ttl"]),
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            metadata=metadata,
        )

    def __repr__(self) -> str:
        emb = f"dim={self.embedding_dim}" if self.has_embedding else "no-emb"
        return (
            f"KnowledgeEntry(path='{self.path}', owner='{self.owner}', "
            f"v{self.version}, {self.type.value}, conf={self.confidence:.2f}, {emb})"
        )
