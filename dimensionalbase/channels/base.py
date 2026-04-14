"""
Channel — abstract interface for all communication channels.

Text is always on. Embedding requires a model. Tensor requires GPU.
Your agent code is identical regardless of which channel is active.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

import numpy as np

from dimensionalbase.core.entry import KnowledgeEntry
from dimensionalbase.core.types import ChannelCapability, ChannelLevel


class Channel(ABC):
    """Abstract base for a storage/communication channel."""

    @abstractmethod
    def store(self, entry: KnowledgeEntry) -> None:
        """Store a knowledge entry."""
        ...

    @abstractmethod
    def retrieve(self, path: str) -> Optional[KnowledgeEntry]:
        """Retrieve an entry by exact path."""
        ...

    @abstractmethod
    def query_by_path(self, pattern: str) -> List[KnowledgeEntry]:
        """Retrieve entries matching a glob pattern."""
        ...

    @abstractmethod
    def delete(self, path: str) -> bool:
        """Delete an entry by path. Returns True if found."""
        ...

    @abstractmethod
    def all_entries(self) -> List[KnowledgeEntry]:
        """Return all stored entries."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Number of entries stored."""
        ...

    @abstractmethod
    def capability(self) -> ChannelCapability:
        """What this channel can do."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources."""
        ...
