"""
DimensionalBase exception hierarchy.

All exceptions inherit from DimensionalBaseError so callers can catch broadly
or narrowly as needed.
"""


class DimensionalBaseError(Exception):
    """Base exception for all DimensionalBase errors."""


class EntryValidationError(DimensionalBaseError, ValueError):
    """Raised when a KnowledgeEntry fails validation (empty path, bad confidence, etc.)."""


class StorageError(DimensionalBaseError):
    """Raised on SQLite or VectorStore failures."""


class ChannelError(DimensionalBaseError):
    """Raised when a channel operation fails."""


class EmbeddingError(DimensionalBaseError):
    """Raised when embedding generation fails."""


class SchemaVersionError(StorageError):
    """Raised when the database schema version is incompatible."""


class BudgetExhaustedError(DimensionalBaseError):
    """Raised when a token budget cannot accommodate any entries."""


class ConflictError(DimensionalBaseError):
    """Raised when a write conflicts with existing knowledge (contradiction)."""


class RateLimitError(DimensionalBaseError):
    """Raised when an API rate limit is exceeded."""
