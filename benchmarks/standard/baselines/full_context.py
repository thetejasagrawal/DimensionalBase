"""
Full context baseline for benchmark comparison.

Represents the traditional approach: pass the entire document as context.
No retrieval, no compression — just raw text.
"""

from __future__ import annotations


class FullContextBaseline:
    """Baseline that passes the full document text.

    This represents the upper bound on information (no retrieval loss)
    and the worst case for token efficiency.
    """

    def __init__(self) -> None:
        self._context: str = ""
        self._total_tokens: int = 0

    def ingest(self, text: str) -> int:
        """Store the full document."""
        self._context = text
        self._total_tokens = len(text) // 4
        return self._total_tokens

    def query(self, question: str, budget: int = 999999) -> str:
        """Return the full context (truncated to budget if necessary)."""
        if budget >= self._total_tokens:
            return self._context

        # Truncate to fit budget
        char_budget = budget * 4
        return self._context[:char_budget]

    def reset(self) -> None:
        self._context = ""
        self._total_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self._total_tokens
