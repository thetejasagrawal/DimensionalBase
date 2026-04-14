"""
DBPS v1.0 Section 14 — Budget Packing Conformance Tests.

Verifies that budget-aware retrieval respects token limits and uses
the three-tier fallback (full → compact → path-only).
"""

from __future__ import annotations

import pytest

from dimensionalbase import DimensionalBase


@pytest.fixture
def db():
    d = DimensionalBase()
    yield d
    d.close()


class TestBudgetPacking:
    """Verify budget-aware context packing."""

    def test_zero_budget_returns_empty(self, db):
        """Budget=0 MUST return zero entries."""
        db.put("x", "value", owner="a")
        result = db.get("**", budget=0)
        assert len(result.entries) == 0

    def test_budget_never_exceeded(self, db):
        """tokens_used MUST NOT exceed budget."""
        for i in range(100):
            db.put(f"entry/{i}", f"This is entry number {i} with some content", owner="a")
        for budget in [10, 50, 100, 500, 1000, 5000]:
            result = db.get("**", budget=budget)
            assert result.tokens_used <= budget, \
                f"Budget {budget} exceeded: {result.tokens_used} tokens used"

    def test_larger_budget_returns_more(self, db):
        """Larger budget SHOULD return at least as many entries."""
        for i in range(50):
            db.put(f"entry/{i}", f"Content for entry {i}", owner="a")
        small = db.get("**", budget=100)
        large = db.get("**", budget=5000)
        assert len(large.entries) >= len(small.entries)

    def test_budget_remaining_correct(self, db):
        """budget_remaining MUST equal budget - tokens_used."""
        db.put("x", "short value", owner="a")
        result = db.get("**", budget=500)
        assert result.budget_remaining == 500 - result.tokens_used

    def test_total_matched_independent_of_budget(self, db):
        """total_matched counts ALL matching entries regardless of budget."""
        for i in range(20):
            db.put(f"x/{i}", f"value {i}", owner="a")
        result = db.get("**", budget=1)  # tiny budget
        assert result.total_matched == 20
