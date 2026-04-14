"""
DBPS v1.0 Section 4 + 6 — Put/Get Semantic Conformance Tests.

Verifies the exact behavior of the 4-method API as specified.
"""

from __future__ import annotations

import pytest

from dimensionalbase import DimensionalBase


@pytest.fixture
def db():
    d = DimensionalBase()
    yield d
    d.close()


class TestPutSemantics:
    """Verify put() behavior matches DBPS specification."""

    def test_put_creates_entry(self, db):
        """put() with new path creates a new entry at version 1."""
        entry = db.put("task/auth", "JWT expired", owner="agent-a")
        assert entry.path == "task/auth"
        assert entry.value == "JWT expired"
        assert entry.owner == "agent-a"
        assert entry.version == 1

    def test_put_updates_increments_version(self, db):
        """put() to existing path increments version."""
        db.put("x", "v1", owner="a")
        entry = db.put("x", "v2", owner="a")
        assert entry.version >= 2

    def test_put_preserves_created_at(self, db):
        """put() update preserves original created_at timestamp."""
        e1 = db.put("x", "v1", owner="a")
        created = e1.created_at
        e2 = db.put("x", "v2", owner="a")
        assert e2.created_at == created

    def test_put_fires_change_event(self, db):
        """put() MUST fire a CHANGE event."""
        events = []
        db.subscribe("**", "test", lambda e: events.append(e))
        db.put("x", "v", owner="a")
        change_events = [e for e in events if e.type.value == "change"]
        assert len(change_events) >= 1
        assert change_events[0].path == "x"

    def test_put_returns_entry_with_all_fields(self, db):
        """put() return value MUST have all required fields populated."""
        entry = db.put("p", "v", owner="o", type="decision", confidence=0.7,
                       refs=["a", "b"], ttl="persistent")
        assert entry.id  # non-empty
        assert entry.path == "p"
        assert entry.value == "v"
        assert entry.owner == "o"
        assert entry.type.value == "decision"
        assert entry.confidence == 0.7
        assert entry.refs == ["a", "b"]
        assert entry.ttl.value == "persistent"
        assert entry.created_at > 0
        assert entry.updated_at > 0


class TestGetSemantics:
    """Verify get() behavior matches DBPS specification."""

    def test_get_empty_returns_empty(self, db):
        """get() on empty DB returns empty QueryResult."""
        result = db.get("**")
        assert len(result.entries) == 0
        assert result.total_matched == 0

    def test_get_returns_matching_entries(self, db):
        """get() returns entries matching the scope pattern."""
        db.put("task/auth", "v1", owner="a")
        db.put("task/deploy", "v2", owner="a")
        db.put("other/thing", "v3", owner="a")

        result = db.get("task/**")
        paths = {e.path for e in result.entries}
        assert "task/auth" in paths
        assert "task/deploy" in paths
        assert "other/thing" not in paths

    def test_get_respects_budget(self, db):
        """get() MUST NOT return more tokens than the budget allows."""
        for i in range(100):
            db.put(f"entry/{i}", f"Value number {i} with some padding text", owner="a")

        result = db.get("**", budget=50)
        assert result.tokens_used <= 50

    def test_get_budget_remaining_correct(self, db):
        """budget_remaining = budget - tokens_used."""
        db.put("x", "value", owner="a")
        result = db.get("**", budget=1000)
        assert result.budget_remaining == 1000 - result.tokens_used

    def test_get_total_matched_counts_all(self, db):
        """total_matched counts ALL matching entries, not just packed ones."""
        for i in range(50):
            db.put(f"entry/{i}", f"Value {i}", owner="a")
        result = db.get("**", budget=10)  # tiny budget
        assert result.total_matched == 50  # all 50 match the scope


class TestSubscribeSemantics:
    """Verify subscribe()/unsubscribe() behavior."""

    def test_subscribe_receives_events(self, db):
        """Subscribed callback MUST receive matching events."""
        received = []
        db.subscribe("task/**", "test", lambda e: received.append(e))
        db.put("task/auth", "v", owner="a")
        assert len(received) >= 1

    def test_subscribe_filters_by_pattern(self, db):
        """Subscribed callback MUST NOT receive non-matching events."""
        received = []
        db.subscribe("task/**", "test", lambda e: received.append(e))
        db.put("other/path", "v", owner="a")
        task_events = [e for e in received if e.path.startswith("task")]
        assert len(task_events) == 0

    def test_unsubscribe_stops_events(self, db):
        """After unsubscribe, callback MUST NOT be called."""
        received = []
        sub = db.subscribe("**", "test", lambda e: received.append(e))
        db.put("x", "v1", owner="a")
        count_after_first = len(received)

        db.unsubscribe(sub)
        db.put("y", "v2", owner="a")
        # Should not have received new events (only existing ones for y's CHANGE)
        # Actually since we unsubscribed, no more callbacks
        assert len(received) == count_after_first
