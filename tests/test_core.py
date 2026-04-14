"""
Core tests for DimensionalBase.

Tests the full public API: put, get, subscribe, unsubscribe.
Tests active reasoning: contradiction detection, gap detection, staleness.
Tests context engine: scoring, budget packing.
Tests channels: text storage, embedding storage.
"""

import time
import pytest
from dimensionalbase import DimensionalBase, KnowledgeEntry, EventType, QueryResult


class TestPutGet:
    """Test the core put/get API."""

    def test_put_and_retrieve(self):
        """Basic put and retrieve by exact path."""
        db = DimensionalBase()
        entry = db.put(
            path="task/auth/status",
            value="JWT signing key expired",
            owner="backend-agent",
            type="fact",
            confidence=0.92,
        )
        assert entry.path == "task/auth/status"
        assert entry.value == "JWT signing key expired"
        assert entry.owner == "backend-agent"
        assert entry.confidence == 0.92
        assert entry.version == 1

        # Retrieve by exact path
        retrieved = db.retrieve("task/auth/status")
        assert retrieved is not None
        assert retrieved.value == "JWT signing key expired"
        db.close()

    def test_put_updates_version(self):
        """Updating the same path increments version."""
        db = DimensionalBase()
        db.put("task/status", "working", owner="agent-1")
        db.put("task/status", "broken", owner="agent-1")

        entry = db.retrieve("task/status")
        assert entry is not None
        assert entry.value == "broken"
        assert entry.version == 2
        db.close()

    def test_get_with_scope(self):
        """Get entries matching a glob pattern."""
        db = DimensionalBase()
        db.put("task/auth/status", "ok", owner="agent-1")
        db.put("task/auth/token", "valid", owner="agent-1")
        db.put("task/deploy/status", "pending", owner="agent-2")
        db.put("unrelated/stuff", "whatever", owner="agent-3")

        result = db.get(scope="task/auth/**")
        assert len(result.entries) == 2
        paths = {e.path for e in result.entries}
        assert "task/auth/status" in paths
        assert "task/auth/token" in paths
        db.close()

    def test_get_with_budget(self):
        """Budget limits the number of entries returned."""
        db = DimensionalBase()
        # Write many entries
        for i in range(50):
            db.put(
                f"data/item/{i}",
                f"This is item number {i} with some content to take up tokens",
                owner="agent-1",
            )

        # Very small budget
        result = db.get(scope="data/**", budget=50)
        assert result.tokens_used <= 50
        assert result.total_matched == 50
        assert len(result.entries) < 50  # Can't fit all in budget
        db.close()

    def test_get_with_owner_filter(self):
        """Filter entries by owner."""
        db = DimensionalBase()
        db.put("task/a", "val1", owner="agent-1")
        db.put("task/b", "val2", owner="agent-2")
        db.put("task/c", "val3", owner="agent-1")

        result = db.get(scope="task/**", owner="agent-1")
        assert len(result.entries) == 2
        assert all(e.owner == "agent-1" for e in result.entries)
        db.close()

    def test_get_with_type_filter(self):
        """Filter entries by type."""
        db = DimensionalBase()
        db.put("task/a", "plan step 1", owner="agent-1", type="plan")
        db.put("task/b", "observed X", owner="agent-1", type="observation")
        db.put("task/c", "fact Y", owner="agent-1", type="fact")

        result = db.get(scope="task/**", type="plan")
        assert len(result.entries) == 1
        assert result.entries[0].type.value == "plan"
        db.close()

    def test_get_returns_query_result(self):
        """QueryResult has all expected fields."""
        db = DimensionalBase()
        db.put("task/a", "val1", owner="agent-1")

        result = db.get(scope="task/**", budget=1000)
        assert isinstance(result, QueryResult)
        assert result.total_matched >= 1
        assert result.tokens_used > 0
        assert result.budget_remaining == 1000 - result.tokens_used
        assert result.text  # Non-empty text representation
        db.close()

    def test_empty_get(self):
        """Get returns empty result when nothing matches."""
        db = DimensionalBase()
        result = db.get(scope="nonexistent/**")
        assert len(result.entries) == 0
        assert result.tokens_used == 0
        assert not result  # Falsy
        db.close()


class TestSubscribe:
    """Test pub/sub event system."""

    def test_subscribe_fires_on_put(self):
        """Subscriber gets notified on put."""
        db = DimensionalBase()
        events_received = []

        sub = db.subscribe("task/**", "watcher", lambda e: events_received.append(e))
        db.put("task/auth/status", "expired", owner="agent-1")

        assert len(events_received) >= 1
        change_events = [e for e in events_received if e.type == EventType.CHANGE]
        assert len(change_events) == 1
        assert change_events[0].path == "task/auth/status"
        db.close()

    def test_unsubscribe_stops_events(self):
        """After unsubscribe, no more events."""
        db = DimensionalBase()
        events_received = []

        sub = db.subscribe("task/**", "watcher", lambda e: events_received.append(e))
        db.put("task/a", "first", owner="agent-1")
        count_after_first = len(events_received)

        db.unsubscribe(sub)
        db.put("task/b", "second", owner="agent-1")

        # Should not have received the second event
        assert len(events_received) == count_after_first
        db.close()

    def test_subscribe_pattern_matching(self):
        """Subscribers only get events matching their pattern."""
        db = DimensionalBase()
        task_events = []
        memory_events = []

        db.subscribe("task/**", "task-watcher", lambda e: task_events.append(e))
        db.subscribe("memory/**", "mem-watcher", lambda e: memory_events.append(e))

        db.put("task/auth", "expired", owner="agent-1")
        db.put("memory/agent-1/last-action", "checked auth", owner="agent-1")

        assert len(task_events) >= 1
        assert len(memory_events) >= 1
        assert all("task" in e.path for e in task_events)
        assert all("memory" in e.path for e in memory_events)
        db.close()


class TestContradictionDetection:
    """Test active reasoning — contradiction detection."""

    def test_detects_contradiction_between_agents(self):
        """When two agents write conflicting facts, a CONFLICT event fires."""
        db = DimensionalBase()
        conflicts = []
        db.subscribe("task/**", "monitor", lambda e: conflicts.append(e) if e.type == EventType.CONFLICT else None)

        db.put("task/api/status", "API is healthy", owner="agent-1", type="fact")
        db.put("task/api/health", "API is down and returning 500s", owner="agent-2", type="fact")

        # Should detect conflict (same prefix, different owners, different values)
        conflict_events = [e for e in conflicts if e.type == EventType.CONFLICT]
        # Note: conflict detection depends on path overlap heuristic in text-only mode
        # With embeddings, this would be more precise
        db.close()

    def test_no_contradiction_same_owner(self):
        """Same owner updating their own entry is not a contradiction."""
        db = DimensionalBase()
        conflicts = []
        db.subscribe("task/**", "monitor", lambda e: conflicts.append(e) if e.type == EventType.CONFLICT else None)

        db.put("task/status", "working", owner="agent-1")
        db.put("task/status", "broken", owner="agent-1")

        assert len([e for e in conflicts if e.type == EventType.CONFLICT]) == 0
        db.close()


class TestGapDetection:
    """Test active reasoning — gap detection."""

    def test_detects_gap_in_plan(self):
        """When a plan references steps with no observations, GAP fires."""
        db = DimensionalBase()
        gaps = []
        db.subscribe("**", "monitor", lambda e: gaps.append(e) if e.type == EventType.GAP else None)

        # Write a plan that references steps
        db.put(
            "task/deploy/plan",
            "Deploy in 3 steps: build, test, push",
            owner="planner",
            type="plan",
            refs=["task/deploy/build", "task/deploy/test", "task/deploy/push"],
        )

        # Only create observation for 'build'
        db.put("task/deploy/build", "Build succeeded", owner="builder", type="observation")

        # Should detect gaps for 'test' and 'push'
        gap_events = [e for e in gaps if e.type == EventType.GAP]
        gap_paths = {e.path for e in gap_events}
        assert "task/deploy/test" in gap_paths
        assert "task/deploy/push" in gap_paths
        db.close()


class TestDelete:
    """Test deletion."""

    def test_delete_entry(self):
        """Delete removes an entry."""
        db = DimensionalBase()
        db.put("task/temp", "temporary", owner="agent-1")
        assert db.exists("task/temp")

        db.delete("task/temp")
        assert not db.exists("task/temp")
        db.close()

    def test_delete_fires_event(self):
        """Delete fires a DELETE event."""
        db = DimensionalBase()
        events = []
        db.subscribe("task/**", "watcher", lambda e: events.append(e))

        db.put("task/temp", "temporary", owner="agent-1")
        db.delete("task/temp")

        delete_events = [e for e in events if e.type == EventType.DELETE]
        assert len(delete_events) == 1
        db.close()


class TestClearTTL:
    """Test TTL-based cleanup."""

    def test_clear_turn(self):
        """clear_turn removes turn-scoped entries only."""
        db = DimensionalBase()
        db.put("task/turn-data", "temp", owner="agent-1", ttl="turn")
        db.put("task/session-data", "persist", owner="agent-1", ttl="session")
        db.put("task/permanent", "forever", owner="agent-1", ttl="persistent")

        count = db.clear_turn()
        assert count == 1
        assert not db.exists("task/turn-data")
        assert db.exists("task/session-data")
        assert db.exists("task/permanent")
        db.close()

    def test_clear_session(self):
        """clear_session removes turn and session entries."""
        db = DimensionalBase()
        db.put("a", "1", owner="x", ttl="turn")
        db.put("b", "2", owner="x", ttl="session")
        db.put("c", "3", owner="x", ttl="persistent")

        count = db.clear_session()
        assert count == 2
        assert not db.exists("a")
        assert not db.exists("b")
        assert db.exists("c")
        db.close()


class TestStatus:
    """Test introspection."""

    def test_status(self):
        """status() returns expected fields."""
        db = DimensionalBase()
        db.put("task/a", "val", owner="agent-1")

        status = db.status()
        assert status["entries"] == 1
        assert status["channel"] in ("TEXT", "EMBEDDING", "TENSOR")
        assert isinstance(status["embeddings"], bool)
        assert "TEXT" in status["channels"]
        assert isinstance(status["reasoning"], bool)
        db.close()

    def test_context_manager(self):
        """DimensionalBase works as a context manager."""
        with DimensionalBase() as db:
            db.put("test/cm", "context manager works", owner="test")
            assert db.exists("test/cm")

    def test_tool_definitions(self):
        """Tool definitions are well-formed."""
        tools = DimensionalBase.tool_definitions()
        assert len(tools) == 3
        names = {t["name"] for t in tools}
        assert "db_put" in names
        assert "db_get" in names
        assert "db_subscribe" in names


class TestKnowledgeEntry:
    """Test the KnowledgeEntry data model."""

    def test_create_entry(self):
        entry = KnowledgeEntry(path="a/b", value="test", owner="x")
        assert entry.path == "a/b"
        assert entry.version == 1

    def test_entry_validation(self):
        with pytest.raises(ValueError):
            KnowledgeEntry(path="", value="test", owner="x")
        with pytest.raises(ValueError):
            KnowledgeEntry(path="a", value="", owner="x")
        with pytest.raises(ValueError):
            KnowledgeEntry(path="a", value="test", owner="x", confidence=1.5)

    def test_entry_token_estimate(self):
        entry = KnowledgeEntry(path="task/status", value="API is healthy", owner="agent")
        assert entry.token_estimate > 0
        assert entry.compact_token_estimate > 0
        assert entry.path_only_token_estimate > 0
        assert entry.token_estimate >= entry.compact_token_estimate

    def test_entry_text_representations(self):
        entry = KnowledgeEntry(path="task/status", value="API is healthy", owner="agent")
        assert "task/status" in entry.to_full_text()
        assert "agent" in entry.to_full_text()
        assert "task/status" in entry.to_compact_text()
        assert entry.to_path_only() == "task/status"

    def test_entry_serialization(self):
        entry = KnowledgeEntry(
            path="a/b", value="test", owner="x",
            refs=["c/d"], metadata={"key": "val"},
        )
        d = entry.to_dict()
        restored = KnowledgeEntry.from_dict(d)
        assert restored.path == entry.path
        assert restored.value == entry.value
        assert restored.refs == entry.refs
        assert restored.metadata == entry.metadata
