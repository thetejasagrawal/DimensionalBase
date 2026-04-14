"""Tests for the MCP server tool and resource handlers."""

import json
import pytest

from dimensionalbase import DimensionalBase


class TestMCPServer:
    """Test MCP tool handlers using direct DB calls (no actual MCP transport)."""

    def test_put_and_get_roundtrip(self, db):
        """Verify entries written via put can be read via get."""
        db.put("task/auth", "JWT expired", owner="agent-a", type="fact", confidence=0.9)
        result = db.get("task/**", budget=500)
        assert len(result.entries) == 1
        assert result.entries[0].path == "task/auth"

    def test_status_returns_dict(self, db):
        """Status should return a dict with standard keys."""
        status = db.status()
        assert "entries" in status
        assert "channel" in status
        assert "reasoning" in status

    def test_subscribe_buffers_events(self, db):
        """Events from subscriptions should be capturable."""
        events = []
        db.subscribe("task/**", "test", lambda e: events.append(e))
        db.put("task/x", "value", owner="a")
        assert len(events) >= 1
        assert events[0].path == "task/x"

    def test_relate_returns_result_or_none(self, db):
        """Relate should return a dict (with embeddings) or None (without)."""
        db.put("a", "hello", owner="x")
        db.put("b", "world", owner="x")
        result = db.relate("a", "b")
        if db.has_embeddings:
            assert isinstance(result, dict)
            assert "cosine" in result or "angular_dist" in result
        else:
            assert result is None

    def test_compose_returns_result_or_none(self, db):
        """Compose should return a vector (with embeddings) or None (without)."""
        db.put("a", "hello", owner="x")
        db.put("b", "world", owner="x")
        result = db.compose(["a", "b"])
        if db.has_embeddings:
            assert result is not None
        else:
            assert result is None

    def test_tool_definitions_structure(self):
        """Tool definitions should have required fields."""
        defs = DimensionalBase.tool_definitions()
        assert len(defs) >= 3
        for d in defs:
            assert "name" in d
            assert "description" in d
            assert "parameters" in d
