"""Tests for error handling and edge cases."""

import pytest

from dimensionalbase import DimensionalBase
from dimensionalbase.core.entry import KnowledgeEntry
from dimensionalbase.exceptions import EntryValidationError


class TestEntryValidation:
    """Test KnowledgeEntry validation."""

    def test_empty_path_raises(self):
        with pytest.raises(EntryValidationError, match="path cannot be empty"):
            KnowledgeEntry(path="", value="x", owner="a")

    def test_empty_value_raises(self):
        with pytest.raises(EntryValidationError, match="value cannot be empty"):
            KnowledgeEntry(path="x", value="", owner="a")

    def test_empty_owner_raises(self):
        with pytest.raises(EntryValidationError, match="owner cannot be empty"):
            KnowledgeEntry(path="x", value="v", owner="")

    def test_confidence_below_zero_raises(self):
        with pytest.raises(EntryValidationError, match="confidence"):
            KnowledgeEntry(path="x", value="v", owner="a", confidence=-0.1)

    def test_confidence_above_one_raises(self):
        with pytest.raises(EntryValidationError, match="confidence"):
            KnowledgeEntry(path="x", value="v", owner="a", confidence=1.1)

    def test_valid_entry_creates(self):
        e = KnowledgeEntry(path="x", value="v", owner="a")
        assert e.path == "x"
        assert e.confidence == 1.0

    def test_update_confidence_validation(self):
        e = KnowledgeEntry(path="x", value="v", owner="a")
        with pytest.raises(EntryValidationError, match="confidence"):
            e.update("new value", confidence=2.0)


class TestDBEdgeCases:
    """Test edge cases in the DimensionalBase API."""

    def test_delete_nonexistent_returns_false(self, db):
        assert db.delete("nonexistent/path") is False

    def test_exists_returns_false_for_missing(self, db):
        assert db.exists("nonexistent") is False

    def test_retrieve_returns_none_for_missing(self, db):
        assert db.retrieve("nonexistent") is None

    def test_get_empty_db(self, db):
        result = db.get("**", budget=500)
        assert len(result.entries) == 0
        assert result.total_matched == 0

    def test_get_with_zero_budget(self, db):
        db.put("x", "value", owner="a")
        result = db.get("**", budget=0)
        assert len(result.entries) == 0

    def test_clear_turn_on_empty_db(self, db):
        assert db.clear_turn() == 0

    def test_clear_session_on_empty_db(self, db):
        assert db.clear_session() == 0

    def test_status_on_empty_db(self, db):
        status = db.status()
        assert status["entries"] == 0
        assert status["total_puts"] == 0

    def test_context_manager(self):
        with DimensionalBase() as db:
            db.put("x", "value", owner="a")
            assert db.entry_count == 1
        # After exit, DB should be closed

    def test_multiple_puts_same_path(self, db):
        """Multiple puts to same path should update, not duplicate."""
        db.put("x", "v1", owner="a")
        db.put("x", "v2", owner="a")
        assert db.entry_count == 1
        entry = db.retrieve("x")
        assert entry.value == "v2"
        assert entry.version >= 2
