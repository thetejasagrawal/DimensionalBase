"""
DBPS v1.0 Section 4 — KnowledgeEntry Serialization Conformance Tests.

Verifies that entries roundtrip through to_dict()/from_dict() without data loss.
"""

from __future__ import annotations

import pytest

from dimensionalbase.core.entry import KnowledgeEntry
from dimensionalbase.core.types import EntryType, TTL


class TestEntrySerialization:
    """Verify KnowledgeEntry serialization is lossless."""

    def test_roundtrip_basic(self):
        """Basic entry survives to_dict/from_dict roundtrip."""
        entry = KnowledgeEntry(path="task/auth", value="JWT expired", owner="agent-a")
        d = entry.to_dict()
        restored = KnowledgeEntry.from_dict(d)

        assert restored.path == entry.path
        assert restored.value == entry.value
        assert restored.owner == entry.owner
        assert restored.type == entry.type
        assert restored.confidence == entry.confidence
        assert restored.version == entry.version
        assert restored.ttl == entry.ttl

    def test_roundtrip_full(self):
        """Entry with all fields populated survives roundtrip."""
        entry = KnowledgeEntry(
            path="task/deploy/v2",
            value="Deploying version 2.1.0",
            owner="deploy-agent",
            type=EntryType.PLAN,
            confidence=0.85,
            refs=["task/auth", "task/staging"],
            version=3,
            ttl=TTL.PERSISTENT,
            metadata={"priority": "high", "environment": "production"},
        )
        d = entry.to_dict()
        restored = KnowledgeEntry.from_dict(d)

        assert restored.path == entry.path
        assert restored.value == entry.value
        assert restored.owner == entry.owner
        assert restored.type == EntryType.PLAN
        assert restored.confidence == 0.85
        assert restored.refs == ["task/auth", "task/staging"]
        assert restored.version == 3
        assert restored.ttl == TTL.PERSISTENT

    def test_all_entry_types(self):
        """Every EntryType value roundtrips correctly."""
        for et in EntryType:
            entry = KnowledgeEntry(path="x", value="v", owner="a", type=et)
            d = entry.to_dict()
            restored = KnowledgeEntry.from_dict(d)
            assert restored.type == et

    def test_all_ttl_values(self):
        """Every TTL value roundtrips correctly."""
        for ttl in TTL:
            entry = KnowledgeEntry(path="x", value="v", owner="a", ttl=ttl)
            d = entry.to_dict()
            restored = KnowledgeEntry.from_dict(d)
            assert restored.ttl == ttl

    def test_empty_refs(self):
        """Empty refs roundtrip as empty list."""
        entry = KnowledgeEntry(path="x", value="v", owner="a", refs=[])
        d = entry.to_dict()
        restored = KnowledgeEntry.from_dict(d)
        assert restored.refs == []

    def test_metadata_roundtrip(self):
        """Metadata keys and values survive roundtrip."""
        entry = KnowledgeEntry(
            path="x", value="v", owner="a",
            metadata={"key1": "val1", "key2": "val2", "key3": "val3"},
        )
        d = entry.to_dict()
        restored = KnowledgeEntry.from_dict(d)
        assert restored.metadata.get("key1") == "val1"
        assert restored.metadata.get("key2") == "val2"
        assert restored.metadata.get("key3") == "val3"

    def test_confidence_boundaries(self):
        """Confidence at 0.0 and 1.0 roundtrip correctly."""
        for conf in [0.0, 1.0, 0.5, 0.001, 0.999]:
            entry = KnowledgeEntry(path="x", value="v", owner="a", confidence=conf)
            d = entry.to_dict()
            restored = KnowledgeEntry.from_dict(d)
            assert abs(restored.confidence - conf) < 1e-10
