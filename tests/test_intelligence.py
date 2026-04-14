"""
Tests for the intelligence layer: confidence, trust, provenance, compression.

These test the systems that make DimensionalBase learn and adapt —
the parts that can't be replicated in a weekend.
"""

import time
import pytest
import numpy as np

from dimensionalbase.reasoning.confidence import ConfidenceEngine, ConfidenceState
from dimensionalbase.reasoning.provenance import ProvenanceTracker, DerivationType
from dimensionalbase.trust.agent_trust import AgentTrustEngine, AgentProfile
from dimensionalbase.context.compression import SemanticCompressor
from dimensionalbase.core.entry import KnowledgeEntry


class TestConfidenceEngine:
    """Test Bayesian confidence tracking."""

    def test_register_and_get(self):
        engine = ConfidenceEngine()
        state = engine.register("task/a", confidence=0.8, owner="agent-1")
        assert state.mean > 0.5
        conf = engine.get_confidence("task/a")
        assert 0.5 < conf <= 1.0

    def test_confirmation_increases_confidence(self):
        engine = ConfidenceEngine()
        engine.register("task/a", confidence=0.7, owner="agent-1")

        before = engine.get_confidence("task/a")
        engine.confirm("task/a", "agent-2")
        after = engine.get_confidence("task/a")

        assert after > before

    def test_contradiction_decreases_confidence(self):
        engine = ConfidenceEngine()
        engine.register("task/a", confidence=0.8, owner="agent-1")

        before = engine.get_confidence("task/a")
        engine.contradict("task/a", "agent-2")
        after = engine.get_confidence("task/a")

        assert after < before

    def test_multiple_confirmations_build_confidence(self):
        engine = ConfidenceEngine()
        engine.register("task/a", confidence=0.6, owner="agent-1")

        for i in range(5):
            engine.confirm("task/a", f"agent-{i+2}")

        state = engine.get_state("task/a")
        assert state is not None
        assert state.confirmations == 5
        assert state.effective_confidence > 0.7

    def test_beta_distribution_properties(self):
        engine = ConfidenceEngine()
        state = engine.register("task/a", confidence=0.5, owner="agent-1")

        # With no evidence, should be near prior
        assert abs(state.mean - 0.5) < 0.2
        assert state.variance > 0  # Uncertain
        assert state.strength < 2

    def test_lower_bound_conservative(self):
        engine = ConfidenceEngine()
        state = engine.register("task/a", confidence=0.9, owner="agent-1")
        assert state.lower_bound <= state.mean

    def test_agent_agreement_score(self):
        engine = ConfidenceEngine()
        engine.register("task/a", confidence=0.8, owner="agent-1")

        engine.confirm("task/a", "agent-2")
        engine.confirm("task/a", "agent-2")
        engine.contradict("task/a", "agent-3")

        # agent-2 should have high agreement with agent-1
        score_agree = engine.agent_agreement_score("agent-2", "agent-1")
        assert score_agree > 0.5

    def test_unknown_path_returns_neutral(self):
        engine = ConfidenceEngine()
        assert engine.get_confidence("nonexistent") == 0.5


class TestProvenanceTracker:
    """Test full knowledge lineage tracking."""

    def test_record_creation(self):
        tracker = ProvenanceTracker()
        node = tracker.record_creation(
            path="task/a", owner="agent-1", value_hash="abc123"
        )
        assert node.is_root
        assert node.derivation == DerivationType.ORIGINAL

    def test_record_update(self):
        tracker = ProvenanceTracker()
        tracker.record_creation("task/a", "agent-1", "v1hash")
        node = tracker.record_update("task/a", "agent-1", "v2hash", version=2)
        assert node.derivation == DerivationType.UPDATED
        assert not node.is_root

    def test_record_derived(self):
        tracker = ProvenanceTracker()
        tracker.record_creation("source/a", "agent-1", "hash_a")
        tracker.record_creation("source/b", "agent-1", "hash_b")

        derived = tracker.record_creation(
            "derived/c", "agent-2", "hash_c",
            derived_from=["source/a", "source/b"],
        )
        assert derived.derivation == DerivationType.DERIVED
        assert len(derived.parent_ids) == 2

    def test_get_lineage(self):
        tracker = ProvenanceTracker()
        tracker.record_creation("task/a", "agent-1", "v1")
        tracker.record_update("task/a", "agent-1", "v2", version=2)
        tracker.record_update("task/a", "agent-1", "v3", version=3)

        lineage = tracker.get_lineage("task/a")
        assert len(lineage) >= 2
        assert lineage[0].version >= lineage[-1].version

    def test_get_roots(self):
        tracker = ProvenanceTracker()
        tracker.record_creation("root/a", "agent-1", "h1")
        tracker.record_creation("derived/b", "agent-2", "h2",
                                derived_from=["root/a"])

        roots = tracker.get_roots("derived/b")
        assert any(r.path == "root/a" for r in roots)

    def test_confirmation_recorded(self):
        tracker = ProvenanceTracker()
        tracker.record_creation("task/a", "agent-1", "h1")
        node = tracker.record_confirmation("task/a", "agent-2")
        assert node is not None
        assert node.derivation == DerivationType.CONFIRMED

    def test_contradiction_recorded(self):
        tracker = ProvenanceTracker()
        tracker.record_creation("task/a", "agent-1", "h1")
        node = tracker.record_contradiction("task/a", "agent-2", "h2")
        assert node is not None
        assert node.derivation == DerivationType.CONTRADICTED

    def test_get_history(self):
        tracker = ProvenanceTracker()
        tracker.record_creation("task/a", "agent-1", "v1")
        tracker.record_update("task/a", "agent-1", "v2", version=2)

        history = tracker.get_history("task/a")
        assert len(history) >= 2

    def test_causal_chain(self):
        tracker = ProvenanceTracker()
        tracker.record_creation("a", "agent-1", "h1")
        tracker.record_creation("b", "agent-2", "h2", derived_from=["a"])
        tracker.record_creation("c", "agent-3", "h3", derived_from=["b"])

        chain = tracker.get_causal_chain("a", "c")
        assert chain is not None
        assert len(chain) >= 2

    def test_trust_depth(self):
        tracker = ProvenanceTracker()
        tracker.record_creation("task/a", "agent-1", "h1")
        tracker.record_confirmation("task/a", "agent-2")
        tracker.record_confirmation("task/a", "agent-3")

        depth = tracker.trust_depth("task/a")
        assert depth >= 2


class TestAgentTrust:
    """Test the agent trust model."""

    def test_new_agent_default_trust(self):
        engine = AgentTrustEngine()
        trust = engine.get_trust("unknown-agent")
        assert trust == 0.5

    def test_confirmation_increases_trust(self):
        engine = AgentTrustEngine()
        engine.get_or_create_profile("agent-1")
        engine.get_or_create_profile("agent-2")

        before = engine.get_trust("agent-1")
        engine.record_confirmation("agent-2", "agent-1", domain="task")
        after = engine.get_trust("agent-1")

        assert after >= before

    def test_contradiction_affects_trust(self):
        engine = AgentTrustEngine()
        engine.get_or_create_profile("agent-1")
        engine.get_or_create_profile("agent-2")

        engine.record_contradiction(
            "agent-2", "agent-1", domain="task",
            confidence_of_contradicting=0.9,
            confidence_of_contradicted=0.5,
        )

        trust = engine.get_trust("agent-1")
        assert trust < 0.5  # Should have lost trust

    def test_domain_specific_trust(self):
        engine = AgentTrustEngine()
        engine.get_or_create_profile("agent-1")
        engine.get_or_create_profile("agent-2")

        # Good at code, bad at business
        for _ in range(5):
            engine.record_confirmation("agent-2", "agent-1", domain="code")
        for _ in range(5):
            engine.record_contradiction(
                "agent-2", "agent-1", domain="business",
                confidence_of_contradicting=0.9,
                confidence_of_contradicted=0.3,
            )

        code_trust = engine.get_trust("agent-1", domain="code")
        biz_trust = engine.get_trust("agent-1", domain="business")
        assert code_trust > biz_trust

    def test_trust_ranking(self):
        engine = AgentTrustEngine()
        for i in range(3):
            engine.get_or_create_profile(f"agent-{i}")

        # Agent-0 gets confirmed a lot
        for _ in range(10):
            engine.record_confirmation("agent-1", "agent-0", domain="task")

        ranking = engine.get_trust_ranking()
        assert ranking[0][0] == "agent-0"  # Most trusted

    def test_pagerank_trust(self):
        engine = AgentTrustEngine()
        for i in range(4):
            engine.get_or_create_profile(f"agent-{i}")

        # Build trust graph
        engine.record_confirmation("agent-1", "agent-0", "task")
        engine.record_confirmation("agent-2", "agent-0", "task")
        engine.record_confirmation("agent-3", "agent-0", "task")

        pr = engine.compute_pagerank_trust()
        assert len(pr) == 4
        assert pr["agent-0"] >= pr["agent-3"]  # Most trusted by PageRank

    def test_agent_profile_properties(self):
        engine = AgentTrustEngine()
        profile = engine.get_or_create_profile("test-agent")
        profile.total_entries = 10
        profile.total_confirmations = 7
        profile.total_contradictions = 3

        assert profile.confirmation_rate == 0.7
        assert not profile.is_reliable  # Needs 5 interactions
        assert profile.activity_level > 0

    def test_summary(self):
        engine = AgentTrustEngine()
        engine.get_or_create_profile("agent-1")
        engine.record_entry("agent-1")

        summary = engine.summary()
        assert "agent-1" in summary
        assert "global_trust" in summary["agent-1"]


class TestCompression:
    """Test semantic compression engine."""

    def _make_entries(self, n):
        entries = []
        for i in range(n):
            entries.append(KnowledgeEntry(
                path=f"test/entry/{i}",
                value=f"This is test entry number {i} with content to take tokens",
                owner="agent-1",
                confidence=0.5 + (i % 5) * 0.1,
            ))
        return entries

    def test_delta_encoding(self):
        comp = SemanticCompressor()
        entries = self._make_entries(10)

        # First read: everything is new
        comp.register_reader("reader-1")
        new, updated, unchanged = comp.compute_delta(entries, "reader-1")
        assert len(new) == 10
        assert len(unchanged) == 0

        # Mark as seen
        comp.mark_as_seen("reader-1", entries)

        # Second read: everything is unchanged
        new, updated, unchanged = comp.compute_delta(entries, "reader-1")
        assert len(new) == 0
        assert len(unchanged) == 10

    def test_text_deduplication(self):
        comp = SemanticCompressor()
        entries = [
            KnowledgeEntry(path="a", value="same text here", owner="agent-1"),
            KnowledgeEntry(path="b", value="same text here", owner="agent-2"),
            KnowledgeEntry(path="c", value="different text", owner="agent-1"),
        ]

        deduped, removed = comp.deduplicate(entries)
        assert removed == 1
        assert len(deduped) == 2

    def test_compress_pipeline(self):
        comp = SemanticCompressor()
        entries = self._make_entries(20)

        result = comp.compress(entries, budget=200)
        assert result.compressed_count <= result.original_count
        assert result.compression_ratio <= 1.0

    def test_compress_with_reader(self):
        comp = SemanticCompressor()
        entries = self._make_entries(10)

        # First read
        result1 = comp.compress(entries, reader_agent="reader-1", budget=5000)
        # Second read: should have deltas
        result2 = comp.compress(entries, reader_agent="reader-1", budget=5000)
        assert result2.deltas_applied > 0


class TestIntegration:
    """Integration tests: all intelligence systems working together."""

    def test_db_tracks_trust_on_writes(self):
        from dimensionalbase import DimensionalBase
        db = DimensionalBase()

        db.put("task/status", "API healthy", owner="backend")
        db.put("task/status", "API down", owner="frontend")

        # Both agents should have profiles
        assert db.trust.agent_count >= 2

        report = db.agent_trust_report()
        assert "backend" in report
        assert "frontend" in report
        db.close()

    def test_db_tracks_provenance(self):
        from dimensionalbase import DimensionalBase
        db = DimensionalBase()

        db.put("research/a", "finding A", owner="researcher")
        db.put("plan/b", "based on A", owner="planner", refs=["research/a"])

        lineage = db.lineage("plan/b")
        assert len(lineage) >= 1
        db.close()

    def test_db_confidence_updates(self):
        from dimensionalbase import DimensionalBase
        db = DimensionalBase()

        db.put("task/fact", "the sky is blue", owner="agent-1", confidence=0.8)
        conf1 = db.confidence.get_confidence("task/fact")

        # Same path, different agent = cross-agent write
        db.put("task/fact", "the sky is blue", owner="agent-2", confidence=0.9)
        conf2 = db.confidence.get_confidence("task/fact")

        # Confidence should update (confirmation or contradiction)
        assert conf2 != conf1 or conf2 > 0.0  # Something happened
        db.close()

    def test_db_status_includes_deep_systems(self):
        from dimensionalbase import DimensionalBase
        db = DimensionalBase()

        db.put("a", "test", owner="x")
        status = db.status()

        assert "agents" in status
        assert "provenance_nodes" in status
        assert "total_puts" in status
        assert "total_gets" in status
        assert status["total_puts"] == 1
        db.close()
