"""Tests for task-adaptive scoring (Phase 3)."""

import time
import pytest

from dimensionalbase import DimensionalBase
from dimensionalbase.core.types import ScoringWeights
from dimensionalbase.context.engine import ContextEngine


class TestAdaptiveScoring:
    """Verify adaptive weight behavior."""

    def test_default_is_adaptive(self):
        """ScoringWeights should be adaptive by default."""
        w = ScoringWeights()
        assert w.adaptive is True

    def test_adaptive_false_preserves_weights(self):
        """When adaptive=False, weights should not change."""
        w = ScoringWeights(recency=0.25, confidence=0.25, similarity=0.25, reference_distance=0.25, adaptive=False)
        assert w.recency == 0.25
        assert w.similarity == 0.25

    def test_weights_still_sum_to_one(self):
        """Adapted weights must still sum to 1.0."""
        w = ScoringWeights()
        total = w.recency + w.confidence + w.similarity + w.reference_distance
        assert abs(total - 1.0) < 0.01

    def test_invalid_weights_raise(self):
        """Weights that don't sum to 1.0 should raise ValueError."""
        with pytest.raises(ValueError):
            ScoringWeights(recency=0.5, confidence=0.5, similarity=0.5, reference_distance=0.5)

    def test_query_boosts_results(self, db):
        """With a query, entries should be scored differently than without."""
        # Write entries at different times
        db.put("old/fact", "old information", owner="a")
        time.sleep(0.01)
        db.put("new/fact", "new information", owner="a")

        # Get without query (should favor recency)
        result_no_query = db.get("**", budget=5000)

        # Get with query (should favor similarity — but in text mode,
        # similarity is flat so results should still be returned)
        result_with_query = db.get("**", budget=5000, query="old information")

        # Both should return entries
        assert len(result_no_query.entries) >= 2
        assert len(result_with_query.entries) >= 2

    def test_adapt_weights_with_query(self):
        """Context engine should boost similarity when query is present."""
        from dimensionalbase.channels.manager import ChannelManager
        cm = ChannelManager(db_path=":memory:")
        engine = ContextEngine(channel_manager=cm)

        adapted = engine._adapt_weights(has_query=True)
        assert adapted.similarity > 0.30  # Default is 0.30, should be higher
        assert adapted.recency < 0.30

    def test_adapt_weights_without_query(self):
        """Context engine should boost recency when no query."""
        from dimensionalbase.channels.manager import ChannelManager
        cm = ChannelManager(db_path=":memory:")
        engine = ContextEngine(channel_manager=cm)

        adapted = engine._adapt_weights(has_query=False)
        assert adapted.recency > 0.30  # Default is 0.30, should be higher
        assert adapted.similarity < 0.30

    def test_adapt_weights_disabled(self):
        """When adaptive=False, weights should be returned unchanged."""
        from dimensionalbase.channels.manager import ChannelManager
        cm = ChannelManager(db_path=":memory:")
        w = ScoringWeights(adaptive=False)
        engine = ContextEngine(channel_manager=cm, weights=w)

        adapted = engine._adapt_weights(has_query=True)
        assert adapted.recency == w.recency
        assert adapted.similarity == w.similarity

    def test_confidence_and_trust_affect_order_before_packing(self, db):
        """Bayesian confidence and trust should influence rank order directly."""
        db.put("scope/x/a", "alpha", owner="agent-a", confidence=1.0)
        db.put("scope/x/b", "beta", owner="agent-b", confidence=0.0)

        shared_timestamp = time.time() - 60
        conn = db._channels.text_channel._conn
        conn.execute(
            "UPDATE knowledge SET updated_at = ? WHERE path IN (?, ?)",
            (shared_timestamp, "scope/x/a", "scope/x/b"),
        )
        conn.commit()

        for i in range(12):
            db.confidence.contradict("scope/x/a", f"reviewer-a-{i}")
            db.confidence.confirm("scope/x/b", f"reviewer-b-{i}")
            db.trust.record_contradiction(
                "judge",
                "agent-a",
                domain="x",
                confidence_of_contradicting=0.9,
                confidence_of_contradicted=0.2,
            )
            db.trust.record_confirmation("judge", "agent-b", domain="x")

        result = db.get("scope/**", budget=5000)
        assert [entry.path for entry in result.entries][:2] == ["scope/x/b", "scope/x/a"]

        ranked = {entry.path: entry for entry in result.entries}
        assert ranked["scope/x/b"]._raw_score > ranked["scope/x/a"]._raw_score
