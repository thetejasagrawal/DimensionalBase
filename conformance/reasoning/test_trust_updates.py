"""
DBPS v1.0 Section 12 — Trust Model Conformance Tests.

Verifies Elo-PageRank trust model behavior.
"""

from __future__ import annotations

import pytest

from dimensionalbase.trust.agent_trust import AgentTrustEngine


@pytest.fixture
def engine():
    return AgentTrustEngine()


class TestTrustModelConformance:
    """Verify trust model semantics from DBPS spec."""

    def test_new_agent_default_trust(self, engine):
        """New agent MUST start with trust approximately 0.5."""
        engine.record_entry("agent-new")
        trust = engine.get_trust("agent-new")
        assert 0.4 <= trust <= 0.6, f"New agent trust should be ~0.5, got {trust}"

    def test_confirmation_increases_confirmed_trust(self, engine):
        """Confirmation MUST increase the confirmed agent's trust."""
        engine.record_entry("a")
        engine.record_entry("b")
        before = engine.get_trust("a")
        engine.record_confirmation("b", "a", "task")
        after = engine.get_trust("a")
        assert after >= before, f"Confirmation decreased trust: {before} -> {after}"

    def test_trust_bounded_zero_one(self, engine):
        """Trust MUST always be in [0.0, 1.0]."""
        engine.record_entry("a")
        engine.record_entry("b")
        for _ in range(100):
            engine.record_confirmation("b", "a", "task")
        assert 0.0 <= engine.get_trust("a") <= 1.0

        engine.record_entry("c")
        for _ in range(100):
            engine.record_contradiction("b", "c", "task", 0.5, 0.5)
        assert 0.0 <= engine.get_trust("c") <= 1.0

    def test_pagerank_produces_valid_scores(self, engine):
        """PageRank trust MUST produce non-negative values for all agents."""
        for agent in ["a", "b", "c", "d"]:
            engine.record_entry(agent)
        engine.record_confirmation("b", "a", "task")
        engine.record_confirmation("c", "a", "task")
        engine.record_confirmation("d", "b", "task")

        pr = engine.compute_pagerank_trust()
        assert all(v >= 0 for v in pr.values()), f"PageRank has negative values: {pr}"
        # Agent "a" was confirmed by both "b" and "c", should have highest PageRank
        assert pr.get("a", 0) >= pr.get("c", 0), \
            f"Most-confirmed agent should have highest PageRank: a={pr.get('a')}, c={pr.get('c')}"

    def test_domain_specific_trust(self, engine):
        """Domain-specific trust MUST be independent across domains."""
        engine.record_entry("a")
        engine.record_entry("b")
        # Build trust in 'code' domain
        for _ in range(5):
            engine.record_confirmation("b", "a", "code")

        trust_code = engine.get_trust("a", domain="code")
        trust_business = engine.get_trust("a", domain="business")
        # Code trust should be higher than business trust
        assert trust_code >= trust_business, \
            f"Domain trust not independent: code={trust_code}, business={trust_business}"

    def test_unknown_agent_default(self, engine):
        """Querying trust for unknown agent MUST return default (0.5)."""
        trust = engine.get_trust("nonexistent")
        assert 0.4 <= trust <= 0.6

    def test_agent_count(self, engine):
        """agent_count MUST reflect registered agents."""
        assert engine.agent_count == 0
        engine.record_entry("a")
        assert engine.agent_count == 1
        engine.record_entry("b")
        assert engine.agent_count == 2
