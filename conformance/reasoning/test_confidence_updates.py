"""
DBPS v1.0 Section 11 — Confidence Model Conformance Tests.

Verifies that the Bayesian confidence model produces the exact behavior
specified in the protocol.
"""

from __future__ import annotations

import json
import os
import pytest

from dimensionalbase.reasoning.confidence import ConfidenceEngine


@pytest.fixture
def engine():
    return ConfidenceEngine()


class TestConfidenceModelConformance:
    """Verify Bayesian confidence semantics from DBPS spec."""

    def test_initial_registration(self, engine):
        """Registered confidence MUST reflect the self-reported value."""
        engine.register("x", 0.8, "agent-a")
        conf = engine.get_confidence("x")
        assert 0.5 <= conf <= 1.0

    def test_unknown_path_returns_neutral(self, engine):
        """Unregistered path MUST return neutral confidence (0.5)."""
        assert engine.get_confidence("nonexistent") == 0.5

    def test_confirmation_increases_confidence(self, engine):
        """confirm() MUST increase confidence."""
        engine.register("x", 0.5, "agent-a")
        before = engine.get_confidence("x")
        engine.confirm("x", "agent-b")
        after = engine.get_confidence("x")
        assert after > before, f"Confirmation did not increase confidence: {before} -> {after}"

    def test_contradiction_decreases_confidence(self, engine):
        """contradict() MUST decrease confidence."""
        engine.register("x", 0.8, "agent-a")
        before = engine.get_confidence("x")
        engine.contradict("x", "agent-b")
        after = engine.get_confidence("x")
        assert after < before, f"Contradiction did not decrease confidence: {before} -> {after}"

    def test_multiple_confirmations_build_confidence(self, engine):
        """Repeated confirmations MUST monotonically increase confidence."""
        engine.register("x", 0.5, "agent-a")
        prev = engine.get_confidence("x")
        for i in range(5):
            engine.confirm("x", f"agent-{i}")
            curr = engine.get_confidence("x")
            assert curr >= prev, f"Confirmation {i} decreased confidence: {prev} -> {curr}"
            prev = curr
        assert prev > 0.6, f"5 confirmations from 0.5 should yield > 0.6, got {prev}"

    def test_confirmation_then_contradiction_sequence(self, engine):
        """Mixed sequence: 3 confirms then 1 contradict should still be above 0.5."""
        engine.register("x", 0.5, "agent-a")
        engine.confirm("x", "b")
        engine.confirm("x", "c")
        engine.confirm("x", "d")
        after_confirms = engine.get_confidence("x")
        engine.contradict("x", "e")
        after_contradict = engine.get_confidence("x")
        assert after_contradict < after_confirms
        assert after_contradict > 0.5, "3 confirms vs 1 contradict should still be > 0.5"

    def test_confidence_bounded_zero_one(self, engine):
        """Confidence MUST always be in [0.0, 1.0]."""
        engine.register("x", 0.9, "a")
        for i in range(100):
            engine.confirm("x", f"confirmer-{i}")
        assert 0.0 <= engine.get_confidence("x") <= 1.0

        engine.register("y", 0.1, "a")
        for i in range(100):
            engine.contradict("y", f"contradicter-{i}")
        assert 0.0 <= engine.get_confidence("y") <= 1.0


class TestConfidenceFromTestVectors:
    """Load canonical test vectors and verify."""

    _VECTORS_PATH = os.path.join(
        os.path.dirname(__file__), "..", "..", "spec", "test-vectors", "confidence.json"
    )

    def _load_vectors(self):
        with open(self._VECTORS_PATH) as f:
            return json.load(f)

    def test_all_vector_cases(self):
        """Every test vector MUST produce expected results."""
        data = self._load_vectors()
        for test in data["tests"]:
            engine = ConfidenceEngine()
            for op in test["operations"]:
                if op["op"] == "register":
                    engine.register(op["path"], op["confidence"], op["owner"])
                elif op["op"] == "confirm":
                    engine.confirm(op["path"], op["confirmer"])
                elif op["op"] == "contradict":
                    engine.contradict(op["path"], op["contradicter"])

            query_path = test.get("query_path", test["operations"][0]["path"] if test["operations"] else "x")
            conf = engine.get_confidence(query_path)

            if "expected_confidence" in test:
                assert abs(conf - test["expected_confidence"]) < 0.1, \
                    f"Vector {test['id']}: expected {test['expected_confidence']}, got {conf}"
            if "expected_confidence_gte" in test:
                assert conf >= test["expected_confidence_gte"], \
                    f"Vector {test['id']}: expected >= {test['expected_confidence_gte']}, got {conf}"
            if "expected_confidence_lte" in test:
                assert conf <= test["expected_confidence_lte"], \
                    f"Vector {test['id']}: expected <= {test['expected_confidence_lte']}, got {conf}"
