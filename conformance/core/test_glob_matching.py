"""
DBPS v1.0 Section 5 — Glob Matching Conformance Tests.

These tests load the canonical test vectors from spec/test-vectors/glob-matching.json
and verify that the implementation produces identical results.

ANY conformant DBPS implementation MUST pass all of these tests.
"""

from __future__ import annotations

import json
import os
import pytest

from dimensionalbase.core.matching import dbps_match

# Load canonical test vectors
_VECTORS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "spec", "test-vectors", "glob-matching.json"
)


def _load_vectors():
    with open(_VECTORS_PATH) as f:
        data = json.load(f)
    return data["tests"]


_VECTORS = _load_vectors()


@pytest.mark.parametrize(
    "test_case",
    _VECTORS,
    ids=[v["id"] for v in _VECTORS],
)
def test_glob_matching_conformance(test_case):
    """Verify canonical glob matching behavior from DBPS spec test vectors."""
    pattern = test_case["pattern"]
    path = test_case["path"]
    expected = test_case["expected"]

    result = dbps_match(pattern, path)
    assert result == expected, (
        f"CONFORMANCE FAILURE: dbps_match({pattern!r}, {path!r}) "
        f"returned {result}, expected {expected}"
    )


class TestGlobMatchingProperties:
    """Property-based tests for glob matching (beyond test vectors)."""

    def test_exact_match_is_reflexive(self):
        """Every path matches itself."""
        paths = ["task", "task/auth", "a/b/c/d", "x"]
        for p in paths:
            assert dbps_match(p, p) is True

    def test_doublestar_matches_everything(self):
        """** matches any path."""
        paths = ["a", "a/b", "a/b/c", "x/y/z/w/v"]
        for p in paths:
            assert dbps_match("**", p) is True

    def test_star_does_not_cross_slash(self):
        """* matches only within one segment."""
        assert dbps_match("a/*", "a/b") is True
        assert dbps_match("a/*", "a/b/c") is False

    def test_doublestar_matches_zero_segments(self):
        """** can match zero segments."""
        assert dbps_match("a/**", "a") is True
        assert dbps_match("a/**/b", "a/b") is True

    def test_no_wildcard_requires_exact(self):
        """Without wildcards, only exact match succeeds."""
        assert dbps_match("a/b", "a/b") is True
        assert dbps_match("a/b", "a/c") is False
        assert dbps_match("a/b", "a/b/c") is False

    def test_empty_pattern_matches_empty_path(self):
        assert dbps_match("", "") is True

    def test_middle_doublestar(self):
        """** in the middle of a pattern works."""
        assert dbps_match("a/**/z", "a/z") is True
        assert dbps_match("a/**/z", "a/b/z") is True
        assert dbps_match("a/**/z", "a/b/c/z") is True
        assert dbps_match("a/**/z", "a/b/c/d") is False
