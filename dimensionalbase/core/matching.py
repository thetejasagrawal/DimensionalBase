"""
Canonical path matching for the DimensionalBase Protocol.

This is the SINGLE source of truth for glob pattern matching across
the entire system.  Every component (TextChannel, EventBus, ACL) MUST
use ``dbps_match`` and ``dbps_pattern_covers`` — no local re-implementations.

Specification reference: DBPS v1.0 Section 5 — Path Addressing and Glob Semantics.

Grammar (ABNF):
    path         = segment *( "/" segment )
    segment      = 1*( ALPHA / DIGIT / "-" / "_" / "." )
    pattern      = pattern-seg *( "/" pattern-seg )
    pattern-seg  = segment / "*" / "**"

Matching rules:
    "*"   matches any single path segment (no "/" crossing)
    "**"  matches zero or more path segments (crosses "/" boundaries)
    Exact characters match literally (case-sensitive)

Examples:
    pattern="task/**"    path="task"              → True  (** matches zero segments)
    pattern="task/**"    path="task/auth"          → True
    pattern="task/**"    path="task/auth/jwt"      → True
    pattern="task/*"     path="task/auth"          → True
    pattern="task/*"     path="task/auth/jwt"      → False (* does not cross /)
    pattern="**"         path="anything/at/all"    → True
    pattern="task/auth"  path="task/auth"          → True
    pattern="task/auth"  path="task/auth/jwt"      → False (no wildcard)
"""

from __future__ import annotations


def dbps_match(pattern: str, path: str) -> bool:
    """Test whether *path* matches *pattern* according to DBPS glob rules.

    This is the canonical matching algorithm.  All DimensionalBase components
    MUST use this function (or an equivalent that produces identical results
    for every input in the conformance test vectors).

    Args:
        pattern: A DBPS glob pattern (may contain ``*`` and ``**``).
        path:    A concrete path (no wildcards).

    Returns:
        True if *path* matches *pattern*.
    """
    # Fast paths
    if pattern == "**":
        return True
    if pattern == path:
        return True
    if "*" not in pattern:
        return pattern == path

    pat_parts = pattern.split("/")
    path_parts = path.split("/")
    return _match_parts(pat_parts, 0, path_parts, 0)


def _match_parts(
    pat: list, pi: int,
    path: list, qi: int,
) -> bool:
    """Recursive glob matching engine.

    Handles ``**`` (matches zero or more segments) and ``*`` (matches exactly
    one segment) with backtracking.
    """
    while pi < len(pat) and qi < len(path):
        token = pat[pi]

        if token == "**":
            # ** at end of pattern matches everything remaining
            if pi == len(pat) - 1:
                return True
            # Try matching ** against 0, 1, 2, … remaining path segments
            for skip in range(qi, len(path) + 1):
                if _match_parts(pat, pi + 1, path, skip):
                    return True
            return False

        elif token == "*":
            # * matches exactly one segment (any content)
            pi += 1
            qi += 1

        elif token == path[qi]:
            pi += 1
            qi += 1

        else:
            return False

    # Consume trailing ** patterns (they can match zero segments)
    while pi < len(pat) and pat[pi] == "**":
        pi += 1

    return pi == len(pat) and qi == len(path)


def dbps_pattern_covers(allowed: str, requested: str) -> bool:
    """Test whether an *allowed* pattern is at least as broad as *requested*.

    Used by ACL to check if an agent's allowed scope covers the requested scope.

    A pattern A covers pattern B if every path matched by B is also matched by A.

    For common cases:
        "**"            covers everything
        "task/**"       covers "task/auth/**"   (prefix containment)
        "task/*"        does NOT cover "task/**" (* is narrower than **)

    Args:
        allowed:   The agent's allowed pattern.
        requested: The scope the agent is trying to access.

    Returns:
        True if *allowed* is guaranteed to cover *requested*.
    """
    if allowed == "**":
        return True
    if allowed == requested:
        return True

    # For ** patterns: check prefix containment
    # "task/**" covers "task/auth/**" because task/ is a prefix of task/auth/
    allowed_parts = allowed.split("/")
    requested_parts = requested.split("/")

    ai = 0
    ri = 0
    while ai < len(allowed_parts) and ri < len(requested_parts):
        a = allowed_parts[ai]
        r = requested_parts[ri]

        if a == "**":
            # ** in allowed at this position covers everything from here
            return True
        elif a == "*":
            # * covers exactly one segment — only if requested also has
            # exactly one segment here (not **)
            if r == "**":
                return False
            ai += 1
            ri += 1
        elif a == r:
            ai += 1
            ri += 1
        else:
            return False

    # Consume trailing **
    while ai < len(allowed_parts) and allowed_parts[ai] == "**":
        ai += 1

    return ai >= len(allowed_parts)
