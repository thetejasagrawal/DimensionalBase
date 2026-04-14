"""
Path-based access control for DimensionalBase.

Each agent has an ``AgentPolicy`` specifying which paths it can read and write.
Patterns use fnmatch glob syntax (consistent with the rest of DimensionalBase).
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dimensionalbase.exceptions import DimensionalBaseError


class AccessDeniedError(DimensionalBaseError):
    """Valid key but operation not allowed on this path."""


@dataclass
class AgentPolicy:
    """Access control policy for a single agent."""
    agent_id: str
    allowed_read_patterns: List[str] = field(default_factory=lambda: ["**"])
    allowed_write_patterns: List[str] = field(default_factory=lambda: ["**"])
    is_admin: bool = False


class AccessController:
    """Enforces path-based access control using AgentPolicy.

    Admin agents bypass all checks.
    """

    def __init__(self) -> None:
        self._policies: Dict[str, AgentPolicy] = {}

    def register_policy(self, policy: AgentPolicy) -> None:
        """Register or update an agent's access policy."""
        self._policies[policy.agent_id] = policy

    def get_policy(self, agent_id: str) -> Optional[AgentPolicy]:
        """Get an agent's policy, or None if not registered."""
        return self._policies.get(agent_id)

    def check_read(self, agent_id: str, scope: str) -> None:
        """Check if an agent can read the given scope. Raises AccessDeniedError."""
        policy = self._policies.get(agent_id)
        if policy is None:
            # No policy = no restrictions (for backward compatibility)
            return
        if policy.is_admin:
            return
        for pattern in policy.allowed_read_patterns:
            if self._pattern_covers(pattern, scope):
                return
        raise AccessDeniedError(
            f"Agent '{agent_id}' is not allowed to read scope '{scope}'"
        )

    def check_write(self, agent_id: str, path: str) -> None:
        """Check if an agent can write to the given path. Raises AccessDeniedError."""
        policy = self._policies.get(agent_id)
        if policy is None:
            return
        if policy.is_admin:
            return
        for pattern in policy.allowed_write_patterns:
            if self._match(pattern, path):
                return
        raise AccessDeniedError(
            f"Agent '{agent_id}' is not allowed to write to path '{path}'"
        )

    @staticmethod
    def _match(pattern: str, path: str) -> bool:
        """Match a path against a glob pattern.  Delegates to canonical DBPS matcher."""
        from dimensionalbase.core.matching import dbps_match
        return dbps_match(pattern, path)

    @staticmethod
    def _pattern_covers(allowed: str, requested: str) -> bool:
        """Check if an allowed pattern covers a requested scope.  Delegates to canonical DBPS matcher."""
        from dimensionalbase.core.matching import dbps_pattern_covers
        return dbps_pattern_covers(allowed, requested)
