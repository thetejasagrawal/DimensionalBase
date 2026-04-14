"""
Security middleware for DimensionalBase.

``SecureDimensionalBase`` wraps a ``DimensionalBase`` instance and applies
authentication, authorization, and input validation on every operation.
The core library remains usable without auth for embedded single-machine use.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from dimensionalbase.core.types import Event, QueryResult, Subscription
from dimensionalbase.security.acl import AccessController, AgentPolicy
from dimensionalbase.security.auth import APIKeyManager, AuthError, APIKey
from dimensionalbase.security.acl import AccessDeniedError
from dimensionalbase.security.validation import (
    validate_confidence,
    validate_metadata,
    validate_owner,
    validate_path,
    validate_value,
)


class SecureDimensionalBase:
    """Security wrapper around DimensionalBase.

    Intercepts all public methods and applies:
    - API key authentication
    - Path-based access control
    - Input validation

    Usage::

        db = DimensionalBase()
        secure_db = SecureDimensionalBase(db, key_manager=keys, acl=acl)
        secure_db.put("task/x", "value", owner="agent-1", api_key="dmb_...")
    """

    def __init__(
        self,
        db: Any,
        key_manager: Optional[APIKeyManager] = None,
        acl: Optional[AccessController] = None,
    ) -> None:
        self._db = db
        self._keys = key_manager
        self._acl = acl or AccessController()

    def _authenticate_key(self, api_key: Optional[str]) -> Optional[APIKey]:
        """Validate an API key and return the key metadata."""
        if self._keys is None:
            return None
        if api_key is None:
            raise AuthError("API key required")
        return self._keys.validate(api_key)

    def authenticate_agent(self, api_key: Optional[str]) -> str:
        """Validate an API key and return the bound agent id."""
        key_obj = self._authenticate_key(api_key)
        return key_obj.agent_id if key_obj is not None else "anonymous"

    def check_read_access(self, agent_id: str, scope: str) -> None:
        """Public helper for server-side ACL checks."""
        self._acl.check_read(agent_id, scope)

    def check_write_access(self, agent_id: str, path: str) -> None:
        """Public helper for server-side ACL checks."""
        self._acl.check_write(agent_id, path)

    def put(
        self,
        path: str,
        value: str,
        owner: str,
        api_key: Optional[str] = None,
        as_owner: Optional[str] = None,
        **kwargs,
    ):
        """Write knowledge with auth, ACL, and validation."""
        key_obj = self._authenticate_key(api_key)
        agent_id = key_obj.agent_id if key_obj is not None else "anonymous"
        is_admin = bool(key_obj.is_admin) if key_obj is not None else True

        validate_path(path)
        validate_value(value)
        validate_owner(owner)
        validate_confidence(kwargs.get("confidence", 1.0))
        if "metadata" in kwargs:
            validate_metadata(kwargs["metadata"])

        effective_owner = owner
        if as_owner is not None:
            validate_owner(as_owner)
            if not is_admin:
                raise AccessDeniedError("Only admin keys may impersonate another owner")
            effective_owner = as_owner
        elif not is_admin and owner != agent_id:
            raise AccessDeniedError(
                f"Authenticated agent '{agent_id}' cannot write as '{owner}'"
            )

        self._acl.check_write(agent_id, path)
        return self._db.put(path=path, value=value, owner=effective_owner, **kwargs)

    def get(
        self,
        scope: str = "**",
        api_key: Optional[str] = None,
        **kwargs,
    ) -> QueryResult:
        """Read knowledge with auth and ACL."""
        agent_id = self.authenticate_agent(api_key)
        self._acl.check_read(agent_id, scope)
        return self._db.get(scope=scope, **kwargs)

    def retrieve(self, path: str, api_key: Optional[str] = None):
        """Retrieve a single entry with auth and ACL."""
        agent_id = self.authenticate_agent(api_key)
        self._acl.check_read(agent_id, path)
        return self._db.retrieve(path)

    def delete(self, path: str, api_key: Optional[str] = None) -> bool:
        """Delete with auth and ACL."""
        agent_id = self.authenticate_agent(api_key)
        self._acl.check_write(agent_id, path)
        return self._db.delete(path)

    def subscribe(
        self,
        pattern: str,
        subscriber: str,
        callback: Callable[[Event], None],
        api_key: Optional[str] = None,
    ) -> Subscription:
        """Subscribe with auth and ACL."""
        agent_id = self.authenticate_agent(api_key)
        self._acl.check_read(agent_id, pattern)
        return self._db.subscribe(pattern, subscriber, callback)

    def status(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Status (requires valid key if key manager is set)."""
        self.authenticate_agent(api_key)
        return self._db.status()

    def agent_trust_report(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        self.authenticate_agent(api_key)
        return self._db.agent_trust_report()

    def knowledge_topology(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        self.authenticate_agent(api_key)
        return self._db.knowledge_topology()

    def lineage(self, path: str, api_key: Optional[str] = None):
        agent_id = self.authenticate_agent(api_key)
        self._acl.check_read(agent_id, path)
        return self._db.lineage(path)

    def relate(self, path_a: str, path_b: str, api_key: Optional[str] = None):
        agent_id = self.authenticate_agent(api_key)
        self._acl.check_read(agent_id, path_a)
        self._acl.check_read(agent_id, path_b)
        return self._db.relate(path_a, path_b)

    def compose(
        self,
        paths: List[str],
        mode: str = "attentive",
        api_key: Optional[str] = None,
    ):
        agent_id = self.authenticate_agent(api_key)
        for path in paths:
            self._acl.check_read(agent_id, path)
        return self._db.compose(paths, mode=mode)

    def close(self) -> None:
        """Close the wrapped DB and any associated key manager."""
        self._db.close()
        if self._keys is not None:
            self._keys.close()

    # Passthrough for methods that don't need auth in embedded mode
    def __getattr__(self, name: str) -> Any:
        return getattr(self._db, name)
