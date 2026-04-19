"""Tests for the security layer — auth, ACL, validation."""

import time

import pytest

from dimensionalbase import DimensionalBase
from dimensionalbase.security.auth import APIKeyManager, AuthError
from dimensionalbase.security.acl import AccessController, AccessDeniedError, AgentPolicy
from dimensionalbase.security.encryption import EncryptionError, FernetEncryptionProvider
from dimensionalbase.security.validation import (
    validate_confidence,
    validate_owner,
    validate_path,
    validate_value,
    ValidationError,
)
from dimensionalbase.security.middleware import SecureDimensionalBase


class TestAPIKeyManager:

    def test_generate_and_validate(self):
        mgr = APIKeyManager()
        key = mgr.generate_key("agent-1")
        assert key.startswith("dmb_")
        result = mgr.validate(key)
        assert result.agent_id == "agent-1"
        mgr.close()

    def test_invalid_key_raises(self):
        mgr = APIKeyManager()
        with pytest.raises(AuthError, match="Invalid"):
            mgr.validate("dmb_fake_key")
        mgr.close()

    def test_revoke_key(self):
        mgr = APIKeyManager()
        key = mgr.generate_key("agent-1")
        assert mgr.revoke(key) is True
        with pytest.raises(AuthError, match="Invalid"):
            mgr.validate(key)
        mgr.close()

    def test_admin_key(self):
        mgr = APIKeyManager()
        key = mgr.generate_key("admin", is_admin=True)
        result = mgr.validate(key)
        assert result.is_admin is True
        mgr.close()

    def test_list_keys(self):
        mgr = APIKeyManager()
        mgr.generate_key("agent-1")
        mgr.generate_key("agent-2")
        keys = mgr.list_keys()
        assert len(keys) == 2
        mgr.close()

    def test_revocation_reaches_other_process_after_cache_ttl(self, tmp_path):
        db_path = str(tmp_path / "keys.db")
        mgr_a = APIKeyManager(db_path=db_path, cache_ttl_seconds=0.01)
        mgr_b = APIKeyManager(db_path=db_path, cache_ttl_seconds=0.01)

        key = mgr_a.generate_key("agent-1")
        assert mgr_a.validate(key).agent_id == "agent-1"

        assert mgr_b.revoke(key) is True
        time.sleep(0.02)

        with pytest.raises(AuthError, match="Invalid"):
            mgr_a.validate(key)

        mgr_a.close()
        mgr_b.close()


class TestAccessControl:

    def test_default_allows_everything(self):
        acl = AccessController()
        # No policy registered = no restrictions
        acl.check_read("unknown-agent", "**")
        acl.check_write("unknown-agent", "any/path")

    def test_admin_bypasses_acl(self):
        acl = AccessController()
        acl.register_policy(AgentPolicy(
            agent_id="admin", is_admin=True,
            allowed_read_patterns=[], allowed_write_patterns=[],
        ))
        acl.check_read("admin", "any/scope")
        acl.check_write("admin", "any/path")

    def test_read_denied(self):
        acl = AccessController()
        acl.register_policy(AgentPolicy(
            agent_id="agent-1",
            allowed_read_patterns=["task/agent-1/**"],
            allowed_write_patterns=["task/agent-1/**"],
        ))
        with pytest.raises(AccessDeniedError):
            acl.check_read("agent-1", "task/agent-2/**")

    def test_write_denied(self):
        acl = AccessController()
        acl.register_policy(AgentPolicy(
            agent_id="agent-1",
            allowed_write_patterns=["task/agent-1/**"],
        ))
        with pytest.raises(AccessDeniedError):
            acl.check_write("agent-1", "task/agent-2/status")

    def test_read_allowed(self):
        acl = AccessController()
        acl.register_policy(AgentPolicy(
            agent_id="agent-1",
            allowed_read_patterns=["task/**"],
        ))
        acl.check_read("agent-1", "task/auth/**")


class TestValidation:

    def test_valid_path(self):
        assert validate_path("task/auth/status") == "task/auth/status"

    def test_empty_path_raises(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_path("")

    def test_traversal_raises(self):
        with pytest.raises(ValidationError, match="traversal"):
            validate_path("task/../../etc/passwd")

    def test_null_byte_raises(self):
        with pytest.raises(ValidationError, match="null"):
            validate_path("task/\x00evil")

    def test_long_path_raises(self):
        with pytest.raises(ValidationError, match="too long"):
            validate_path("a" * 600)

    def test_invalid_chars_raises(self):
        with pytest.raises(ValidationError, match="invalid"):
            validate_path("task/<script>alert</script>")

    def test_path_cannot_start_with_slash(self):
        with pytest.raises(ValidationError, match="start"):
            validate_path("/task/auth/status")

    def test_path_cannot_end_with_slash(self):
        with pytest.raises(ValidationError, match="end"):
            validate_path("task/auth/status/")

    def test_path_cannot_have_empty_segments(self):
        with pytest.raises(ValidationError, match="empty segments"):
            validate_path("task//auth/status")

    def test_valid_value(self):
        assert validate_value("hello world") == "hello world"

    def test_empty_value_raises(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_value("")

    def test_valid_owner(self):
        assert validate_owner("agent-1") == "agent-1"

    def test_invalid_owner_raises(self):
        with pytest.raises(ValidationError, match="invalid"):
            validate_owner("agent 1")

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValidationError, match="0.0-1.0"):
            validate_confidence(1.5)


class TestSecureDimensionalBase:

    def test_put_without_key_when_required(self):
        db = DimensionalBase()
        mgr = APIKeyManager()
        secure = SecureDimensionalBase(db, key_manager=mgr)
        with pytest.raises(AuthError, match="required"):
            secure.put("x", "v", owner="a")
        db.close()
        mgr.close()

    def test_put_with_valid_key(self):
        db = DimensionalBase()
        mgr = APIKeyManager()
        key = mgr.generate_key("agent-1")
        secure = SecureDimensionalBase(db, key_manager=mgr)
        entry = secure.put("task/x", "value", owner="agent-1", api_key=key)
        assert entry.path == "task/x"
        db.close()
        mgr.close()

    def test_acl_prevents_write(self):
        db = DimensionalBase()
        mgr = APIKeyManager()
        key = mgr.generate_key("agent-1")
        acl = AccessController()
        acl.register_policy(AgentPolicy(
            agent_id="agent-1",
            allowed_write_patterns=["agent-1/**"],
        ))
        secure = SecureDimensionalBase(db, key_manager=mgr, acl=acl)
        with pytest.raises(AccessDeniedError):
            secure.put("agent-2/secret", "data", owner="agent-1", api_key=key)
        db.close()
        mgr.close()

    def test_non_admin_cannot_spoof_owner(self):
        db = DimensionalBase()
        mgr = APIKeyManager()
        key = mgr.generate_key("agent-1")
        secure = SecureDimensionalBase(db, key_manager=mgr)
        with pytest.raises(AccessDeniedError, match="cannot write as"):
            secure.put("task/x", "value", owner="agent-2", api_key=key)
        db.close()
        mgr.close()

    def test_admin_can_impersonate_with_as_owner(self):
        db = DimensionalBase()
        mgr = APIKeyManager()
        key = mgr.generate_key("admin", is_admin=True)
        secure = SecureDimensionalBase(db, key_manager=mgr)
        entry = secure.put(
            "task/x",
            "value",
            owner="admin",
            as_owner="agent-2",
            api_key=key,
        )
        assert entry.owner == "agent-2"
        db.close()
        mgr.close()

    def test_no_auth_passthrough(self):
        """Without a key manager, all operations pass through."""
        db = DimensionalBase()
        secure = SecureDimensionalBase(db)
        entry = secure.put("x", "v", owner="a")
        assert entry.path == "x"
        db.close()


class TestEncryption:
    def test_provider_requires_explicit_key_or_passphrase(self):
        with pytest.raises(EncryptionError, match="explicit key or passphrase"):
            FernetEncryptionProvider()
