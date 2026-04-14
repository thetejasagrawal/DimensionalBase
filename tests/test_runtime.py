"""Tests for runtime/bootstrap helpers."""

import json

import pytest

from dimensionalbase.runtime import RuntimeSettings, ServerSettings, build_database, wrap_for_server
from dimensionalbase.security.middleware import SecureDimensionalBase


class TestRuntimeSettings:
    def test_legacy_embedding_provider_config_is_respected(self, tmp_path):
        config_path = tmp_path / "dmb.json"
        config_path.write_text(json.dumps({"db_path": "runtime.db", "embedding_provider": "local"}))

        settings = RuntimeSettings.from_sources(config_path=str(config_path), environ={})

        assert settings.db_path == "runtime.db"
        assert settings.prefer_embedding == "local"

    def test_secure_server_requires_api_key(self, tmp_db_path):
        settings = ServerSettings(db_path=tmp_db_path, secure=True, api_key=None)
        db = build_database(settings)
        with pytest.raises(RuntimeError, match="API key"):
            wrap_for_server(db, settings)
        db.close()

    def test_secure_server_wraps_db_with_bootstrapped_key(self, tmp_db_path):
        settings = ServerSettings(
            db_path=tmp_db_path,
            secure=True,
            api_key="super-secret",
            admin_agent_id="ops-admin",
        )
        secure = wrap_for_server(build_database(settings), settings)

        assert isinstance(secure, SecureDimensionalBase)
        entry = secure.put("task/x", "value", owner="ops-admin", api_key="super-secret")
        assert entry.path == "task/x"
        assert secure.authenticate_agent("super-secret") == "ops-admin"
        secure.close()
