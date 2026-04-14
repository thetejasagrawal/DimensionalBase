"""Runtime/bootstrap helpers for DimensionalBase applications."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from dimensionalbase import DimensionalBase
from dimensionalbase.security.auth import APIKeyManager
from dimensionalbase.security.middleware import SecureDimensionalBase

DEFAULT_CONFIG_FILE = ".dimensionalbase.json"


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_project_config(path: str = DEFAULT_CONFIG_FILE) -> Dict[str, Any]:
    """Load local project config if present."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


@dataclass
class RuntimeSettings:
    """Unified runtime settings for local, CLI, and server entrypoints."""

    db_path: str = ":memory:"
    prefer_embedding: Optional[str] = None
    openai_api_key: Optional[str] = None
    encryption_key: Optional[str] = None
    encryption_passphrase: Optional[str] = None

    @classmethod
    def from_sources(
        cls,
        config_path: str = DEFAULT_CONFIG_FILE,
        overrides: Optional[Mapping[str, Any]] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "RuntimeSettings":
        config = load_project_config(config_path)
        env = dict(environ or os.environ)
        merged: Dict[str, Any] = {}
        merged.update(config)
        merged.update(
            {
                k: v
                for k, v in {
                    "db_path": env.get("DMB_DB_PATH"),
                    "prefer_embedding": env.get("DMB_PREFER_EMBEDDING") or env.get("DMB_EMBEDDING_PROVIDER"),
                    "openai_api_key": env.get("DMB_OPENAI_API_KEY") or env.get("OPENAI_API_KEY"),
                    "encryption_key": env.get("DMB_ENCRYPTION_KEY"),
                    "encryption_passphrase": env.get("DMB_ENCRYPTION_PASSPHRASE"),
                }.items()
                if v is not None
            }
        )
        if overrides:
            merged.update({k: v for k, v in overrides.items() if v is not None})

        prefer_embedding = merged.get("prefer_embedding", merged.get("embedding_provider"))
        if prefer_embedding == "auto":
            prefer_embedding = None

        return cls(
            db_path=str(merged.get("db_path") or ":memory:"),
            prefer_embedding=prefer_embedding,
            openai_api_key=merged.get("openai_api_key"),
            encryption_key=merged.get("encryption_key"),
            encryption_passphrase=merged.get("encryption_passphrase"),
        )

    def db_kwargs(self) -> Dict[str, Any]:
        return {
            "db_path": self.db_path,
            "prefer_embedding": self.prefer_embedding,
            "openai_api_key": self.openai_api_key,
            "encryption_key": self.encryption_key,
            "encryption_passphrase": self.encryption_passphrase,
        }


@dataclass
class ServerSettings(RuntimeSettings):
    host: str = "0.0.0.0"
    port: int = 8420
    reload: bool = False
    secure: bool = True
    api_key: Optional[str] = None
    admin_agent_id: str = "admin"

    @classmethod
    def from_sources(
        cls,
        config_path: str = DEFAULT_CONFIG_FILE,
        overrides: Optional[Mapping[str, Any]] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "ServerSettings":
        base = RuntimeSettings.from_sources(
            config_path=config_path,
            overrides=overrides,
            environ=environ,
        )
        env = dict(environ or os.environ)
        config = load_project_config(config_path)
        merged: Dict[str, Any] = {}
        merged.update(config)
        merged.update(
            {
                k: v
                for k, v in {
                    "host": env.get("DMB_HOST"),
                    "port": env.get("DMB_PORT"),
                    "reload": env.get("DMB_RELOAD"),
                    "secure": env.get("DMB_SECURE"),
                    "insecure": env.get("DMB_INSECURE"),
                    "api_key": env.get("DMB_API_KEY"),
                    "admin_agent_id": env.get("DMB_ADMIN_AGENT_ID"),
                }.items()
                if v is not None
            }
        )
        if overrides:
            merged.update({k: v for k, v in overrides.items() if v is not None})

        secure_default = not _parse_bool(merged.get("insecure"), False)
        secure = _parse_bool(
            None if merged.get("secure") is None else str(merged.get("secure")),
            secure_default,
        )

        return cls(
            db_path=base.db_path,
            prefer_embedding=base.prefer_embedding,
            openai_api_key=base.openai_api_key,
            encryption_key=base.encryption_key,
            encryption_passphrase=base.encryption_passphrase,
            host=str(merged.get("host") or "0.0.0.0"),
            port=int(merged.get("port") or 8420),
            reload=bool(merged.get("reload")) if isinstance(merged.get("reload"), bool) else _parse_bool(merged.get("reload"), False),
            secure=secure,
            api_key=merged.get("api_key"),
            admin_agent_id=str(merged.get("admin_agent_id") or "admin"),
        )


def build_database(settings: RuntimeSettings) -> DimensionalBase:
    """Construct a DimensionalBase instance from runtime settings."""
    return DimensionalBase(**settings.db_kwargs())


def wrap_for_server(db: DimensionalBase, settings: ServerSettings):
    """Apply secure server defaults around a database instance."""
    if not settings.secure:
        return db

    if not settings.api_key:
        raise RuntimeError(
            "Secure server mode requires an API key. "
            "Set DMB_API_KEY or pass --api-key, or explicitly start with --insecure."
        )

    key_manager = APIKeyManager(db_path=settings.db_path)
    key_manager.ensure_key(
        settings.api_key,
        agent_id=settings.admin_agent_id,
        is_admin=True,
    )
    return SecureDimensionalBase(db, key_manager=key_manager)
