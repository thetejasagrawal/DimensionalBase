"""
Entry point for ``python -m dimensionalbase.server`` or ``dimensionalbase-server``.
"""

from __future__ import annotations

import argparse
import sys


def _is_loopback_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def main() -> None:
    parser = argparse.ArgumentParser(description="DimensionalBase REST API Server")
    parser.add_argument("--host", default=None, help="Bind host")
    parser.add_argument("--port", type=int, default=None, help="Bind port")
    parser.add_argument("--db-path", default=None, help="SQLite path")
    parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable auto-reload for development",
    )
    parser.add_argument("--config", default=".dimensionalbase.json", help="Project config path")
    parser.add_argument("--prefer-embedding", choices=["local", "openai"], default=None)
    parser.add_argument("--openai-api-key", default=None, help="OpenAI API key for embeddings")
    parser.add_argument("--api-key", default=None, help="Admin API key for secure server mode")
    parser.add_argument("--admin-agent-id", default="admin", help="Admin agent id for the bootstrapped API key")
    parser.add_argument(
        "--insecure",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Disable API key protection for local-only use",
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required. Install with: pip install dimensionalbase[server]")
        raise SystemExit(1)

    from dimensionalbase.runtime import ServerSettings, build_database, wrap_for_server
    from dimensionalbase.server.app import create_app
    from dimensionalbase.server.logging_config import configure_logging

    configure_logging()

    settings = ServerSettings.from_sources(
        config_path=args.config,
        overrides={
            "host": args.host,
            "port": args.port,
            "db_path": args.db_path,
            "reload": args.reload,
            "prefer_embedding": args.prefer_embedding,
            "openai_api_key": args.openai_api_key,
            "api_key": args.api_key,
            "admin_agent_id": args.admin_agent_id,
            "secure": None if args.insecure is None else not args.insecure,
        },
    )
    db = wrap_for_server(build_database(settings), settings)
    if not settings.secure:
        print(
            "WARNING: DimensionalBase is starting with authentication disabled.",
            file=sys.stderr,
        )
        if not _is_loopback_host(settings.host):
            print(
                f"WARNING: insecure mode on {settings.host} exposes the server to the network.",
                file=sys.stderr,
            )
    app = create_app(db, server_config={
        "cors_origins": [o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        "cors_origin_regex": settings.cors_origin_regex,
        "rate_limit_read": settings.rate_limit_read,
        "rate_limit_write": settings.rate_limit_write,
        "max_request_body_bytes": settings.max_request_body_bytes,
        "request_timeout_seconds": settings.request_timeout_seconds,
    })

    uvicorn.run(app, host=settings.host, port=settings.port, reload=settings.reload, timeout_graceful_shutdown=10)


if __name__ == "__main__":
    main()
