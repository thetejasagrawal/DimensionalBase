"""
Entry point for ``python -m dimensionalbase.server`` or ``dimensionalbase-server``.
"""

from __future__ import annotations

import argparse


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
    app = create_app(db)

    uvicorn.run(app, host=settings.host, port=settings.port, reload=settings.reload)


if __name__ == "__main__":
    main()
