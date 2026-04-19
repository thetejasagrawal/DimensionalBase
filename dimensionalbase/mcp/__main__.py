"""
Entry point for ``python -m dimensionalbase.mcp`` or ``dimensionalbase-mcp``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def _missing_mcp_message() -> str:
    if sys.version_info < (3, 10):
        return (
            "Error: dimensionalbase[mcp] requires Python 3.10+ because the upstream "
            "'mcp' package does not support Python 3.9."
        )
    return "Error: The 'mcp' package is required. Install it with: pip install dimensionalbase[mcp]"


def main() -> None:
    parser = argparse.ArgumentParser(description="DimensionalBase MCP Server")
    parser.add_argument(
        "--db-path",
        default="./dimensionalbase.db",
        help="SQLite database path (default: ./dimensionalbase.db)",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["auto", "local", "openai", "none"],
        default="auto",
        help="Embedding provider to use (default: auto-detect)",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="OpenAI API key (if using openai embeddings)",
    )
    args = parser.parse_args()

    from dimensionalbase import DimensionalBase
    from dimensionalbase.mcp.server import create_server

    provider = None
    if args.embedding_provider == "none":
        from dimensionalbase.embeddings.provider import NullEmbeddingProvider
        provider = NullEmbeddingProvider()

    db = DimensionalBase(
        db_path=args.db_path,
        embedding_provider=provider,
        openai_api_key=args.openai_api_key,
    )

    server = create_server(db)

    async def run() -> None:
        try:
            from mcp.server.stdio import stdio_server
        except ImportError:
            print(_missing_mcp_message(), file=sys.stderr)
            sys.exit(1)

        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    finally:
        db.close()


if __name__ == "__main__":
    main()
