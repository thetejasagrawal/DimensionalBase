"""
Entry point for ``python -m dimensionalbase.mcp`` or ``dimensionalbase-mcp``.

Starts the MCP server over stdio transport so Claude Code, Cursor, etc.
can connect to a shared DimensionalBase instance.
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DimensionalBase MCP Server",
    )
    parser.add_argument(
        "--db-path",
        default=":memory:",
        help="SQLite database path (default: in-memory)",
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

    # Import here to avoid slow startup if just checking --help
    from dimensionalbase import DimensionalBase
    from dimensionalbase.mcp.server import create_server

    # Determine embedding provider
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
            print(
                "Error: The 'mcp' package is required. "
                "Install it with: pip install dimensionalbase[mcp]",
                file=sys.stderr,
            )
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
