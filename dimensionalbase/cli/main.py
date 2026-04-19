"""
DimensionalBase CLI — manage, inspect, and operate a DimensionalBase instance.
"""

from __future__ import annotations

import json

try:
    import click
except ImportError:
    print("Error: click is required. Install with: pip install dimensionalbase[cli]")
    raise SystemExit(1)

from dimensionalbase import DimensionalBase
from dimensionalbase.runtime import (
    DEFAULT_CONFIG_FILE,
    RuntimeSettings,
    ServerSettings,
    build_database,
    wrap_for_server,
)

_CONFIG_FILE = DEFAULT_CONFIG_FILE


def _is_loopback_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def _close_ctx_db(ctx) -> None:
    db = ctx.obj.get("db") if ctx.obj else None
    if db is not None:
        db.close()
        ctx.obj["db"] = None


def _get_db(ctx) -> DimensionalBase:
    ctx.ensure_object(dict)
    if ctx.obj.get("db") is None:
        settings = RuntimeSettings.from_sources(config_path=_CONFIG_FILE)
        ctx.obj["db"] = build_database(settings)
    return ctx.obj["db"]


@click.group()
@click.pass_context
def cli(ctx):
    """DimensionalBase — the protocol and database for AI communication."""
    ctx.ensure_object(dict)
    ctx.call_on_close(lambda: _close_ctx_db(ctx))


@cli.command()
@click.option("--path", default="./dimensionalbase.db", help="Database file path")
def init(path):
    """Initialize a new DimensionalBase project."""
    config = {"db_path": path, "prefer_embedding": None}
    with open(_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    db = build_database(RuntimeSettings(db_path=path))
    db.close()
    click.echo(f"Initialized DimensionalBase at {path}")
    click.echo(f"Config written to {_CONFIG_FILE}")


@cli.command()
@click.argument("path")
@click.argument("value")
@click.option("--owner", required=True, help="Agent identifier")
@click.option("--type", "entry_type", default="fact", type=click.Choice(["fact", "decision", "plan", "observation"]))
@click.option("--confidence", default=1.0, type=float, help="Confidence 0.0-1.0")
@click.option("--ttl", default="session", type=click.Choice(["turn", "session", "persistent"]))
@click.option("--refs", default="", help="Comma-separated related paths")
@click.pass_context
def put(ctx, path, value, owner, entry_type, confidence, ttl, refs):
    """Write a knowledge entry."""
    db = _get_db(ctx)
    refs_list = [r.strip() for r in refs.split(",") if r.strip()] if refs else []
    entry = db.put(path=path, value=value, owner=owner, type=entry_type,
                   confidence=confidence, ttl=ttl, refs=refs_list)
    click.echo(f"Stored: {entry.path} (v{entry.version}, {entry.type.value}, conf={entry.confidence})")


@cli.command()
@click.argument("scope", default="**")
@click.option("--budget", default=2000, type=int, help="Token budget")
@click.option("--query", default=None, help="Semantic search query")
@click.option("--owner", default=None, help="Filter by owner")
@click.option("--type", "entry_type", default=None, help="Filter by type")
@click.option("--format", "fmt", default="text", type=click.Choice(["text", "json"]))
@click.pass_context
def get(ctx, scope, budget, query, owner, entry_type, fmt):
    """Read knowledge entries within a token budget."""
    db = _get_db(ctx)
    result = db.get(scope=scope, budget=budget, query=query, owner=owner, type=entry_type)
    if fmt == "json":
        entries = []
        for entry in result.entries:
            entries.append({
                "path": entry.path,
                "value": entry.value,
                "owner": entry.owner,
                "type": entry.type.value,
                "confidence": entry.confidence,
                "version": entry.version,
                "raw_score": entry._raw_score,
                "score": entry._score,
            })
        click.echo(json.dumps({
            "entries": entries,
            "total_matched": result.total_matched,
            "tokens_used": result.tokens_used,
            "budget_remaining": result.budget_remaining,
        }, indent=2))
    else:
        if not result.entries:
            click.echo("(no entries found)")
        else:
            try:
                from dimensionalbase.cli.formatters import format_entries_table
                click.echo(format_entries_table(
                    result.entries, result.total_matched,
                    result.tokens_used, result.budget_remaining,
                ))
            except Exception:
                for entry in result.entries:
                    click.echo(f"  [{entry.path}] ({entry.owner}, conf={entry.confidence:.2f}): {entry.value}")
                click.echo(f"\n  {result.total_matched} matched, {result.tokens_used} tokens used")


@cli.command()
@click.option("--format", "fmt", default="text", type=click.Choice(["text", "json"]))
@click.pass_context
def status(ctx, fmt):
    """Show database status."""
    db = _get_db(ctx)
    report = db.status()
    if fmt == "json":
        click.echo(json.dumps(report, indent=2, default=str))
    else:
        try:
            from dimensionalbase.cli.formatters import format_status_table
            click.echo(format_status_table(report))
        except Exception:
            click.echo(f"  Entries:       {report['entries']}")
            click.echo(f"  Channel:       {report['channel']}")
            click.echo(f"  Embeddings:    {report['embeddings']}")
            click.echo(f"  Reasoning:     {report['reasoning']}")
            click.echo(f"  Subscriptions: {report['subscriptions']}")
            click.echo(f"  Uptime:        {report['uptime_seconds']}s")
            click.echo(f"  Puts/Gets:     {report['total_puts']}/{report['total_gets']}")
            if report.get("agents"):
                click.echo(f"  Agents:        {list(report['agents'].keys())}")


@cli.command("trust-report")
@click.option("--format", "fmt", default="text", type=click.Choice(["text", "json"]))
@click.pass_context
def trust_report(ctx, fmt):
    """Show agent trust scores."""
    db = _get_db(ctx)
    report = db.agent_trust_report()
    if fmt == "json":
        click.echo(json.dumps(report, indent=2, default=str))
    else:
        if not report:
            click.echo("  (no agents registered)")
        else:
            try:
                from dimensionalbase.cli.formatters import format_trust_table
                click.echo(format_trust_table(report))
            except Exception:
                for agent_id, data in report.items():
                    trust = data.get("global_trust", 0)
                    pagerank = data.get("pagerank_trust", 0)
                    click.echo(f"  {agent_id}: trust={trust:.3f}, pagerank={pagerank:.3f}")


@cli.command()
@click.pass_context
def topology(ctx):
    """Show knowledge topology."""
    db = _get_db(ctx)
    click.echo(json.dumps(db.knowledge_topology(), indent=2, default=str))


@cli.command()
@click.argument("path")
@click.pass_context
def lineage(ctx, path):
    """Show provenance lineage for an entry."""
    db = _get_db(ctx)
    chain = db.lineage(path)
    if not chain:
        click.echo(f"  No lineage found for: {path}")
    else:
        try:
            from dimensionalbase.cli.formatters import format_lineage
            click.echo(format_lineage(chain))
        except Exception:
            for node in chain:
                click.echo(f"  {node}")


@cli.command()
@click.argument("path")
@click.pass_context
def delete(ctx, path):
    """Delete an entry."""
    db = _get_db(ctx)
    deleted = db.delete(path)
    if deleted:
        click.echo(f"  Deleted: {path}")
    else:
        click.echo(f"  Not found: {path}")


@cli.command("export")
@click.option("--output", "-o", default=None, help="Output file (default: stdout)")
@click.option("--format", "fmt", default="json", type=click.Choice(["json"]))
@click.pass_context
def export_db(ctx, output, fmt):
    """Export all entries."""
    db = _get_db(ctx)
    result = db.get(scope="**", budget=999999)
    entries = []
    for entry in result.entries:
        entries.append({
            "path": entry.path,
            "value": entry.value,
            "owner": entry.owner,
            "type": entry.type.value,
            "confidence": entry.confidence,
            "refs": entry.refs,
            "version": entry.version,
            "ttl": entry.ttl.value,
            "metadata": entry.metadata,
        })
    data = json.dumps({"entries": entries, "count": len(entries)}, indent=2)
    if output:
        with open(output, "w") as f:
            f.write(data)
        click.echo(f"  Exported {len(entries)} entries to {output}")
    else:
        click.echo(data)


@cli.command("import")
@click.argument("file", type=click.Path(exists=True))
@click.pass_context
def import_db(ctx, file):
    """Import entries from a JSON file."""
    db = _get_db(ctx)
    with open(file) as f:
        data = json.load(f)
    entries = data.get("entries", [])
    count = 0
    for entry in entries:
        db.put(
            path=entry["path"],
            value=entry["value"],
            owner=entry["owner"],
            type=entry.get("type", "fact"),
            confidence=entry.get("confidence", 1.0),
            refs=entry.get("refs", []),
            ttl=entry.get("ttl", "session"),
            metadata=entry.get("metadata", {}),
        )
        count += 1
    click.echo(f"  Imported {count} entries from {file}")


@cli.command()
def demo():
    """Run an interactive demo — see DimensionalBase in action in 30 seconds."""
    from dimensionalbase.cli.demo import run_demo
    run_demo()


@cli.command()
@click.option("--host", default=None, help="Bind host")
@click.option("--port", default=None, type=int, help="Bind port")
@click.option("--config", "config_path", default=_CONFIG_FILE, help="Project config path")
@click.option("--api-key", default=None, help="Admin API key for secure server mode")
@click.option("--admin-agent-id", default="admin", help="Admin agent id")
@click.option("--prefer-embedding", type=click.Choice(["local", "openai"]), default=None)
@click.option("--insecure", is_flag=True, default=False, help="Disable API key protection for local-only use")
@click.option("--seed-demo", is_flag=True, default=False, help="Populate with demo data on startup")
@click.pass_context
def serve(ctx, host, port, config_path, api_key, admin_agent_id, prefer_embedding, insecure, seed_demo):
    """Start the REST API server."""
    try:
        import uvicorn
    except ImportError:
        click.echo("Error: uvicorn required. Install with: pip install dimensionalbase[server]")
        raise SystemExit(1)

    from dimensionalbase.server.app import create_app

    settings = ServerSettings.from_sources(
        config_path=config_path,
        overrides={
            "host": host,
            "port": port,
            "api_key": api_key,
            "admin_agent_id": admin_agent_id,
            "prefer_embedding": prefer_embedding,
            "secure": None if not insecure else False,
        },
    )
    db_instance = build_database(settings)

    if seed_demo:
        from dimensionalbase.server.seed import seed_demo_data
        seed_demo_data(db_instance)
        click.echo("Demo data loaded.")

    db = wrap_for_server(db_instance, settings)
    if not settings.secure:
        click.echo(
            "WARNING: DimensionalBase is starting with authentication disabled.",
            err=True,
        )
        if not _is_loopback_host(settings.host):
            click.echo(
                f"WARNING: insecure mode on {settings.host} exposes the server to the network.",
                err=True,
            )

    app = create_app(db, server_config={
        "cors_origins": [o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        "cors_origin_regex": settings.cors_origin_regex,
        "rate_limit_read": settings.rate_limit_read,
        "rate_limit_write": settings.rate_limit_write,
        "max_request_body_bytes": settings.max_request_body_bytes,
        "request_timeout_seconds": settings.request_timeout_seconds,
    })

    dashboard_host = "localhost" if _is_loopback_host(settings.host) else settings.host
    dashboard_url = f"http://{dashboard_host}:{settings.port}/dashboard/"
    click.echo(f"Starting DimensionalBase server at {settings.host}:{settings.port} (db: {settings.db_path})")
    if seed_demo:
        click.echo(f"Dashboard: {dashboard_url}")
    uvicorn.run(app, host=settings.host, port=settings.port)
