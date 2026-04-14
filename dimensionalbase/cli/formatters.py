"""Rich-based output formatters for the CLI."""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.tree import Tree

    _rich_available = True
except ImportError:
    _rich_available = False


def format_status_table(status: Dict[str, Any]) -> str:
    """Format status dict as a rich table."""
    if not _rich_available:
        import json
        return json.dumps(status, indent=2, default=str)

    console = Console(record=True)
    table = Table(title="DimensionalBase Status", show_lines=True)
    table.add_column("Property", style="cyan", min_width=20)
    table.add_column("Value", style="green")

    table.add_row("Entries", str(status.get("entries", 0)))
    table.add_row("Channel", status.get("channel", "unknown"))
    emb = status.get("embeddings", False)
    emb_str = f"[green]{status.get('embedding_provider', '?')}[/green] ({status.get('embedding_dimension', 0)}d)" if emb else "[dim]off[/dim]"
    table.add_row("Embeddings", emb_str)
    table.add_row("Vector Entries", str(status.get("vector_entries", 0)))
    table.add_row("Encryption", "[green]on[/green]" if status.get("encryption_enabled") else "[dim]off[/dim]")
    table.add_row("Reasoning", "[green]on[/green]" if status.get("reasoning") else "[dim]off[/dim]")
    table.add_row("Subscriptions", str(status.get("subscriptions", 0)))
    table.add_row("Uptime", f"{status.get('uptime_seconds', 0):.1f}s")
    table.add_row("Puts / Gets", f"{status.get('total_puts', 0)} / {status.get('total_gets', 0)}")
    agents = status.get("agents", {})
    if agents:
        table.add_row("Agents", ", ".join(agents.keys()))
    table.add_row("Provenance Nodes", str(status.get("provenance_nodes", 0)))

    console.print(table)
    return console.export_text()


def format_trust_table(report: Dict[str, Any]) -> str:
    """Format trust report as a rich table."""
    if not _rich_available:
        import json
        return json.dumps(report, indent=2, default=str)

    console = Console(record=True)
    table = Table(title="Agent Trust Report", show_lines=True)
    table.add_column("Agent", style="cyan")
    table.add_column("Trust", style="green", justify="right")
    table.add_column("PageRank", style="yellow", justify="right")
    table.add_column("Rate", justify="right")
    table.add_column("Entries", justify="right")
    table.add_column("Reliable", justify="center")

    for agent_id, data in report.items():
        trust = data.get("global_trust", 0)
        pr = data.get("pagerank_trust", 0)
        rate = data.get("confirmation_rate", 0)
        entries = data.get("total_entries", 0)
        reliable = data.get("is_reliable", False)
        table.add_row(
            agent_id,
            f"{trust:.3f}",
            f"{pr:.3f}",
            f"{rate:.0%}",
            str(entries),
            "[green]yes[/green]" if reliable else "[dim]no[/dim]",
        )

    console.print(table)
    return console.export_text()


def format_entries_table(entries: list, total_matched: int, tokens_used: int,
                         budget_remaining: int = 0) -> str:
    """Format query result entries as a rich table."""
    if not _rich_available:
        lines = []
        for e in entries:
            lines.append(f"  [{e.path}] ({e.owner}, conf={e.confidence:.2f}): {e.value}")
        lines.append(f"\n  {total_matched} matched, {tokens_used} tokens used")
        return "\n".join(lines)

    console = Console(record=True)
    table = Table(title=f"Query Results ({total_matched} matched, {tokens_used} tokens)",
                  show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Path", style="cyan", max_width=30)
    table.add_column("Owner", style="blue")
    table.add_column("Type", style="magenta")
    table.add_column("Conf", style="green", justify="right", width=5)
    table.add_column("Value", max_width=60)

    for i, e in enumerate(entries, 1):
        val = e.value if len(e.value) <= 60 else e.value[:57] + "..."
        table.add_row(str(i), e.path, e.owner, e.type.value, f"{e.confidence:.2f}", val)

    console.print(table)
    return console.export_text()


def format_lineage(chain: list) -> str:
    """Format provenance lineage as a rich tree."""
    if not _rich_available:
        return "\n".join(f"  {node}" for node in chain)

    console = Console(record=True)
    if not chain:
        console.print("[dim]  No lineage found.[/dim]")
        return console.export_text()

    first = chain[0]
    tree = Tree(f"[bold cyan]{getattr(first, 'path', str(first))}[/bold cyan]")
    for node in chain:
        derivation = getattr(node, "derivation", None)
        owner = getattr(node, "owner", "?")
        version = getattr(node, "version", "?")
        d_name = derivation.value if derivation else "?"
        style = "red" if d_name == "contradicted" else "green" if d_name == "confirmed" else "white"
        tree.add(f"[{style}]v{version}[/{style}] [{style}]{d_name}[/{style}] by [blue]{owner}[/blue]")

    console.print(tree)
    return console.export_text()
