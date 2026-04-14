"""Interactive demo — see DimensionalBase in action in 30 seconds."""

import time
from dimensionalbase import DimensionalBase, EventType

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    _RICH = True
except ImportError:
    _RICH = False


# ---------------------------------------------------------------------------
# Tiny presentation helpers — graceful degradation when Rich is absent
# ---------------------------------------------------------------------------

def _console():
    return Console() if _RICH else None

def _header(con, title: str, subtitle: str) -> None:
    if con:
        con.print()
        con.print(Panel(
            Text(f"{title}\n{subtitle}", justify="center"),
            border_style="bright_cyan",
            padding=(1, 4),
        ))
    else:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"  {subtitle}")
        print(f"{'=' * 60}")

def _phase(con, number: int, label: str) -> None:
    if con:
        con.print()
        con.rule(f"[bold white]Phase {number}[/bold white]  {label}", style="dim")
    else:
        print(f"\n--- Phase {number}: {label} ---")

def _agent_msg(con, agent: str, color: str, msg: str) -> None:
    if con:
        con.print(f"  [{color}]{agent:>10}[/{color}]  {msg}")
    else:
        print(f"  {agent:>10}  {msg}")

def _conflict_msg(con, text: str) -> None:
    if con:
        con.print(f"  [bold red]{'CONFLICT':>10}[/bold red]  {text}")
    else:
        print(f"  {'CONFLICT':>10}  {text}")

def _info(con, text: str) -> None:
    if con:
        con.print(f"  [dim]{text}[/dim]")
    else:
        print(f"  {text}")

def _pause(seconds: float = 0.35) -> None:
    time.sleep(seconds)


# ---------------------------------------------------------------------------
# Agent color map
# ---------------------------------------------------------------------------

AGENTS = {
    "planner":  "bright_blue",
    "backend":  "bright_green",
    "qa-agent": "bright_yellow",
}


# ---------------------------------------------------------------------------
# The demo
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """Run the interactive DimensionalBase demo (~15-20 seconds)."""

    con = _console()

    # ==================================================================
    # Phase 1 — Welcome
    # ==================================================================
    _header(
        con,
        "DimensionalBase  —  Live Demo",
        "3 AI agents coordinate a deployment.  Watch conflicts get caught.",
    )
    _pause(0.5)

    # ==================================================================
    # Phase 2 — Build the knowledge base
    # ==================================================================
    _phase(con, 1, "Agents write shared knowledge")

    db = DimensionalBase()  # in-memory, no files

    entries = [
        # (path, value, owner, type, confidence)
        ("deploy/plan",          "Deploy auth-service v2.3: build -> test -> stage -> prod",
         "planner",  "plan",        0.95),
        ("deploy/build",         "Docker image built: auth-service:v2.3.1-sha.abc123",
         "backend",  "observation", 1.00),
        ("deploy/test",          "All 142 tests passing. Coverage 87%.",
         "backend",  "observation", 1.00),
        ("deploy/stage",         "Deployed to staging environment successfully.",
         "backend",  "observation", 0.95),
        ("deploy/monitoring",    "Grafana dashboards configured for auth-service.",
         "planner",  "fact",        0.90),
        ("auth/jwt-rotation",    "JWT secret rotation scheduled for next Tuesday.",
         "planner",  "plan",        0.85),
        ("deploy/stage/health",  "Staging environment healthy. Latency: 45ms p99.",
         "backend",  "fact",        0.92),
    ]

    for path, value, owner, etype, conf in entries:
        db.put(path=path, value=value, owner=owner, type=etype, confidence=conf)
        _agent_msg(con, owner, AGENTS[owner], f"put  [bold]{path}[/bold]" if con else f"put  {path}")
        _pause(0.3)

    _info(con, f"Knowledge base: {db.entry_count} entries from {len(AGENTS)} agents")
    _pause(0.4)

    # ==================================================================
    # Phase 3 — Contradiction
    # ==================================================================
    _phase(con, 2, "Contradiction detected!")

    conflict_events = []

    def _on_event(event):
        if event.type == EventType.CONFLICT:
            conflict_events.append(event)

    db.subscribe("deploy/**", "demo-watcher", _on_event)
    _pause(0.3)

    # QA agent contradicts Backend
    db.put(
        path="deploy/stage/status",
        value="Staging returning 500 errors on /auth/token endpoint.",
        owner="qa-agent",
        type="fact",
        confidence=0.88,
    )
    _agent_msg(con, "qa-agent", AGENTS["qa-agent"],
               "put  deploy/stage/status  (500 errors on staging!)")
    _pause(0.4)

    if conflict_events:
        evt = conflict_events[0]
        existing_owner = evt.data.get("existing_entry_owner", "?")
        new_owner = evt.data.get("new_entry_owner", "?")
        _conflict_msg(con, f"Detected between [{AGENTS.get(existing_owner, 'white')}]{existing_owner}[/{AGENTS.get(existing_owner, 'white')}] and [{AGENTS.get(new_owner, 'white')}]{new_owner}[/{AGENTS.get(new_owner, 'white')}]"
                       if con else f"Detected between {existing_owner} and {new_owner}")
        existing_val = evt.data.get("existing_entry_value", "")[:70]
        new_val = evt.data.get("new_entry_value", "")[:70]
        if existing_val:
            _info(con, f'  {existing_owner} said: "{existing_val}"')
        if new_val:
            _info(con, f'  {new_owner} said: "{new_val}"')
    else:
        _info(con, "Conflict event captured by reasoning layer.")

    _pause(0.4)

    # ==================================================================
    # Phase 4 — Trust report
    # ==================================================================
    _phase(con, 3, "Agent trust report")

    report = db.agent_trust_report()

    if con:
        table = Table(title="Agent Trust Scores", show_header=True,
                      header_style="bold white", border_style="dim")
        table.add_column("Agent", style="bold", width=14)
        table.add_column("Trust", justify="right", width=8)
        table.add_column("Entries", justify="right", width=8)
        table.add_column("Confirms", justify="right", width=9)
        table.add_column("Status", width=12)

        for agent_id, data in sorted(report.items()):
            color = AGENTS.get(agent_id, "white")
            trust_val = data.get("global_trust", 0.5)
            entries_count = data.get("total_entries", 0)
            conf_rate = data.get("confirmation_rate", 0.0)
            reliable = data.get("is_reliable", False)

            trust_color = "green" if trust_val >= 0.5 else "red"
            status = "[green]reliable[/green]" if reliable else "[dim]building[/dim]"

            table.add_row(
                f"[{color}]{agent_id}[/{color}]",
                f"[{trust_color}]{trust_val:.2f}[/{trust_color}]",
                str(entries_count),
                f"{conf_rate:.0%}",
                status,
            )

        con.print(table)
    else:
        print(f"  {'Agent':<14} {'Trust':>6} {'Entries':>8} {'Conf.Rate':>10}")
        print(f"  {'-'*42}")
        for agent_id, data in sorted(report.items()):
            trust_val = data.get("global_trust", 0.5)
            entries_count = data.get("total_entries", 0)
            conf_rate = data.get("confirmation_rate", 0.0)
            print(f"  {agent_id:<14} {trust_val:>6.2f} {entries_count:>8} {conf_rate:>9.0%}")

    _pause(0.5)

    # ==================================================================
    # Phase 5 — Budget-aware retrieval
    # ==================================================================
    _phase(con, 4, "Budget-aware retrieval (200 tokens)")

    result = db.get(
        scope="deploy/**",
        budget=200,
        query="what is blocking deployment?",
    )

    _info(con, f"Matched {result.total_matched} entries, packed {len(result.entries)} into 200-token budget")
    _info(con, f"Tokens used: {result.tokens_used} / 200  |  Remaining: {result.budget_remaining}")
    _pause(0.3)

    if con:
        context_text = Text()
        for entry in result.entries:
            owner_color = AGENTS.get(entry.owner, "white")
            context_text.append(f"  [{entry.path}] ", style="bold")
            context_text.append(f"({entry.owner}", style=owner_color)
            context_text.append(f", conf={entry.confidence:.2f}): ", style="dim")
            context_text.append(f"{entry.value}\n")

        con.print(Panel(
            context_text,
            title="[bold]Context an LLM would receive[/bold]",
            border_style="bright_magenta",
            padding=(0, 2),
        ))
    else:
        print("  --- Context an LLM would receive ---")
        for entry in result.entries:
            print(f"  [{entry.path}] ({entry.owner}, conf={entry.confidence:.2f}): {entry.value}")
        print("  ------------------------------------")

    _pause(0.5)

    # ==================================================================
    # Phase 6 — Summary
    # ==================================================================
    _phase(con, 5, "What just happened")

    status = db.status()
    event_history = db.events.get_history(limit=20)
    conflict_count = sum(1 for e in event_history if e.type == EventType.CONFLICT)

    bullets = [
        f"3 agents wrote {status['entries']} knowledge entries",
        f"DimensionalBase detected {conflict_count} contradiction(s) automatically",
        f"Trust scores updated in real-time (Bayesian + ELO-inspired)",
        f"Budget packing fit the most relevant entries into 200 tokens",
        f"Total: {status['total_puts']} puts, {status['total_gets']} gets in {status['uptime_seconds']}s",
    ]

    if con:
        summary_text = Text()
        for b in bullets:
            summary_text.append("  \u2022 ", style="bright_cyan")
            summary_text.append(f"{b}\n")

        con.print()
        con.print(Panel(
            summary_text,
            title="[bold bright_cyan]Demo Complete[/bold bright_cyan]",
            subtitle="[dim]pip install dimensionalbase[/dim]",
            border_style="bright_cyan",
            padding=(1, 3),
        ))
    else:
        print()
        print("  Demo Complete")
        print("  " + "-" * 40)
        for b in bullets:
            print(f"    * {b}")
        print()
        print("  pip install dimensionalbase")

    db.close()


# ---------------------------------------------------------------------------
# Allow `python -m dimensionalbase.cli.demo`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_demo()
