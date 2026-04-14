"""
MCP server implementation for DimensionalBase.

Exposes 6 tools and 3 resources so any MCP-compatible client (Claude Code,
Cursor, Windsurf, etc.) can read/write the shared knowledge space.
"""

from __future__ import annotations

import collections
import json
import time
from typing import Any, Deque, Dict, List, Optional

from dimensionalbase.core.types import Event

# Lazy import — the mcp package is optional
_mcp_available = False
try:
    from mcp.server import Server
    from mcp.types import Resource, TextContent, Tool
    _mcp_available = True
except ImportError:
    pass


def _entry_to_dict(entry) -> Dict[str, Any]:
    """Serialize a KnowledgeEntry to a JSON-safe dict."""
    return {
        "id": entry.id,
        "path": entry.path,
        "value": entry.value,
        "owner": entry.owner,
        "type": entry.type.value,
        "confidence": entry.confidence,
        "refs": entry.refs,
        "version": entry.version,
        "ttl": entry.ttl.value,
        "created_at": entry.created_at,
        "updated_at": entry.updated_at,
        "metadata": entry.metadata,
        "raw_score": getattr(entry, "_raw_score", 0.0),
        "score": getattr(entry, "_score", 0.0),
    }


def create_server(db) -> "Server":
    """Create an MCP server wrapping a DimensionalBase instance.

    Args:
        db: A ``DimensionalBase`` instance.

    Returns:
        An MCP ``Server`` ready to be run with stdio transport.
    """
    if not _mcp_available:
        raise ImportError(
            "The 'mcp' package is required for MCP server support. "
            "Install it with: pip install dimensionalbase[mcp]"
        )

    server = Server("dimensionalbase")

    # Event buffer for subscriptions (MCP is request/response, so we buffer)
    _event_buffer: Deque[Dict[str, Any]] = collections.deque(maxlen=100)

    def _buffer_event(event: Event) -> None:
        _event_buffer.append({
            "type": event.type.value,
            "path": event.path,
            "data": event.data,
            "source_owner": event.source_owner,
            "timestamp": event.timestamp,
        })

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        return [
            Tool(
                name="db_put",
                description=(
                    "Write knowledge to the shared DimensionalBase store. "
                    "Paths are hierarchical (e.g., 'task/auth/status'). "
                    "Types: fact, decision, plan, observation. "
                    "Confidence: 0.0-1.0. Refs: list of related paths."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Hierarchical path (e.g., 'task/auth/status')"},
                        "value": {"type": "string", "description": "The knowledge content"},
                        "owner": {"type": "string", "description": "Agent identifier writing this entry"},
                        "type": {"type": "string", "enum": ["fact", "decision", "plan", "observation"], "default": "fact"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 1.0},
                        "refs": {"type": "array", "items": {"type": "string"}, "default": []},
                        "ttl": {"type": "string", "enum": ["turn", "session", "persistent"], "default": "session"},
                    },
                    "required": ["path", "value", "owner"],
                },
            ),
            Tool(
                name="db_get",
                description=(
                    "Read relevant knowledge from DimensionalBase within a token budget. "
                    "The system scores entries by recency, confidence, semantic similarity, "
                    "and reference distance, then packs the most relevant ones into the budget."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "scope": {"type": "string", "default": "**", "description": "Glob pattern for paths (e.g., 'task/**')"},
                        "budget": {"type": "integer", "default": 2000, "description": "Token budget"},
                        "query": {"type": "string", "description": "Semantic query to boost relevant entries"},
                        "owner": {"type": "string", "description": "Filter by owner"},
                        "type": {"type": "string", "description": "Filter by entry type"},
                    },
                },
            ),
            Tool(
                name="db_relate",
                description=(
                    "Discover the mathematical relationship between two entries. "
                    "Returns cosine similarity, angular distance, parallelism, "
                    "opposition, independence, projection, and residual."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path_a": {"type": "string"},
                        "path_b": {"type": "string"},
                    },
                    "required": ["path_a", "path_b"],
                },
            ),
            Tool(
                name="db_compose",
                description=(
                    "Merge multiple entries into a unified representation, then find "
                    "the top-k nearest entries. Modes: weighted_mean, principal, "
                    "grassmann, attentive."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "paths": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                        "mode": {"type": "string", "enum": ["weighted_mean", "principal", "grassmann", "attentive"], "default": "attentive"},
                        "k": {"type": "integer", "default": 5},
                    },
                    "required": ["paths"],
                },
            ),
            Tool(
                name="db_status",
                description="Get the full status of the DimensionalBase — entry count, channels, agents, trust, topology.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="db_subscribe",
                description=(
                    "Subscribe to changes matching a glob pattern. "
                    "Events are buffered and can be read via the "
                    "dimensionalbase://events/recent resource."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Glob pattern (e.g., 'task/**')"},
                        "subscriber": {"type": "string", "description": "Subscriber identifier"},
                    },
                    "required": ["pattern", "subscriber"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> List[TextContent]:
        if name == "db_put":
            entry = db.put(
                path=arguments["path"],
                value=arguments["value"],
                owner=arguments["owner"],
                type=arguments.get("type", "fact"),
                confidence=arguments.get("confidence", 1.0),
                refs=arguments.get("refs", []),
                ttl=arguments.get("ttl", "session"),
            )
            result = _entry_to_dict(entry)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "db_get":
            qr = db.get(
                scope=arguments.get("scope", "**"),
                budget=arguments.get("budget", 2000),
                query=arguments.get("query"),
                owner=arguments.get("owner"),
                type=arguments.get("type"),
            )
            result = {
                "entries": [_entry_to_dict(e) for e in qr.entries],
                "total_matched": qr.total_matched,
                "tokens_used": qr.tokens_used,
                "budget_remaining": qr.budget_remaining,
                "channel_used": qr.channel_used.name,
                "text": qr.text,
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "db_relate":
            rel = db.relate(arguments["path_a"], arguments["path_b"])
            if rel is None:
                return [TextContent(type="text", text='{"error": "Entries not found or embeddings unavailable"}')]
            # Convert numpy floats to Python floats
            result = {k: round(float(v), 6) for k, v in rel.items()}
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "db_compose":
            paths = arguments["paths"]
            mode = arguments.get("mode", "attentive")
            k = arguments.get("k", 5)
            vec = db.compose(paths, mode=mode)
            if vec is None:
                return [TextContent(type="text", text='{"error": "Could not compose — entries not found or embeddings unavailable"}')]
            nearest = db.materialize(vec, k=k)
            result = {
                "nearest": [{"path": p, "similarity": round(float(s), 4)} for p, s in nearest],
                "mode": mode,
                "input_paths": paths,
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "db_status":
            status = db.status()
            return [TextContent(type="text", text=json.dumps(status, indent=2, default=str))]

        elif name == "db_subscribe":
            pattern = arguments["pattern"]
            subscriber = arguments["subscriber"]
            sub = db.subscribe(pattern, subscriber, _buffer_event)
            return [TextContent(
                type="text",
                text=json.dumps({
                    "subscribed": True,
                    "subscription_id": sub.id,
                    "pattern": pattern,
                    "subscriber": subscriber,
                    "note": "Events buffered at dimensionalbase://events/recent",
                }, indent=2),
            )]

        return [TextContent(type="text", text=f'{{"error": "Unknown tool: {name}"}}')]

    # ------------------------------------------------------------------
    # Resource definitions
    # ------------------------------------------------------------------

    @server.list_resources()
    async def list_resources() -> List[Resource]:
        return [
            Resource(
                uri="dimensionalbase://trust-report",
                name="Agent Trust Report",
                description="Reliability scores for all agents (Elo + PageRank).",
                mimeType="application/json",
            ),
            Resource(
                uri="dimensionalbase://topology",
                name="Knowledge Topology",
                description="Cluster analysis and void detection in the knowledge space.",
                mimeType="application/json",
            ),
            Resource(
                uri="dimensionalbase://events/recent",
                name="Recent Events",
                description="Buffered events from active subscriptions (max 100).",
                mimeType="application/json",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        if uri == "dimensionalbase://trust-report":
            report = db.agent_trust_report()
            return json.dumps(report, indent=2, default=str)

        elif uri == "dimensionalbase://topology":
            topo = db.knowledge_topology()
            return json.dumps(topo, indent=2, default=str)

        elif uri == "dimensionalbase://events/recent":
            events = list(_event_buffer)
            return json.dumps({"events": events, "count": len(events)}, indent=2, default=str)

        # Entry lookup by path
        elif uri.startswith("dimensionalbase://entries/"):
            path = uri.replace("dimensionalbase://entries/", "")
            entry = db.retrieve(path)
            if entry is None:
                return json.dumps({"error": f"Entry not found: {path}"})
            return json.dumps(_entry_to_dict(entry), indent=2, default=str)

        return json.dumps({"error": f"Unknown resource: {uri}"})

    return server
