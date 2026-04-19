"""
MCP server implementation for DimensionalBase.

Exposes tools and resources so any MCP-compatible client can read and write
the shared knowledge space.
"""

from __future__ import annotations

import collections
import json
import sys
from typing import Any, Deque, Dict, List

from dimensionalbase.core.types import Event

_mcp_available = False
try:
    from mcp.server import Server
    from mcp.types import Resource, TextContent, Tool
    _mcp_available = True
except ImportError:
    pass


def _entry_to_dict(entry) -> Dict[str, Any]:
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
    if not _mcp_available:
        if sys.version_info < (3, 10):
            raise ImportError(
                "The optional 'mcp' dependency requires Python 3.10+ because the upstream "
                "'mcp' package does not support Python 3.9."
            )
        raise ImportError(
            "The 'mcp' package is required for MCP server support. "
            "Install it with: pip install dimensionalbase[mcp]"
        )

    server = Server("dimensionalbase")
    _event_buffer: Deque[Dict[str, Any]] = collections.deque(maxlen=100)
    _subscriptions: Dict[str, Any] = {}

    def _buffer_event(event: Event) -> None:
        _event_buffer.append({
            "type": event.type.value,
            "path": event.path,
            "data": event.data,
            "source_owner": event.source_owner,
            "timestamp": event.timestamp,
        })

    def _json_text(payload: Dict[str, Any]) -> List[TextContent]:
        return [TextContent(type="text", text=json.dumps(payload, indent=2, default=str))]

    def _error_payload(message: str, *, code: str) -> List[TextContent]:
        return _json_text({"error": {"code": code, "message": message}})

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        return [
            Tool(
                name="db_put",
                description="Write knowledge to the shared DimensionalBase store.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "value": {"type": "string"},
                        "owner": {"type": "string"},
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
                description="Read relevant knowledge from DimensionalBase within a token budget.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "scope": {"type": "string", "default": "**"},
                        "budget": {"type": "integer", "default": 2000},
                        "query": {"type": "string"},
                        "owner": {"type": "string"},
                        "type": {"type": "string"},
                    },
                },
            ),
            Tool(
                name="db_relate",
                description="Discover the mathematical relationship between two entries.",
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
                description="Merge multiple entries into a unified representation, then find nearest entries.",
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
                description="Get the full status of DimensionalBase.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="db_subscribe",
                description="Subscribe to changes matching a glob pattern.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "subscriber": {"type": "string"},
                    },
                    "required": ["pattern", "subscriber"],
                },
            ),
            Tool(
                name="db_unsubscribe",
                description="Cancel a previous subscription created by db_subscribe.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "subscription_id": {"type": "string"},
                    },
                    "required": ["subscription_id"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> List[TextContent]:
        try:
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
                return _json_text(_entry_to_dict(entry))

            if name == "db_get":
                qr = db.get(
                    scope=arguments.get("scope", "**"),
                    budget=arguments.get("budget", 2000),
                    query=arguments.get("query"),
                    owner=arguments.get("owner"),
                    type=arguments.get("type"),
                )
                return _json_text({
                    "entries": [_entry_to_dict(e) for e in qr.entries],
                    "total_matched": qr.total_matched,
                    "tokens_used": qr.tokens_used,
                    "budget_remaining": qr.budget_remaining,
                    "channel_used": qr.channel_used.name,
                    "text": qr.text,
                })

            if name == "db_relate":
                rel = db.relate(arguments["path_a"], arguments["path_b"])
                if rel is None:
                    return _error_payload(
                        "Entries not found or embeddings unavailable",
                        code="relation_unavailable",
                    )
                return _json_text({key: round(float(value), 6) for key, value in rel.items()})

            if name == "db_compose":
                paths = arguments["paths"]
                mode = arguments.get("mode", "attentive")
                k = arguments.get("k", 5)
                vec = db.compose(paths, mode=mode)
                if vec is None:
                    return _error_payload(
                        "Could not compose — entries not found or embeddings unavailable",
                        code="compose_unavailable",
                    )
                nearest = db.materialize(vec, k=k)
                return _json_text({
                    "nearest": [{"path": path, "similarity": round(float(similarity), 4)} for path, similarity in nearest],
                    "mode": mode,
                    "input_paths": paths,
                })

            if name == "db_status":
                return _json_text(db.status())

            if name == "db_subscribe":
                pattern = arguments["pattern"]
                subscriber = arguments["subscriber"]
                sub = db.subscribe(pattern, subscriber, _buffer_event)
                _subscriptions[sub.id] = sub
                return _json_text({
                    "subscribed": True,
                    "subscription_id": sub.id,
                    "pattern": pattern,
                    "subscriber": subscriber,
                    "note": "Events buffered at dimensionalbase://events/recent",
                })

            if name == "db_unsubscribe":
                subscription_id = arguments["subscription_id"]
                sub = _subscriptions.pop(subscription_id, None)
                return _json_text({
                    "unsubscribed": bool(sub and db.unsubscribe(sub)),
                    "subscription_id": subscription_id,
                })

            return _error_payload(f"Unknown tool: {name}", code="unknown_tool")
        except Exception as exc:
            return _error_payload(str(exc), code=f"{name}_failed")

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
        try:
            if uri == "dimensionalbase://trust-report":
                return json.dumps(db.agent_trust_report(), indent=2, default=str)

            if uri == "dimensionalbase://topology":
                return json.dumps(db.knowledge_topology(), indent=2, default=str)

            if uri == "dimensionalbase://events/recent":
                events = list(_event_buffer)
                return json.dumps({"events": events, "count": len(events)}, indent=2, default=str)

            if uri.startswith("dimensionalbase://entries/"):
                path = uri.replace("dimensionalbase://entries/", "")
                entry = db.retrieve(path)
                if entry is None:
                    return json.dumps({"error": {"code": "not_found", "message": f"Entry not found: {path}"}})
                return json.dumps(_entry_to_dict(entry), indent=2, default=str)

            return json.dumps({"error": {"code": "unknown_resource", "message": f"Unknown resource: {uri}"}})
        except Exception as exc:
            return json.dumps({"error": {"code": "resource_failed", "message": str(exc)}})

    return server
