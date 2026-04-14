"""WebSocket connection manager for real-time event streaming."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("dimensionalbase.server.ws")

try:
    from fastapi import WebSocket
except ImportError:
    WebSocket = None  # type: ignore


class ConnectionManager:
    """Manages WebSocket connections for DimensionalBase event subscriptions."""

    def __init__(self) -> None:
        self._active: Dict["WebSocket", Dict[str, Optional[str]]] = {}

    async def connect(
        self,
        websocket: "WebSocket",
        pattern: str = "**",
        agent_id: Optional[str] = None,
    ) -> None:
        await websocket.accept()
        self._active[websocket] = {"pattern": pattern, "agent_id": agent_id}
        logger.info("WebSocket client connected (%d total)", len(self._active))

    def disconnect(self, websocket: "WebSocket") -> None:
        self._active.pop(websocket, None)
        logger.info("WebSocket client disconnected (%d total)", len(self._active))

    async def update_pattern(self, websocket: "WebSocket", pattern: str) -> None:
        """Update the subscribed pattern for a connection."""
        if websocket in self._active:
            self._active[websocket]["pattern"] = pattern

    def agent_for(self, websocket: "WebSocket") -> Optional[str]:
        state = self._active.get(websocket, {})
        return state.get("agent_id")

    async def broadcast_event(self, message: Dict[str, Any]) -> None:
        """Send a JSON message to clients whose patterns match the event path."""
        text = json.dumps(message, default=str)
        disconnected = []
        event_path = str(message.get("path", ""))
        for ws, state in list(self._active.items()):
            pattern = state.get("pattern") or "**"
            if not self._matches(pattern, event_path):
                continue
            try:
                await ws.send_text(text)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)

    @property
    def connection_count(self) -> int:
        return len(self._active)

    @staticmethod
    def _matches(pattern: str, path: str) -> bool:
        from dimensionalbase.core.matching import dbps_match
        return dbps_match(pattern, path)
