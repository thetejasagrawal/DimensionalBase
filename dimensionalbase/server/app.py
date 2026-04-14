"""
FastAPI application factory for DimensionalBase REST API.

Usage::

    from dimensionalbase import DimensionalBase
    from dimensionalbase.server import create_app

    db = DimensionalBase()
    app = create_app(db)
    # Run with: uvicorn dimensionalbase.server.app:app
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("dimensionalbase.server")

try:
    from fastapi import FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse, FileResponse
except ImportError:
    raise ImportError(
        "fastapi is required for the server. "
        "Install it with: pip install dimensionalbase[server]"
    )

from dimensionalbase.server.models import (
    ComposeRequest,
    EntryResponse,
    PutRequest,
    QueryResultResponse,
    RelateRequest,
)
from dimensionalbase.server.ws import ConnectionManager
from dimensionalbase.exceptions import BudgetExhaustedError, EntryValidationError
from dimensionalbase.security.auth import AuthError
from dimensionalbase.security.acl import AccessDeniedError
from dimensionalbase.security.middleware import SecureDimensionalBase
from dimensionalbase.security.validation import ValidationError


def _entry_to_response(entry) -> Dict[str, Any]:
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


def create_app(db=None) -> FastAPI:
    """Create a FastAPI application wrapping a DimensionalBase instance.

    Args:
        db: A ``DimensionalBase`` instance. If None, creates an in-memory one.

    Returns:
        A FastAPI application.
    """
    if db is None:
        from dimensionalbase import DimensionalBase
        db = DimensionalBase()

    app = FastAPI(
        title="DimensionalBase API",
        description="REST API for the DimensionalBase protocol and database.",
        version="0.4.0",
    )

    ws_manager = ConnectionManager()
    secure_db = isinstance(db, SecureDimensionalBase)

    # Helper to push events to WebSocket clients
    def _on_event(event) -> None:
        data = {
            "type": event.type.value,
            "path": event.path,
            "data": event.data,
            "source_owner": event.source_owner,
            "timestamp": event.timestamp,
        }
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(ws_manager.broadcast_event(data))
            else:
                loop.run_until_complete(ws_manager.broadcast_event(data))
        except RuntimeError:
            asyncio.run(ws_manager.broadcast_event(data))

    def _handle_api_error(exc: Exception) -> HTTPException:
        if isinstance(exc, HTTPException):
            return exc
        if isinstance(exc, AuthError):
            return HTTPException(status_code=401, detail=str(exc))
        if isinstance(exc, AccessDeniedError):
            return HTTPException(status_code=403, detail=str(exc))
        if isinstance(exc, (ValidationError, EntryValidationError, BudgetExhaustedError, ValueError)):
            return HTTPException(status_code=400, detail=str(exc))
        logger.exception("Unhandled server error")
        return HTTPException(status_code=500, detail="Internal server error")

    # Subscribe to all events for WS broadcasting. Use the raw DB when the
    # public surface is wrapped by SecureDimensionalBase.
    event_source = db._db if secure_db else db
    event_source.subscribe("**", "_ws_broadcaster", _on_event)

    # ------------------------------------------------------------------
    # REST endpoints
    # ------------------------------------------------------------------

    @app.get("/healthz")
    def healthcheck():
        return {"ok": True}

    @app.post("/api/v1/entries", response_model=EntryResponse)
    def put_entry(
        req: PutRequest,
        api_key: Optional[str] = Header(None, alias="X-DMB-API-Key"),
    ):
        try:
            kwargs = dict(
                path=req.path,
                value=req.value,
                owner=req.owner,
                type=req.type,
                confidence=req.confidence,
                refs=req.refs,
                ttl=req.ttl,
                metadata=req.metadata,
            )
            if secure_db:
                kwargs["api_key"] = api_key
                kwargs["as_owner"] = req.as_owner
            entry = db.put(**kwargs)
        except Exception as exc:
            raise _handle_api_error(exc) from exc
        return _entry_to_response(entry)

    @app.get("/api/v1/entries", response_model=QueryResultResponse)
    def get_entries(
        scope: str = Query("**"),
        budget: int = Query(2000),
        query: Optional[str] = Query(None),
        owner: Optional[str] = Query(None),
        type: Optional[str] = Query(None),
        reader: Optional[str] = Query(None),
        api_key: Optional[str] = Header(None, alias="X-DMB-API-Key"),
    ):
        try:
            kwargs = dict(
                scope=scope,
                budget=budget,
                query=query,
                owner=owner,
                type=type,
                reader=reader,
            )
            if secure_db:
                kwargs["api_key"] = api_key
            result = db.get(**kwargs)
        except Exception as exc:
            raise _handle_api_error(exc) from exc
        return {
            "entries": [_entry_to_response(e) for e in result.entries],
            "total_matched": result.total_matched,
            "tokens_used": result.tokens_used,
            "budget_remaining": result.budget_remaining,
            "channel_used": result.channel_used.name,
            "text": result.text,
        }

    @app.get("/api/v1/entries/{path:path}")
    def get_entry_by_path(
        path: str,
        api_key: Optional[str] = Header(None, alias="X-DMB-API-Key"),
    ):
        try:
            if secure_db:
                entry = db.retrieve(path, api_key=api_key)
            else:
                entry = db.retrieve(path)
        except Exception as exc:
            raise _handle_api_error(exc) from exc
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Entry not found: {path}")
        return _entry_to_response(entry)

    @app.delete("/api/v1/entries/{path:path}")
    def delete_entry(
        path: str,
        api_key: Optional[str] = Header(None, alias="X-DMB-API-Key"),
    ):
        try:
            deleted = db.delete(path, api_key=api_key) if secure_db else db.delete(path)
        except Exception as exc:
            raise _handle_api_error(exc) from exc
        return {"deleted": deleted, "path": path}

    @app.get("/api/v1/status")
    def get_status(api_key: Optional[str] = Header(None, alias="X-DMB-API-Key")):
        try:
            content = db.status(api_key=api_key) if secure_db else db.status()
        except Exception as exc:
            raise _handle_api_error(exc) from exc
        return JSONResponse(content=content)

    @app.get("/api/v1/trust")
    def get_trust(api_key: Optional[str] = Header(None, alias="X-DMB-API-Key")):
        try:
            content = (
                db.agent_trust_report(api_key=api_key)
                if secure_db else db.agent_trust_report()
            )
        except Exception as exc:
            raise _handle_api_error(exc) from exc
        return JSONResponse(content=content)

    @app.get("/api/v1/topology")
    def get_topology(api_key: Optional[str] = Header(None, alias="X-DMB-API-Key")):
        try:
            content = (
                db.knowledge_topology(api_key=api_key)
                if secure_db else db.knowledge_topology()
            )
        except Exception as exc:
            raise _handle_api_error(exc) from exc
        return JSONResponse(content=content)

    @app.get("/api/v1/lineage/{path:path}")
    def get_lineage(
        path: str,
        api_key: Optional[str] = Header(None, alias="X-DMB-API-Key"),
    ):
        try:
            lineage = db.lineage(path, api_key=api_key) if secure_db else db.lineage(path)
        except Exception as exc:
            raise _handle_api_error(exc) from exc
        return {"path": path, "lineage": [str(n) for n in lineage]}

    @app.post("/api/v1/relate")
    def relate_entries(
        req: RelateRequest,
        api_key: Optional[str] = Header(None, alias="X-DMB-API-Key"),
    ):
        try:
            rel = (
                db.relate(req.path_a, req.path_b, api_key=api_key)
                if secure_db else db.relate(req.path_a, req.path_b)
            )
        except Exception as exc:
            raise _handle_api_error(exc) from exc
        if rel is None:
            raise HTTPException(
                status_code=400,
                detail="Entries not found or embeddings unavailable",
            )
        return {k: round(float(v), 6) for k, v in rel.items()}

    @app.post("/api/v1/compose")
    def compose_entries(
        req: ComposeRequest,
        api_key: Optional[str] = Header(None, alias="X-DMB-API-Key"),
    ):
        try:
            vec = (
                db.compose(req.paths, mode=req.mode, api_key=api_key)
                if secure_db else db.compose(req.paths, mode=req.mode)
            )
        except Exception as exc:
            raise _handle_api_error(exc) from exc
        if vec is None:
            raise HTTPException(
                status_code=400,
                detail="Could not compose — entries not found or embeddings unavailable",
            )
        nearest = db.materialize(vec, k=req.k)
        if secure_db:
            filtered = []
            agent_id = db.authenticate_agent(api_key)
            for path, similarity in nearest:
                try:
                    db.check_read_access(agent_id, path)
                except Exception:
                    continue
                filtered.append((path, similarity))
            nearest = filtered
        return {
            "nearest": [{"path": p, "similarity": round(float(s), 4)} for p, s in nearest],
            "mode": req.mode,
            "input_paths": req.paths,
        }

    # ------------------------------------------------------------------
    # Event history
    # ------------------------------------------------------------------

    @app.get("/api/v1/events")
    def get_events(
        limit: int = Query(50),
        pattern: Optional[str] = Query(None),
        api_key: Optional[str] = Header(None, alias="X-DMB-API-Key"),
    ):
        try:
            source = db._db if secure_db else db
            history = source.events.get_history(pattern=pattern, limit=limit)
            return [
                {
                    "type": e.type.value,
                    "path": e.path,
                    "data": e.data,
                    "source_owner": getattr(e, "source_owner", ""),
                    "timestamp": e.timestamp,
                }
                for e in history
            ]
        except Exception as exc:
            raise _handle_api_error(exc) from exc

    # ------------------------------------------------------------------
    # Dashboard (static files)
    # ------------------------------------------------------------------

    from pathlib import Path as _Path
    _static_dir = _Path(__file__).parent / "static"
    if _static_dir.is_dir():
        try:
            from fastapi.staticfiles import StaticFiles
            app.mount("/dashboard", StaticFiles(directory=str(_static_dir), html=True), name="dashboard")
        except Exception:
            # Fallback: serve index.html manually
            @app.get("/dashboard")
            def dashboard_index():
                index = _static_dir / "index.html"
                if index.exists():
                    return FileResponse(str(index))
                raise HTTPException(404, "Dashboard not found")

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    @app.websocket("/ws/subscribe")
    async def ws_subscribe(websocket: WebSocket):
        agent_id: Optional[str] = None
        api_key = websocket.query_params.get("api_key")
        if secure_db:
            try:
                agent_id = db.authenticate_agent(api_key)
            except Exception:
                await websocket.close(code=1008)
                return

        await ws_manager.connect(websocket, pattern="**", agent_id=agent_id)
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    payload = json.loads(data) if data else {}
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({"ack": False, "error": "Invalid JSON"}))
                    continue

                pattern = payload.get("pattern", "**")
                if secure_db:
                    try:
                        db.check_read_access(agent_id or "anonymous", pattern)
                    except Exception:
                        await websocket.send_text(
                            json.dumps({"ack": False, "error": "Access denied"})
                        )
                        continue

                await ws_manager.update_pattern(websocket, pattern)
                await websocket.send_text(json.dumps({"ack": True, "pattern": pattern}))
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)

    return app
