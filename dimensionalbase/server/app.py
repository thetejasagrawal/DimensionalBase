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
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger("dimensionalbase.server")

try:
    from fastapi import FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    from starlette.middleware.cors import CORSMiddleware
    from starlette.middleware.gzip import GZipMiddleware
except ImportError:
    raise ImportError(
        "fastapi is required for the server. "
        "Install it with: pip install dimensionalbase[server]"
    )

from dimensionalbase import __version__
from dimensionalbase.exceptions import (
    BudgetExhaustedError,
    ConflictError,
    DimensionalBaseError,
    EntryValidationError,
    RateLimitError as DMBRateLimitError,
)
from dimensionalbase.security.acl import AccessDeniedError
from dimensionalbase.security.auth import AuthError
from dimensionalbase.security.middleware import SecureDimensionalBase
from dimensionalbase.security.validation import ValidationError
from dimensionalbase.server.logging_config import get_request_id
from dimensionalbase.server.middleware import (
    BodySizeLimitMiddleware,
    RateLimiter,
    RateLimitMiddleware,
    RequestContextMiddleware,
    TimeoutMiddleware,
)
from dimensionalbase.server.models import (
    ComposeRequest,
    EntryResponse,
    PutRequest,
    QueryResultResponse,
    RelateRequest,
)
from dimensionalbase.server.ws import ConnectionManager


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
        "relevance_score": getattr(entry, "_score", 0.0),
    }


@asynccontextmanager
async def _lifespan(app: FastAPI):
    logger.info("DimensionalBase server starting")
    app.state.server_loop = asyncio.get_running_loop()
    yield
    logger.info("DimensionalBase server shutting down")
    subscription = getattr(app.state, "ws_event_subscription", None)
    event_source = getattr(app.state, "event_source", None)
    if subscription is not None and event_source is not None:
        try:
            event_source.unsubscribe(subscription)
        except Exception:
            logger.exception("Failed to remove WebSocket broadcast subscription cleanly")
    if hasattr(app.state, "ws_manager"):
        await app.state.ws_manager.close_all()
    if hasattr(app.state, "db"):
        app.state.db.close()
    logger.info("DimensionalBase server shutdown complete")


def create_app(db=None, server_config=None) -> FastAPI:
    """Create a FastAPI application wrapping a DimensionalBase instance."""
    if db is None:
        from dimensionalbase import DimensionalBase
        db = DimensionalBase()

    openapi_tags = [
        {"name": "Entries", "description": "Create, read, update, and delete knowledge entries"},
        {"name": "Algebra", "description": "Dimensional algebra operations (compose, relate)"},
        {"name": "Introspection", "description": "System status, trust reports, topology"},
        {"name": "Events", "description": "Event history and WebSocket subscriptions"},
        {"name": "Health", "description": "Health and readiness checks"},
    ]

    app = FastAPI(
        title="DimensionalBase API",
        description="REST API for the DimensionalBase protocol and database.",
        version=__version__,
        lifespan=_lifespan,
        openapi_tags=openapi_tags,
    )

    ws_manager = ConnectionManager()
    secure_db = isinstance(db, SecureDimensionalBase)

    cfg = dict(server_config or {})
    cors_origins = cfg.get("cors_origins", [])
    if isinstance(cors_origins, str):
        cors_origins = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]
    allow_all_origins = cors_origins == ["*"]

    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(TimeoutMiddleware, timeout_seconds=cfg.get("request_timeout_seconds", 30.0))
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_body_bytes=cfg.get("max_request_body_bytes", 1_048_576),
    )
    app.add_middleware(
        RateLimitMiddleware,
        limiter=RateLimiter(
            read_limit=cfg.get("rate_limit_read", 100.0),
            write_limit=cfg.get("rate_limit_write", 50.0),
        ),
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=None if allow_all_origins else cfg.get("cors_origin_regex"),
        allow_credentials=not allow_all_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    app.state.db = db
    app.state.ws_manager = ws_manager

    @app.exception_handler(DimensionalBaseError)
    async def dmb_exception_handler(request, exc):
        if isinstance(exc, AuthError):
            status, etype, code = 401, "auth_error", "invalid_api_key"
        elif isinstance(exc, AccessDeniedError):
            status, etype, code = 403, "permission_error", "access_denied"
        elif isinstance(exc, (ValidationError, EntryValidationError)):
            status, etype, code = 422, "validation_error", "entry_validation_failed"
        elif isinstance(exc, BudgetExhaustedError):
            status, etype, code = 400, "invalid_request", "budget_exhausted"
        elif isinstance(exc, ConflictError):
            status, etype, code = 409, "conflict", "knowledge_conflict"
        elif isinstance(exc, DMBRateLimitError):
            status, etype, code = 429, "rate_limit_exceeded", "rate_limit_exceeded"
        else:
            status, etype, code = 500, "internal_error", "unknown_error"
        rid = ""
        try:
            rid = get_request_id()
        except Exception:
            pass
        return JSONResponse(
            status_code=status,
            content={"error": {"type": etype, "code": code, "message": str(exc), "request_id": rid}},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request, exc):
        if isinstance(exc, HTTPException):
            raise exc
        logger.exception("Unhandled server error")
        rid = ""
        try:
            rid = get_request_id()
        except Exception:
            pass
        return JSONResponse(
            status_code=500,
            content={"error": {"type": "internal_error", "code": "internal_error", "message": "Internal server error", "request_id": rid}},
        )

    def _on_event(event) -> None:
        data = {
            "type": event.type.value,
            "path": event.path,
            "data": event.data,
            "source_owner": event.source_owner,
            "timestamp": event.timestamp,
        }
        loop = getattr(app.state, "server_loop", None)
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(
                lambda: loop.create_task(ws_manager.broadcast_event(data))
            )
            return

        try:
            asyncio.run(ws_manager.broadcast_event(data))
        except RuntimeError:
            logger.exception("Failed to broadcast WebSocket event")

    event_source = db._db if secure_db else db
    app.state.event_source = event_source
    app.state.ws_event_subscription = event_source.subscribe("**", "_ws_broadcaster", _on_event)

    @app.get("/healthz", tags=["Health"])
    def healthcheck():
        try:
            count = db.entry_count if not secure_db else db._db.entry_count
        except Exception:
            logger.exception("Health check failed")
            return JSONResponse(
                status_code=503,
                content={"ok": False, "version": __version__},
            )
        return {"ok": True, "version": __version__, "entries": count}

    @app.get("/readyz", tags=["Health"])
    def readiness_check():
        checks = {}
        overall = True
        try:
            count = db.entry_count if not secure_db else db._db.entry_count
            checks["database"] = {"status": "ok", "entries": count}
        except Exception as exc:
            checks["database"] = {"status": "error", "detail": str(exc)}
            overall = False
        raw_db = db._db if secure_db else db
        if raw_db.has_embeddings:
            checks["embeddings"] = {
                "status": "ok",
                "provider": raw_db._channels.embedding_provider.name,
            }
        else:
            checks["embeddings"] = {"status": "unavailable"}
        checks["websocket"] = {"status": "ok", "connections": ws_manager.connection_count}
        status_code = 200 if overall else 503
        return JSONResponse(
            content={"ok": overall, "version": __version__, "checks": checks},
            status_code=status_code,
        )

    @app.post("/api/v1/entries", response_model=EntryResponse, tags=["Entries"], status_code=201)
    def put_entry(
        req: PutRequest,
        api_key: Optional[str] = Header(None, alias="X-DMB-API-Key"),
    ):
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
        return _entry_to_response(entry)

    @app.get("/api/v1/entries", response_model=QueryResultResponse, tags=["Entries"])
    def get_entries(
        scope: str = Query("**"),
        budget: int = Query(2000),
        query: Optional[str] = Query(None),
        owner: Optional[str] = Query(None),
        type: Optional[str] = Query(None),
        reader: Optional[str] = Query(None),
        api_key: Optional[str] = Header(None, alias="X-DMB-API-Key"),
    ):
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
        return {
            "entries": [_entry_to_response(e) for e in result.entries],
            "total_matched": result.total_matched,
            "tokens_used": result.tokens_used,
            "budget_remaining": result.budget_remaining,
            "channel_used": result.channel_used.name,
            "text": result.text,
        }

    @app.get("/api/v1/entries/{path:path}", tags=["Entries"])
    def get_entry_by_path(
        path: str,
        api_key: Optional[str] = Header(None, alias="X-DMB-API-Key"),
    ):
        if secure_db:
            entry = db.retrieve(path, api_key=api_key)
        else:
            entry = db.retrieve(path)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Entry not found: {path}")
        return _entry_to_response(entry)

    @app.delete("/api/v1/entries/{path:path}", tags=["Entries"])
    def delete_entry(
        path: str,
        api_key: Optional[str] = Header(None, alias="X-DMB-API-Key"),
    ):
        deleted = db.delete(path, api_key=api_key) if secure_db else db.delete(path)
        return {"deleted": deleted, "path": path}

    @app.get("/api/v1/status", tags=["Introspection"])
    def get_status(api_key: Optional[str] = Header(None, alias="X-DMB-API-Key")):
        content = db.status(api_key=api_key) if secure_db else db.status()
        return JSONResponse(content=content)

    @app.get("/api/v1/trust", tags=["Introspection"])
    def get_trust(api_key: Optional[str] = Header(None, alias="X-DMB-API-Key")):
        content = (
            db.agent_trust_report(api_key=api_key)
            if secure_db else db.agent_trust_report()
        )
        return JSONResponse(content=content)

    @app.get("/api/v1/topology", tags=["Introspection"])
    def get_topology(api_key: Optional[str] = Header(None, alias="X-DMB-API-Key")):
        content = (
            db.knowledge_topology(api_key=api_key)
            if secure_db else db.knowledge_topology()
        )
        return JSONResponse(content=content)

    @app.get("/api/v1/lineage/{path:path}", tags=["Introspection"])
    def get_lineage(
        path: str,
        api_key: Optional[str] = Header(None, alias="X-DMB-API-Key"),
    ):
        lineage = db.lineage(path, api_key=api_key) if secure_db else db.lineage(path)
        return {"path": path, "lineage": [str(n) for n in lineage]}

    @app.post("/api/v1/relate", tags=["Algebra"])
    def relate_entries(
        req: RelateRequest,
        api_key: Optional[str] = Header(None, alias="X-DMB-API-Key"),
    ):
        rel = (
            db.relate(req.path_a, req.path_b, api_key=api_key)
            if secure_db else db.relate(req.path_a, req.path_b)
        )
        if rel is None:
            raise HTTPException(
                status_code=400,
                detail="Entries not found or embeddings unavailable",
            )
        return {k: round(float(v), 6) for k, v in rel.items()}

    @app.post("/api/v1/compose", tags=["Algebra"])
    def compose_entries(
        req: ComposeRequest,
        api_key: Optional[str] = Header(None, alias="X-DMB-API-Key"),
    ):
        vec = (
            db.compose(req.paths, mode=req.mode, api_key=api_key)
            if secure_db else db.compose(req.paths, mode=req.mode)
        )
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
                except AccessDeniedError:
                    continue
                filtered.append((path, similarity))
            nearest = filtered
        return {
            "nearest": [{"path": p, "similarity": round(float(s), 4)} for p, s in nearest],
            "mode": req.mode,
            "input_paths": req.paths,
        }

    @app.get("/api/v1/events", tags=["Events"])
    def get_events(
        limit: int = Query(50),
        pattern: Optional[str] = Query(None),
        api_key: Optional[str] = Header(None, alias="X-DMB-API-Key"),
    ):
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

    from pathlib import Path as _Path
    _static_dir = _Path(__file__).parent / "static"
    if _static_dir.is_dir():
        try:
            from fastapi.staticfiles import StaticFiles
            app.mount("/dashboard", StaticFiles(directory=str(_static_dir), html=True), name="dashboard")
        except Exception:
            @app.get("/dashboard")
            def dashboard_index():
                index = _static_dir / "index.html"
                if index.exists():
                    return FileResponse(str(index))
                raise HTTPException(404, "Dashboard not found")

    @app.websocket("/ws/subscribe")
    async def ws_subscribe(websocket: WebSocket):
        agent_id: Optional[str] = None
        api_key = websocket.query_params.get("api_key")
        if secure_db:
            try:
                agent_id = db.authenticate_agent(api_key)
            except AuthError:
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
                    except AccessDeniedError:
                        await websocket.send_text(
                            json.dumps({"ack": False, "error": "Access denied"})
                        )
                        continue

                await ws_manager.update_pattern(websocket, pattern)
                await websocket.send_text(json.dumps({"ack": True, "pattern": pattern}))
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)

    return app
