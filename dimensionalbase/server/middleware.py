"""
FastAPI/Starlette middleware for DimensionalBase server.

Provides: request context (correlation IDs), rate limiting, request timeouts.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Tuple

from dimensionalbase.server.logging_config import (
    generate_request_id,
    set_agent_id,
    set_request_id,
)

logger = logging.getLogger("dimensionalbase.server.middleware")

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse
except ImportError:
    raise ImportError("starlette is required. Install with: pip install dimensionalbase[server]")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request ID, logs request start/end, sets X-Request-ID header."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or generate_request_id()
        set_request_id(request_id)

        api_key = request.headers.get("x-dmb-api-key")
        if api_key:
            set_agent_id(api_key[:8] + "...")

        start = time.perf_counter()
        logger.info("Request started: %s %s", request.method, request.url.path)

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "Request completed: %s %s -> %d (%.1fms)",
            request.method, request.url.path, response.status_code, duration_ms,
        )
        return response


class _RequestTooLargeError(Exception):
    """Internal signal for oversized request bodies."""


class RateLimiter:
    """In-memory token bucket rate limiter per API key / IP."""

    def __init__(
        self,
        read_limit: float = 100.0,
        write_limit: float = 50.0,
        window_seconds: float = 1.0,
    ):
        self._read_limit = read_limit
        self._write_limit = write_limit
        self._window = window_seconds
        self._buckets: Dict[str, Tuple[float, float]] = {}

    def allow(self, key: str, is_write: bool = False) -> Tuple[bool, Dict[str, str]]:
        """Check if a request is allowed. Returns (allowed, headers_dict)."""
        limit = self._write_limit if is_write else self._read_limit
        now = time.time()

        if len(self._buckets) > 10000:
            cutoff = now - self._window * 10
            self._buckets = {
                bucket_key: bucket
                for bucket_key, bucket in self._buckets.items()
                if bucket[1] > cutoff
            }

        if key not in self._buckets:
            self._buckets[key] = (limit, now)

        tokens, last_refill = self._buckets[key]
        elapsed = now - last_refill
        tokens = min(limit, tokens + elapsed * (limit / self._window))

        headers = {
            "X-RateLimit-Limit": str(int(limit)),
            "X-RateLimit-Remaining": str(max(0, int(tokens - 1))),
            "X-RateLimit-Reset": str(int(now + self._window)),
        }

        if tokens >= 1.0:
            self._buckets[key] = (tokens - 1.0, now)
            return True, headers

        self._buckets[key] = (tokens, now)
        headers["Retry-After"] = str(int(self._window))
        return False, headers


class BodySizeLimitMiddleware:
    """Reject request bodies that exceed a configured byte threshold."""

    def __init__(self, app, max_body_bytes: int = 1_048_576):
        self.app = app
        self._max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or self._max_body_bytes <= 0:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length.decode()) > self._max_body_bytes:
                    response = JSONResponse(
                        status_code=413,
                        content={
                            "error": {
                                "type": "payload_too_large",
                                "code": "payload_too_large",
                                "message": f"Request body exceeds {self._max_body_bytes} bytes",
                            }
                        },
                    )
                    await response(scope, receive, send)
                    return
            except ValueError:
                logger.warning("Ignoring invalid Content-Length header")

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._max_body_bytes:
                    raise _RequestTooLargeError
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestTooLargeError:
            response = JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "type": "payload_too_large",
                        "code": "payload_too_large",
                        "message": f"Request body exceeds {self._max_body_bytes} bytes",
                    }
                },
            )
            await response(scope, receive, send)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforce rate limits on all API endpoints."""

    def __init__(self, app, limiter: RateLimiter):
        super().__init__(app)
        self._limiter = limiter

    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/healthz", "/readyz"):
            return await call_next(request)

        key = (
            request.headers.get("x-dmb-api-key")
            or (request.client.host if request.client else None)
            or "unknown"
        )
        is_write = request.method in ("POST", "PUT", "DELETE", "PATCH")

        allowed, headers = self._limiter.allow(key, is_write=is_write)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "type": "rate_limit_exceeded",
                        "code": "rate_limit_exceeded",
                        "message": "Too many requests. Please retry after the Retry-After period.",
                    }
                },
                headers=headers,
            )

        response = await call_next(request)
        for header, value in headers.items():
            response.headers[header] = value
        return response


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Enforce a maximum request duration."""

    def __init__(self, app, timeout_seconds: float = 30.0):
        super().__init__(app)
        self._timeout = timeout_seconds

    async def dispatch(self, request: Request, call_next):
        try:
            return await asyncio.wait_for(
                call_next(request), timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={
                    "error": {
                        "type": "timeout",
                        "code": "request_timeout",
                        "message": f"Request timed out after {self._timeout}s",
                    }
                },
            )
