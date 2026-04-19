"""
Structured JSON logging for DimensionalBase server.

Provides request correlation IDs, JSON-formatted log output, and
context propagation via contextvars (works with both sync and async).
"""

from __future__ import annotations

import contextvars
import json
import logging
import uuid
from typing import Any, Dict


_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")
_agent_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("agent_id", default="")


def generate_request_id() -> str:
    """Generate a short, sortable request ID."""
    return uuid.uuid4().hex[:12]


def get_request_id() -> str:
    return _request_id_var.get()


def set_request_id(rid: str) -> None:
    _request_id_var.set(rid)


def set_agent_id(aid: str) -> None:
    _agent_id_var.set(aid)


class ContextFilter(logging.Filter):
    """Inject request_id and agent_id into all log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get()  # type: ignore[attr-defined]
        record.agent_id = _agent_id_var.get()  # type: ignore[attr-defined]
        return True


class JSONFormatter(logging.Formatter):
    """Emit structured JSON log lines (one per line)."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = getattr(record, "request_id", "")
        if rid:
            log_entry["request_id"] = rid
        aid = getattr(record, "agent_id", "")
        if aid:
            log_entry["agent_id"] = aid
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


def configure_logging(json_format: bool = True, level: int = logging.INFO) -> None:
    """Configure the root logger with structured output."""
    root = logging.getLogger()
    root.setLevel(level)

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.addFilter(ContextFilter())
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s [%(request_id)s] %(message)s"
        ))
    root.addHandler(handler)
