"""Pydantic request/response models for the REST API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "pydantic is required for the server. "
        "Install it with: pip install dimensionalbase[server]"
    )


class PutRequest(BaseModel):
    path: str
    value: str
    owner: str
    as_owner: Optional[str] = None
    type: str = "fact"
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    refs: List[str] = Field(default_factory=list)
    ttl: str = "session"
    metadata: Dict[str, str] = Field(default_factory=dict)


class GetParams(BaseModel):
    scope: str = "**"
    budget: int = 2000
    query: Optional[str] = None
    owner: Optional[str] = None
    type: Optional[str] = None
    reader: Optional[str] = None


class RelateRequest(BaseModel):
    path_a: str
    path_b: str


class ComposeRequest(BaseModel):
    paths: List[str] = Field(min_length=2)
    mode: str = "attentive"
    k: int = 5


class EntryResponse(BaseModel):
    id: str
    path: str
    value: str
    owner: str
    type: str
    confidence: float
    refs: List[str]
    version: int
    ttl: str
    created_at: float
    updated_at: float
    metadata: Dict[str, str]
    raw_score: float = 0.0
    score: float = 0.0


class QueryResultResponse(BaseModel):
    entries: List[EntryResponse]
    total_matched: int
    tokens_used: int
    budget_remaining: int
    channel_used: str
    text: str
