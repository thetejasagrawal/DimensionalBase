"""
LangChain BaseTool implementations for DimensionalBase.

Usage::

    from dimensionalbase import DimensionalBase
    from dimensionalbase.integrations.langchain import get_dimensionalbase_tools

    db = DimensionalBase()
    tools = get_dimensionalbase_tools(db)
    agent = initialize_agent(tools=tools, llm=llm)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

try:
    from langchain_core.tools import BaseTool
except ImportError:
    raise ImportError(
        "langchain-core is required for this integration. "
        "Install it with: pip install dimensionalbase[langchain]"
    )


class DimensionalBasePutTool(BaseTool):
    """Write knowledge to the shared DimensionalBase store."""

    name: str = "dimensionalbase_put"
    description: str = (
        "Write knowledge to the shared DimensionalBase store. "
        "Requires: path (hierarchical like 'task/auth/status'), value (the knowledge), "
        "owner (your agent name). Optional: type (fact/decision/plan/observation), "
        "confidence (0.0-1.0), refs (list of related paths)."
    )
    db: Any = None

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        path: str,
        value: str,
        owner: str = "langchain-agent",
        type: str = "fact",
        confidence: float = 1.0,
        refs: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        entry = self.db.put(
            path=path, value=value, owner=owner,
            type=type, confidence=confidence, refs=refs or [],
        )
        return f"Stored at {entry.path} (v{entry.version}, confidence={entry.confidence})"


class DimensionalBaseGetTool(BaseTool):
    """Read relevant knowledge from DimensionalBase within a token budget."""

    name: str = "dimensionalbase_get"
    description: str = (
        "Read relevant knowledge from the shared DimensionalBase within a token budget. "
        "Optional: scope (glob pattern like 'task/**'), budget (token limit, default 2000), "
        "query (semantic search query to boost relevant results)."
    )
    db: Any = None

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        scope: str = "**",
        budget: int = 2000,
        query: Optional[str] = None,
        **kwargs,
    ) -> str:
        result = self.db.get(scope=scope, budget=budget, query=query)
        return result.text if result.text else "(no entries found)"


class DimensionalBaseStatusTool(BaseTool):
    """Get the status of the shared DimensionalBase."""

    name: str = "dimensionalbase_status"
    description: str = (
        "Get the current status of the shared DimensionalBase — "
        "entry count, active channels, agent trust scores, and knowledge topology."
    )
    db: Any = None

    class Config:
        arbitrary_types_allowed = True

    def _run(self, **kwargs) -> str:
        return json.dumps(self.db.status(), indent=2, default=str)


def get_dimensionalbase_tools(db) -> List[BaseTool]:
    """Get all DimensionalBase tools for a LangChain agent.

    Args:
        db: A ``DimensionalBase`` instance.

    Returns:
        List of LangChain tools.
    """
    return [
        DimensionalBasePutTool(db=db),
        DimensionalBaseGetTool(db=db),
        DimensionalBaseStatusTool(db=db),
    ]
