"""
CrewAI tool implementations for DimensionalBase.

Usage::

    from dimensionalbase import DimensionalBase
    from dimensionalbase.integrations.crewai import get_dimensionalbase_crew_tools

    db = DimensionalBase()
    tools = get_dimensionalbase_crew_tools(db)
    agent = Agent(role="researcher", tools=tools)
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

try:
    from crewai_tools import BaseTool as CrewAIBaseTool
except ImportError:
    raise ImportError(
        "crewai-tools is required for this integration. "
        "Install it with: pip install dimensionalbase[crewai]"
    )


class DimensionalBasePutTool(CrewAIBaseTool):
    """Write knowledge to the shared DimensionalBase store."""

    name: str = "DimensionalBase Put"
    description: str = (
        "Write knowledge to the shared DimensionalBase store. "
        "Input format: path|value|owner (e.g., 'task/auth/status|JWT expired|backend-agent')"
    )
    db: Any = None

    class Config:
        arbitrary_types_allowed = True

    def _run(self, input_text: str) -> str:
        parts = input_text.split("|", 2)
        if len(parts) < 3:
            return "Error: Input must be 'path|value|owner'"
        path, value, owner = parts[0].strip(), parts[1].strip(), parts[2].strip()
        entry = self.db.put(path=path, value=value, owner=owner)
        return f"Stored: {entry.path} v{entry.version}"


class DimensionalBaseGetTool(CrewAIBaseTool):
    """Read relevant knowledge from DimensionalBase."""

    name: str = "DimensionalBase Get"
    description: str = (
        "Read relevant knowledge from DimensionalBase within a token budget. "
        "Input: a search query or scope pattern (e.g., 'task/**' or 'What is blocking deployment?')"
    )
    db: Any = None

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str) -> str:
        # If it looks like a glob pattern, use as scope; otherwise as semantic query
        if "*" in query or "/" in query:
            result = self.db.get(scope=query, budget=2000)
        else:
            result = self.db.get(scope="**", budget=2000, query=query)
        return result.text if result.text else "(no entries found)"


class DimensionalBaseStatusTool(CrewAIBaseTool):
    """Get the status of the shared DimensionalBase."""

    name: str = "DimensionalBase Status"
    description: str = "Get the current status of the shared DimensionalBase."
    db: Any = None

    class Config:
        arbitrary_types_allowed = True

    def _run(self, _input: str = "") -> str:
        return json.dumps(self.db.status(), indent=2, default=str)


def get_dimensionalbase_crew_tools(db) -> List[CrewAIBaseTool]:
    """Get all DimensionalBase tools for CrewAI agents.

    Args:
        db: A ``DimensionalBase`` instance.

    Returns:
        List of CrewAI tools.
    """
    return [
        DimensionalBasePutTool(db=db),
        DimensionalBaseGetTool(db=db),
        DimensionalBaseStatusTool(db=db),
    ]
