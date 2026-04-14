"""
LangChain BaseMemory implementation backed by DimensionalBase.

Usage::

    from dimensionalbase import DimensionalBase
    from dimensionalbase.integrations.langchain import DimensionalBaseMemory

    db = DimensionalBase()
    memory = DimensionalBaseMemory(db=db, scope="conversation/**")
    chain = ConversationChain(llm=llm, memory=memory)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from langchain_core.memory import BaseMemory
except ImportError:
    raise ImportError(
        "langchain-core is required for this integration. "
        "Install it with: pip install dimensionalbase[langchain]"
    )


class DimensionalBaseMemory(BaseMemory):
    """LangChain memory backed by DimensionalBase.

    On ``load_memory_variables``, queries the DB with the current user input
    as a semantic query — this gives budget-aware, relevance-ranked context
    instead of a flat conversation buffer.

    On ``save_context``, stores the AI's output as an observation.
    """

    db: Any  # DimensionalBase instance
    scope: str = "**"
    budget: int = 2000
    owner: str = "langchain-agent"
    memory_key: str = "context"

    class Config:
        arbitrary_types_allowed = True

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Query DimensionalBase for relevant context."""
        query = inputs.get("input") or inputs.get("question") or None
        result = self.db.get(
            scope=self.scope,
            budget=self.budget,
            query=query,
            reader=self.owner,
        )
        return {self.memory_key: result.text}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Store the AI output as an observation in DimensionalBase."""
        output_text = outputs.get("output") or outputs.get("text") or str(outputs)
        input_text = inputs.get("input") or inputs.get("question") or str(inputs)
        path = f"langchain/{self.owner}/turn/{hash(input_text) % 10000}"
        self.db.put(
            path=path,
            value=output_text,
            owner=self.owner,
            type="observation",
            ttl="session",
        )

    def clear(self) -> None:
        """Clear session-scoped entries."""
        self.db.clear_session()
