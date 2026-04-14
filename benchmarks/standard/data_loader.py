"""
Dataset loading utilities for standard benchmarks.

Handles downloading, caching, and formatting benchmark datasets.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional


def load_longbench_v2(max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load LongBench v2 dataset.

    Requires: ``pip install datasets``
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    ds = load_dataset("THUDM/LongBench-v2", split="test")
    examples = []
    for item in ds:
        examples.append({
            "id": str(item.get("id", "")),
            "context": item.get("context", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "length": item.get("length", ""),
            "difficulty": item.get("difficulty", ""),
        })
        if max_examples and len(examples) >= max_examples:
            break
    return examples


def load_hotpotqa(
    split: str = "validation",
    max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load HotPotQA dataset.

    Requires: ``pip install datasets``
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    ds = load_dataset("hotpot_qa", "fullwiki", split=split)
    examples = []
    for item in ds:
        context_parts = []
        titles = item.get("context", {}).get("title", [])
        sentences = item.get("context", {}).get("sentences", [])
        for title, sents in zip(titles, sentences):
            context_parts.append(f"## {title}\n{''.join(sents)}")

        examples.append({
            "id": item.get("id", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "context": "\n\n".join(context_parts),
            "type": item.get("type", ""),
            "level": item.get("level", ""),
        })
        if max_examples and len(examples) >= max_examples:
            break
    return examples
