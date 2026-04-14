"""
LongBench v2 benchmark evaluator for DimensionalBase.

Compares DimensionalBase's budget-aware retrieval against full-context
and naive RAG baselines on long-document reading comprehension.

This directly competes with Ramp Labs' Latent Briefing results.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from dimensionalbase import DimensionalBase
from benchmarks.standard.base_evaluator import BaseEvaluator


class LongBenchV2Evaluator(BaseEvaluator):
    """Evaluate DimensionalBase on LongBench v2.

    Requires: ``pip install datasets`` for loading from HuggingFace.
    """

    def __init__(
        self,
        subset: str = "all",
        budgets: Optional[List[int]] = None,
        max_questions: Optional[int] = None,
        chunk_size: int = 500,
    ) -> None:
        super().__init__(budgets=budgets, max_questions=max_questions)
        self.subset = subset
        self.chunk_size = chunk_size

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load LongBench v2 from HuggingFace datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required for LongBench v2. "
                "Install it with: pip install datasets"
            )

        ds = load_dataset("THUDM/LongBench-v2", split="test")
        examples = []
        for item in ds:
            examples.append({
                "id": str(item.get("id", hashlib.md5(str(item).encode()).hexdigest()[:8])),
                "context": item.get("context", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "length": item.get("length", ""),
                "difficulty": item.get("difficulty", ""),
            })
        return examples

    def ingest_document(self, db: DimensionalBase, example: Dict[str, Any]) -> int:
        """Split document into chunks and write as KnowledgeEntries."""
        doc_id = example["id"]
        context = example["context"]
        total_tokens = 0

        # Split into chunks by paragraph, then by size
        paragraphs = context.split("\n\n")
        chunk_idx = 0
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) > self.chunk_size * 4:  # ~4 chars per token
                if current_chunk.strip():
                    path = f"doc/{doc_id}/chunk/{chunk_idx}"
                    db.put(
                        path=path,
                        value=current_chunk.strip(),
                        owner="data-loader",
                        type="fact",
                        confidence=1.0,
                        ttl="session",
                    )
                    total_tokens += len(current_chunk) // 4
                    chunk_idx += 1
                current_chunk = para
            else:
                current_chunk += "\n\n" + para

        # Final chunk
        if current_chunk.strip():
            path = f"doc/{doc_id}/chunk/{chunk_idx}"
            db.put(
                path=path,
                value=current_chunk.strip(),
                owner="data-loader",
                type="fact",
                confidence=1.0,
                ttl="session",
            )
            total_tokens += len(current_chunk) // 4

        return total_tokens


def run_longbench_v2(
    max_questions: int = 10,
    budgets: Optional[List[int]] = None,
    output_path: str = "benchmarks/standard/results_longbench_v2.json",
):
    """Run the LongBench v2 benchmark and save results."""
    evaluator = LongBenchV2Evaluator(
        max_questions=max_questions,
        budgets=budgets or [500, 1000, 2000, 4000],
    )
    evaluator.evaluate(method_name="dimensionalbase")
    evaluator.save_results(output_path)
    print(f"\nResults saved to {output_path}")
    print("\nSummary by budget:")
    for budget, metrics in evaluator.summary().items():
        print(f"  Budget {budget}: F1={metrics['avg_f1']:.3f}, "
              f"Efficiency={metrics['avg_token_efficiency']:.3f}")
    return evaluator


if __name__ == "__main__":
    run_longbench_v2()
