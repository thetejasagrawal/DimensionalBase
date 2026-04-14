"""
HotPotQA benchmark evaluator for DimensionalBase.

Tests multi-hop reasoning: questions require reasoning across 2+ paragraphs.
DimensionalBase's reference graph and multi-entry retrieval should help
agents find the right supporting paragraphs.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from dimensionalbase import DimensionalBase
from benchmarks.standard.base_evaluator import BaseEvaluator


class HotPotQAEvaluator(BaseEvaluator):
    """Evaluate DimensionalBase on HotPotQA.

    Requires: ``pip install datasets`` for loading from HuggingFace.
    """

    def __init__(
        self,
        split: str = "validation",
        budgets: Optional[List[int]] = None,
        max_questions: Optional[int] = None,
    ) -> None:
        super().__init__(budgets=budgets, max_questions=max_questions)
        self.split = split

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load HotPotQA from HuggingFace datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required for HotPotQA. "
                "Install it with: pip install datasets"
            )

        ds = load_dataset("hotpot_qa", "fullwiki", split=self.split)
        examples = []
        for item in ds:
            # Combine all context paragraphs
            context_parts = []
            for title, sentences in zip(item.get("context", {}).get("title", []),
                                         item.get("context", {}).get("sentences", [])):
                context_parts.append(f"## {title}\n{''.join(sentences)}")

            examples.append({
                "id": item.get("id", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "context": "\n\n".join(context_parts),
                "supporting_facts": item.get("supporting_facts", {}),
                "type": item.get("type", ""),
                "level": item.get("level", ""),
            })
        return examples

    def ingest_document(self, db: DimensionalBase, example: Dict[str, Any]) -> int:
        """Ingest HotPotQA context paragraphs as individual entries with refs."""
        doc_id = example["id"][:8]
        context = example["context"]
        total_tokens = 0

        paragraphs = context.split("\n\n")
        paths = []

        for i, para in enumerate(paragraphs):
            if not para.strip():
                continue
            path = f"doc/{doc_id}/para/{i}"
            paths.append(path)

            # Add refs to adjacent paragraphs (building the reference graph)
            refs = []
            if i > 0:
                refs.append(f"doc/{doc_id}/para/{i-1}")

            db.put(
                path=path,
                value=para.strip(),
                owner="data-loader",
                type="fact",
                confidence=1.0,
                refs=refs,
                ttl="session",
            )
            total_tokens += len(para) // 4

        return total_tokens


def run_hotpotqa(
    max_questions: int = 10,
    budgets: Optional[List[int]] = None,
    output_path: str = "benchmarks/standard/results_hotpotqa.json",
):
    """Run the HotPotQA benchmark and save results."""
    evaluator = HotPotQAEvaluator(
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
    run_hotpotqa()
