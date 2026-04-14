"""
Base evaluator for standard benchmarks.

Provides the shared infrastructure for ingesting documents into DimensionalBase,
querying with budget constraints, and scoring against ground truth.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dimensionalbase import DimensionalBase
from benchmarks.standard.metrics import exact_match, f1_score, token_efficiency


@dataclass
class BenchmarkResult:
    """Result of evaluating one question."""
    question_id: str
    question: str
    ground_truth: str
    prediction: str
    f1: float
    exact_match: float
    tokens_used: int
    total_tokens: int
    token_efficiency: float
    retrieval_time_ms: float
    method: str
    budget: int


class BaseEvaluator(ABC):
    """Abstract base class for benchmark evaluators.

    Subclasses implement ``load_dataset`` and ``ingest_document`` for
    their specific benchmark format.
    """

    def __init__(
        self,
        budgets: Optional[List[int]] = None,
        max_questions: Optional[int] = None,
    ) -> None:
        self.budgets = budgets or [500, 1000, 2000, 4000]
        self.max_questions = max_questions
        self.results: List[BenchmarkResult] = []

    @abstractmethod
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the benchmark dataset. Returns list of examples."""

    @abstractmethod
    def ingest_document(self, db: DimensionalBase, example: Dict[str, Any]) -> int:
        """Ingest a single document into DimensionalBase. Returns token count."""

    def evaluate(
        self,
        predict_fn=None,
        method_name: str = "dimensionalbase",
    ) -> List[BenchmarkResult]:
        """Run the full benchmark evaluation.

        Args:
            predict_fn: Function(context, question) -> answer. If None, returns context as-is.
            method_name: Name of the method being evaluated.

        Returns:
            List of BenchmarkResult for each (question, budget) pair.
        """
        dataset = self.load_dataset()
        if self.max_questions:
            dataset = dataset[:self.max_questions]

        results = []
        for example in dataset:
            db = DimensionalBase()
            total_tokens = self.ingest_document(db, example)
            question = example.get("question", "")
            ground_truth = example.get("answer", "")

            for budget in self.budgets:
                start = time.time()
                qr = db.get(scope="**", budget=budget, query=question)
                retrieval_ms = (time.time() - start) * 1000

                context = qr.text
                if predict_fn:
                    prediction = predict_fn(context, question)
                else:
                    prediction = context

                f1 = f1_score(prediction, ground_truth)
                em = exact_match(prediction, ground_truth)
                eff = token_efficiency(qr.tokens_used, total_tokens)

                result = BenchmarkResult(
                    question_id=example.get("id", ""),
                    question=question,
                    ground_truth=ground_truth,
                    prediction=prediction,
                    f1=f1,
                    exact_match=em,
                    tokens_used=qr.tokens_used,
                    total_tokens=total_tokens,
                    token_efficiency=eff,
                    retrieval_time_ms=retrieval_ms,
                    method=method_name,
                    budget=budget,
                )
                results.append(result)

            db.close()

        self.results = results
        return results

    def summary(self) -> Dict[str, Any]:
        """Generate a summary of results grouped by budget."""
        from benchmarks.standard.metrics import aggregate_metrics

        by_budget: Dict[int, List[dict]] = {}
        for r in self.results:
            by_budget.setdefault(r.budget, []).append({
                "f1": r.f1,
                "exact_match": r.exact_match,
                "tokens_used": r.tokens_used,
                "token_efficiency": r.token_efficiency,
            })

        return {
            budget: aggregate_metrics(results)
            for budget, results in sorted(by_budget.items())
        }

    def save_results(self, path: str) -> None:
        """Save results to a JSON file."""
        data = {
            "results": [
                {
                    "question_id": r.question_id,
                    "method": r.method,
                    "budget": r.budget,
                    "f1": r.f1,
                    "exact_match": r.exact_match,
                    "tokens_used": r.tokens_used,
                    "token_efficiency": r.token_efficiency,
                    "retrieval_time_ms": round(r.retrieval_time_ms, 2),
                }
                for r in self.results
            ],
            "summary": self.summary(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
