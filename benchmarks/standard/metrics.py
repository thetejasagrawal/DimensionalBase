"""
Evaluation metrics for standard benchmarks.

Provides F1, exact match, and token efficiency scoring.
"""

from __future__ import annotations

import re
from typing import List, Set


def normalize_text(text: str) -> str:
    """Normalize text for comparison — lowercase, strip punctuation and articles."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, ground_truth: str) -> float:
    """Exact match after normalization. Returns 0.0 or 1.0."""
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score between prediction and ground truth."""
    pred_tokens = set(normalize_text(prediction).split())
    truth_tokens = set(normalize_text(ground_truth).split())

    if not pred_tokens or not truth_tokens:
        return 0.0

    common = pred_tokens & truth_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def token_efficiency(tokens_used: int, total_tokens: int) -> float:
    """Token efficiency: what fraction of the total was used."""
    if total_tokens == 0:
        return 0.0
    return 1.0 - (tokens_used / total_tokens)


def aggregate_metrics(results: List[dict]) -> dict:
    """Aggregate metrics across multiple benchmark results."""
    if not results:
        return {}

    n = len(results)
    avg_f1 = sum(r.get("f1", 0) for r in results) / n
    avg_em = sum(r.get("exact_match", 0) for r in results) / n
    avg_tokens = sum(r.get("tokens_used", 0) for r in results) / n
    avg_efficiency = sum(r.get("token_efficiency", 0) for r in results) / n

    return {
        "count": n,
        "avg_f1": round(avg_f1, 4),
        "avg_exact_match": round(avg_em, 4),
        "avg_tokens_used": round(avg_tokens, 1),
        "avg_token_efficiency": round(avg_efficiency, 4),
    }
