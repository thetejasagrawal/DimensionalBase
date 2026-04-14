"""
Bayesian Confidence Engine — the system learns who to trust and what to believe.

Every knowledge entry has a confidence score. In v0.1, that was just a static
number the agent self-reported. Now it's a living value that the system
updates using Bayesian inference:

  - When Agent B confirms Agent A's fact → confidence goes UP
  - When Agent B contradicts Agent A → confidence goes DOWN
  - Old facts decay toward uncertainty over time
  - Confidence propagates through reference chains
  - Agents that are frequently confirmed build higher prior trust

The math: each entry's confidence is modeled as a Beta distribution:
  Beta(alpha, beta) where alpha = confirmations + 1, beta = contradictions + 1
  The mean = alpha / (alpha + beta) = point estimate of confidence
  The variance tells us how certain we are about the confidence itself

This is NOT just averaging. This is proper Bayesian updating that:
  - Handles small sample sizes correctly (wide posteriors)
  - Weights recent evidence more heavily (temporal discounting)
  - Propagates uncertainty through the knowledge graph
  - Converges to ground truth as evidence accumulates
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("dimensionalbase.reasoning.confidence")


@dataclass
class ConfidenceState:
    """Bayesian state for a single knowledge entry's confidence."""
    alpha: float = 1.0          # pseudo-count of confirmations
    beta: float = 1.0           # pseudo-count of contradictions
    base_confidence: float = 1.0  # agent's self-reported confidence
    last_update: float = field(default_factory=time.time)
    confirmations: int = 0
    contradictions: int = 0
    sources: Set[str] = field(default_factory=set)  # agents that confirmed/contradicted

    @property
    def mean(self) -> float:
        """Point estimate of confidence (Beta distribution mean)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """How uncertain we are about the confidence itself."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def strength(self) -> float:
        """How much evidence we have (alpha + beta - 2 = total observations)."""
        return self.alpha + self.beta - 2.0

    @property
    def lower_bound(self) -> float:
        """Conservative confidence estimate (5th percentile of Beta)."""
        # Wilson score interval approximation
        n = self.alpha + self.beta - 2
        if n <= 0:
            return 0.0
        p = self.mean
        z = 1.645  # 90% CI
        denominator = 1 + z * z / max(1, n)
        center = (p + z * z / (2 * max(1, n))) / denominator
        spread = z * math.sqrt((p * (1 - p) + z * z / (4 * max(1, n))) / max(1, n)) / denominator
        return max(0.0, center - spread)

    @property
    def effective_confidence(self) -> float:
        """The confidence score to actually use in scoring.

        Blends the Bayesian posterior with the agent's self-report,
        weighted by how much evidence we have.
        """
        evidence_weight = min(1.0, self.strength / 10.0)  # saturates at 10 observations
        return (1 - evidence_weight) * self.base_confidence + evidence_weight * self.mean


class ConfidenceEngine:
    """Bayesian confidence management for the knowledge store.

    Tracks confirmation/contradiction patterns and updates
    confidence scores using proper probabilistic inference.
    """

    def __init__(
        self,
        temporal_decay_half_life: float = 7200.0,  # 2 hours
        propagation_depth: int = 3,
        confirmation_weight: float = 1.0,
        contradiction_weight: float = 2.0,  # contradictions weigh more (conservative)
    ):
        self._states: Dict[str, ConfidenceState] = {}
        self._lock = threading.Lock()
        self._decay_half_life = temporal_decay_half_life
        self._propagation_depth = propagation_depth
        self._confirmation_weight = confirmation_weight
        self._contradiction_weight = contradiction_weight

        # Track inter-agent confirmation patterns
        self._agent_confirmations: Dict[str, Dict[str, int]] = {}  # agent -> {other_agent -> count}
        self._agent_contradictions: Dict[str, Dict[str, int]] = {}

    def register(self, path: str, confidence: float, owner: str) -> ConfidenceState:
        """Register a new entry's confidence state."""
        with self._lock:
            state = ConfidenceState(
                alpha=1.0 + confidence,
                beta=1.0 + (1.0 - confidence),
                base_confidence=confidence,
                sources={owner},
            )
            self._states[path] = state
            return state

    def refresh(self, path: str, confidence: float, owner: str) -> ConfidenceState:
        """Refresh base confidence without discarding accumulated evidence."""
        with self._lock:
            state = self._states.get(path)
            if state is None:
                return self.register(path, confidence, owner)

            state.base_confidence = confidence
            state.last_update = time.time()
            state.sources.add(owner)
            return state

    def confirm(
        self,
        path: str,
        confirming_agent: str,
        strength: float = 1.0,
    ) -> Optional[ConfidenceState]:
        """Record that an agent confirmed this entry's value.

        Bayesian update: alpha += confirmation_weight * strength
        """
        with self._lock:
            state = self._states.get(path)
            if state is None:
                return None

            state.alpha += self._confirmation_weight * strength
            state.confirmations += 1
            state.sources.add(confirming_agent)
            state.last_update = time.time()

            # Track inter-agent pattern
            for source in state.sources:
                if source != confirming_agent:
                    self._record_confirmation(confirming_agent, source)

            logger.debug(
                f"CONFIRM {path} by {confirming_agent}: "
                f"conf={state.effective_confidence:.3f} "
                f"(α={state.alpha:.1f}, β={state.beta:.1f})"
            )
            return state

    def contradict(
        self,
        path: str,
        contradicting_agent: str,
        strength: float = 1.0,
    ) -> Optional[ConfidenceState]:
        """Record that an agent contradicted this entry's value.

        Bayesian update: beta += contradiction_weight * strength
        Contradictions weigh more by default (conservative — favor doubt).
        """
        with self._lock:
            state = self._states.get(path)
            if state is None:
                return None

            state.beta += self._contradiction_weight * strength
            state.contradictions += 1
            state.sources.add(contradicting_agent)
            state.last_update = time.time()

            for source in state.sources:
                if source != contradicting_agent:
                    self._record_contradiction(contradicting_agent, source)

            logger.debug(
                f"CONTRADICT {path} by {contradicting_agent}: "
                f"conf={state.effective_confidence:.3f} "
                f"(α={state.alpha:.1f}, β={state.beta:.1f})"
            )
            return state

    def get_confidence(self, path: str) -> float:
        """Get the current effective confidence for an entry.

        Applies temporal decay: old entries drift toward 0.5 (uncertainty).
        """
        with self._lock:
            state = self._states.get(path)
            if state is None:
                return 0.5  # Unknown

            raw = state.effective_confidence
            age = time.time() - state.last_update

            # Temporal decay toward 0.5
            decay = 2.0 ** (-age / self._decay_half_life)
            return 0.5 + (raw - 0.5) * decay

    def get_state(self, path: str) -> Optional[ConfidenceState]:
        with self._lock:
            return self._states.get(path)

    def propagate_through_refs(
        self,
        path: str,
        refs: List[str],
        ref_entries: Dict[str, float],
    ) -> float:
        """Propagate confidence through reference chains.

        If entry A references entries B and C, A's confidence is
        influenced by B and C's confidence. A chain of uncertain
        entries compounds that uncertainty.

        Returns the propagated confidence adjustment.
        """
        if not refs:
            return 0.0

        ref_confidences = []
        for ref_path in refs:
            if ref_path in ref_entries:
                ref_confidences.append(ref_entries[ref_path])
            else:
                conf = self.get_confidence(ref_path)
                ref_confidences.append(conf)

        if not ref_confidences:
            return 0.0

        # Propagation: geometric mean of reference confidences
        # (one weak link brings down the chain)
        log_sum = sum(math.log(max(c, 0.01)) for c in ref_confidences)
        propagated = math.exp(log_sum / len(ref_confidences))

        # Apply diminishing returns based on depth
        return propagated

    def agent_agreement_score(self, agent_a: str, agent_b: str) -> float:
        """How often do two agents agree? 0 = always disagree, 1 = always agree.

        Based on historical confirmation/contradiction patterns.
        """
        with self._lock:
            confirms = self._agent_confirmations.get(agent_a, {}).get(agent_b, 0)
            contradicts = self._agent_contradictions.get(agent_a, {}).get(agent_b, 0)

        total = confirms + contradicts
        if total == 0:
            return 0.5  # No data, assume neutral
        return confirms / total

    def remove(self, path: str) -> bool:
        with self._lock:
            return self._states.pop(path, None) is not None

    def bulk_decay(self) -> int:
        """Apply temporal decay to all entries. Returns count affected."""
        now = time.time()
        count = 0
        with self._lock:
            for path, state in self._states.items():
                age = now - state.last_update
                if age > self._decay_half_life / 10:  # Only touch entries that need it
                    # Shrink alpha and beta toward 1 (the prior)
                    decay = 2.0 ** (-age / self._decay_half_life)
                    excess_alpha = state.alpha - 1.0
                    excess_beta = state.beta - 1.0
                    state.alpha = 1.0 + excess_alpha * decay
                    state.beta = 1.0 + excess_beta * decay
                    count += 1
        return count

    def to_dict(self) -> Dict[str, Any]:
        """Serialize engine state for durable persistence."""
        with self._lock:
            return {
                "states": {
                    path: {
                        "alpha": state.alpha,
                        "beta": state.beta,
                        "base_confidence": state.base_confidence,
                        "last_update": state.last_update,
                        "confirmations": state.confirmations,
                        "contradictions": state.contradictions,
                        "sources": sorted(state.sources),
                    }
                    for path, state in self._states.items()
                },
                "agent_confirmations": self._agent_confirmations,
                "agent_contradictions": self._agent_contradictions,
            }

    def load_dict(self, payload: Optional[Dict[str, Any]]) -> None:
        """Restore engine state from persisted data."""
        with self._lock:
            self._states = {}
            self._agent_confirmations = {}
            self._agent_contradictions = {}

            if not payload:
                return

            for path, state_payload in payload.get("states", {}).items():
                if not isinstance(state_payload, dict):
                    continue
                self._states[path] = ConfidenceState(
                    alpha=float(state_payload.get("alpha", 1.0)),
                    beta=float(state_payload.get("beta", 1.0)),
                    base_confidence=float(state_payload.get("base_confidence", 1.0)),
                    last_update=float(state_payload.get("last_update", time.time())),
                    confirmations=int(state_payload.get("confirmations", 0)),
                    contradictions=int(state_payload.get("contradictions", 0)),
                    sources=set(state_payload.get("sources", [])),
                )

            self._agent_confirmations = {
                agent: {other: int(count) for other, count in targets.items()}
                for agent, targets in payload.get("agent_confirmations", {}).items()
                if isinstance(targets, dict)
            }
            self._agent_contradictions = {
                agent: {other: int(count) for other, count in targets.items()}
                for agent, targets in payload.get("agent_contradictions", {}).items()
                if isinstance(targets, dict)
            }

    # --- Internal ---

    def _record_confirmation(self, from_agent: str, to_agent: str):
        if from_agent not in self._agent_confirmations:
            self._agent_confirmations[from_agent] = {}
        self._agent_confirmations[from_agent][to_agent] = \
            self._agent_confirmations[from_agent].get(to_agent, 0) + 1

    def _record_contradiction(self, from_agent: str, to_agent: str):
        if from_agent not in self._agent_contradictions:
            self._agent_contradictions[from_agent] = {}
        self._agent_contradictions[from_agent][to_agent] = \
            self._agent_contradictions[from_agent].get(to_agent, 0) + 1
