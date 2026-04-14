"""
Agent Trust Engine — the system learns which agents are reliable.

Every agent that writes to DimensionalBase builds a trust profile.
Trust is earned, not declared:

  - Agents whose facts get confirmed earn higher trust
  - Agents whose facts get contradicted lose trust
  - Trust is domain-specific (an agent might be reliable for code but not business)
  - Recent performance is weighted more heavily (trust decays and rebuilds)
  - Trust feeds back into the scoring engine (higher-trust agents' entries rank higher)

The model is inspired by:
  - Elo rating systems (relative performance, not absolute)
  - PageRank (trust from trusted agents matters more)
  - Bayesian bandits (exploration vs. exploitation of agent reliability)

This is NOT a simple counter. This is a multi-dimensional trust model
that captures the nuanced reality of multi-agent collaboration.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("dimensionalbase.trust")

# Base trust for a new agent
DEFAULT_TRUST = 0.5
# How fast trust changes (higher = more reactive, lower = more stable)
TRUST_K_FACTOR = 32.0
# Decay half-life for trust evidence (recent matters more)
TRUST_DECAY_HALF_LIFE = 86400.0  # 24 hours
# Minimum interactions before trust score is considered reliable
MIN_INTERACTIONS_FOR_RELIABLE = 5


@dataclass
class TrustEvent:
    """A single trust-relevant event."""
    timestamp: float
    event_type: str       # "confirmation", "contradiction", "gap_fill", "stale_update"
    other_agent: str      # The agent involved
    domain: str           # Path prefix (domain of the interaction)
    magnitude: float      # How significant (0-1)


@dataclass
class AgentProfile:
    """Complete trust profile for a single agent."""
    agent_id: str
    global_trust: float = DEFAULT_TRUST
    domain_trust: Dict[str, float] = field(default_factory=dict)  # domain -> trust
    total_entries: int = 0
    total_confirmations: int = 0
    total_contradictions: int = 0
    total_interactions: int = 0
    first_seen: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    events: List[TrustEvent] = field(default_factory=list)
    _max_events: int = field(default=500, repr=False)

    @property
    def confirmation_rate(self) -> float:
        """Fraction of interactions that were confirmations."""
        total = self.total_confirmations + self.total_contradictions
        if total == 0:
            return 0.5
        return self.total_confirmations / total

    @property
    def is_reliable(self) -> bool:
        """Whether we have enough data to consider this agent's trust reliable."""
        return self.total_interactions >= MIN_INTERACTIONS_FOR_RELIABLE

    @property
    def activity_level(self) -> float:
        """How active this agent has been recently (0-1)."""
        if self.total_entries == 0:
            return 0.0
        age = time.time() - self.first_seen
        if age < 1:
            return 1.0
        recency = time.time() - self.last_active
        return math.exp(-recency / 3600)  # Decay over 1 hour

    def trust_for_domain(self, domain: str) -> float:
        """Get trust score for a specific domain.

        Falls back to global trust if no domain-specific data.
        Blends domain and global based on how much domain data exists.
        """
        if domain not in self.domain_trust:
            return self.global_trust

        domain_score = self.domain_trust[domain]
        # Count domain-specific events
        domain_events = sum(1 for e in self.events if e.domain == domain)
        domain_weight = min(1.0, domain_events / 10.0)

        return domain_weight * domain_score + (1 - domain_weight) * self.global_trust

    def add_event(self, event: TrustEvent):
        """Add a trust event, maintaining max history."""
        self.events.append(event)
        if len(self.events) > self._max_events:
            self.events = self.events[-self._max_events:]
        self.last_active = event.timestamp


class AgentTrustEngine:
    """Manages trust scores for all agents in the system.

    Trust updating uses an Elo-like system:
      - Each agent has a rating
      - When agent A confirms agent B, both gain trust
      - When agent A contradicts agent B, the higher-confidence
        agent gains trust and the other loses it
      - Trust changes are proportional to surprise (unexpected
        confirmation/contradiction changes trust more)
    """

    def __init__(
        self,
        k_factor: float = TRUST_K_FACTOR,
        decay_half_life: float = TRUST_DECAY_HALF_LIFE,
        default_trust: float = DEFAULT_TRUST,
        min_interactions: int = MIN_INTERACTIONS_FOR_RELIABLE,
        pagerank_max_iterations: int = 20,
        pagerank_epsilon: float = 1e-6,
    ):
        self._profiles: Dict[str, AgentProfile] = {}
        self._lock = threading.Lock()
        self._k_factor = k_factor
        self._decay_half_life = decay_half_life
        self._default_trust = default_trust
        self._min_interactions = min_interactions
        self._pagerank_max_iterations = pagerank_max_iterations
        self._pagerank_epsilon = pagerank_epsilon

        # Trust graph: who trusts whom
        self._trust_graph: Dict[str, Dict[str, float]] = {}

    def get_or_create_profile(self, agent_id: str) -> AgentProfile:
        """Get an agent's profile, creating if needed."""
        with self._lock:
            if agent_id not in self._profiles:
                self._profiles[agent_id] = AgentProfile(
                    agent_id=agent_id,
                    global_trust=self._default_trust,
                )
            return self._profiles[agent_id]

    def record_entry(self, agent_id: str) -> None:
        """Record that an agent wrote an entry."""
        profile = self.get_or_create_profile(agent_id)
        with self._lock:
            profile.total_entries += 1
            profile.last_active = time.time()

    def record_confirmation(
        self,
        confirming_agent: str,
        confirmed_agent: str,
        domain: str,
        confidence_of_confirmed: float = 0.5,
    ) -> Tuple[float, float]:
        """Record that one agent confirmed another's knowledge.

        Returns (confirming_agent_new_trust, confirmed_agent_new_trust).
        """
        confirmer = self.get_or_create_profile(confirming_agent)
        confirmed = self.get_or_create_profile(confirmed_agent)

        with self._lock:
            # Elo-like update
            # Expected confirmation probability based on trust difference
            trust_diff = confirmed.global_trust - 0.5
            expected = 1.0 / (1.0 + math.exp(-trust_diff * 4))  # Sigmoid

            # Surprise factor: unexpected confirmation changes trust more
            surprise = 1.0 - expected

            # Update trust scores
            delta = self._k_factor * surprise / 100.0
            confirmed.global_trust = min(1.0, confirmed.global_trust + delta)
            confirmer.global_trust = min(1.0, confirmer.global_trust + delta * 0.5)

            # Domain-specific trust
            if domain not in confirmed.domain_trust:
                confirmed.domain_trust[domain] = DEFAULT_TRUST
            confirmed.domain_trust[domain] = min(1.0, confirmed.domain_trust[domain] + delta)

            # Record events
            now = time.time()
            confirmed.total_confirmations += 1
            confirmed.total_interactions += 1
            confirmer.total_interactions += 1

            confirmed.add_event(TrustEvent(
                timestamp=now,
                event_type="confirmation",
                other_agent=confirming_agent,
                domain=domain,
                magnitude=delta,
            ))

            # Update trust graph
            if confirming_agent not in self._trust_graph:
                self._trust_graph[confirming_agent] = {}
            current = self._trust_graph[confirming_agent].get(confirmed_agent, 0.5)
            self._trust_graph[confirming_agent][confirmed_agent] = min(1.0, current + 0.05)

            return confirmer.global_trust, confirmed.global_trust

    def record_contradiction(
        self,
        contradicting_agent: str,
        contradicted_agent: str,
        domain: str,
        confidence_of_contradicting: float = 0.5,
        confidence_of_contradicted: float = 0.5,
    ) -> Tuple[float, float]:
        """Record that one agent contradicted another.

        The agent with higher confidence gains trust; the other loses it.
        If confidences are equal, both lose a small amount (uncertainty).

        Returns (contradicting_agent_new_trust, contradicted_agent_new_trust).
        """
        contradicter = self.get_or_create_profile(contradicting_agent)
        contradicted = self.get_or_create_profile(contradicted_agent)

        with self._lock:
            delta = self._k_factor / 100.0

            if confidence_of_contradicting > confidence_of_contradicted:
                # Contradicting agent is more confident — they gain, other loses
                contradicter.global_trust = min(1.0, contradicter.global_trust + delta * 0.3)
                contradicted.global_trust = max(0.0, contradicted.global_trust - delta)
            elif confidence_of_contradicted > confidence_of_contradicting:
                # Contradicted agent is more confident — they keep trust
                contradicter.global_trust = max(0.0, contradicter.global_trust - delta)
                contradicted.global_trust = min(1.0, contradicted.global_trust + delta * 0.3)
            else:
                # Equal confidence — both lose slightly (ambiguity penalty)
                contradicter.global_trust = max(0.0, contradicter.global_trust - delta * 0.2)
                contradicted.global_trust = max(0.0, contradicted.global_trust - delta * 0.2)

            # Domain trust
            if domain not in contradicted.domain_trust:
                contradicted.domain_trust[domain] = DEFAULT_TRUST
            contradicted.domain_trust[domain] = max(0.0, contradicted.domain_trust[domain] - delta)

            # Record
            now = time.time()
            contradicted.total_contradictions += 1
            contradicted.total_interactions += 1
            contradicter.total_interactions += 1

            contradicted.add_event(TrustEvent(
                timestamp=now,
                event_type="contradiction",
                other_agent=contradicting_agent,
                domain=domain,
                magnitude=delta,
            ))

            # Update trust graph
            if contradicting_agent not in self._trust_graph:
                self._trust_graph[contradicting_agent] = {}
            current = self._trust_graph[contradicting_agent].get(contradicted_agent, 0.5)
            self._trust_graph[contradicting_agent][contradicted_agent] = max(0.0, current - 0.1)

            return contradicter.global_trust, contradicted.global_trust

    def get_trust(self, agent_id: str, domain: Optional[str] = None) -> float:
        """Get an agent's current trust score."""
        with self._lock:
            profile = self._profiles.get(agent_id)
            if not profile:
                return DEFAULT_TRUST
            if domain:
                return profile.trust_for_domain(domain)
            return profile.global_trust

    def get_trust_ranking(self, domain: Optional[str] = None) -> List[Tuple[str, float]]:
        """Get all agents ranked by trust, optionally for a specific domain."""
        with self._lock:
            rankings = []
            for agent_id, profile in self._profiles.items():
                trust = profile.trust_for_domain(domain) if domain else profile.global_trust
                rankings.append((agent_id, trust))
        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def compute_pagerank_trust(self, damping: float = 0.85, iterations: int = 0) -> Dict[str, float]:
        """Compute PageRank-style trust scores with early convergence.

        Trust from a trusted agent is worth more than trust from
        an untrusted one. This captures the recursive nature of trust.
        Uses epsilon convergence check to exit early when scores stabilise.
        """
        max_iter = iterations or self._pagerank_max_iterations
        eps = self._pagerank_epsilon

        with self._lock:
            agents = list(self._profiles.keys())
            n = len(agents)
            if n == 0:
                return {}

            agent_idx = {a: i for i, a in enumerate(agents)}

            # Build adjacency matrix from trust graph
            matrix = np.zeros((n, n))
            for from_agent, targets in self._trust_graph.items():
                if from_agent not in agent_idx:
                    continue
                i = agent_idx[from_agent]
                for to_agent, weight in targets.items():
                    if to_agent not in agent_idx:
                        continue
                    j = agent_idx[to_agent]
                    matrix[i, j] = weight

        # Normalize columns
        col_sums = matrix.sum(axis=0)
        col_sums[col_sums == 0] = 1
        matrix /= col_sums

        # Power iteration with convergence check
        scores = np.ones(n) / n
        for it in range(max_iter):
            new_scores = (1 - damping) / n + damping * matrix.T @ scores
            diff = float(np.linalg.norm(new_scores - scores))
            scores = new_scores
            if diff < eps:
                logger.debug("PageRank converged in %d iterations (diff=%.2e)", it + 1, diff)
                break

        # Normalize to 0-1
        if scores.max() > 0:
            scores = scores / scores.max()

        return {agents[i]: float(scores[i]) for i in range(n)}

    def apply_temporal_decay(self) -> int:
        """Decay all trust scores toward the default. Returns agents affected."""
        now = time.time()
        count = 0
        with self._lock:
            for profile in self._profiles.values():
                age = now - profile.last_active
                decay = math.exp(-age / self._decay_half_life)
                old_trust = profile.global_trust
                profile.global_trust = DEFAULT_TRUST + (old_trust - DEFAULT_TRUST) * decay

                for domain in profile.domain_trust:
                    old_dt = profile.domain_trust[domain]
                    profile.domain_trust[domain] = DEFAULT_TRUST + (old_dt - DEFAULT_TRUST) * decay

                if abs(profile.global_trust - old_trust) > 0.001:
                    count += 1
        return count

    @property
    def agent_count(self) -> int:
        with self._lock:
            return len(self._profiles)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get a summary of all agent trust profiles."""
        with self._lock:
            return {
                agent_id: {
                    "global_trust": round(p.global_trust, 4),
                    "confirmation_rate": round(p.confirmation_rate, 4),
                    "total_entries": p.total_entries,
                    "total_interactions": p.total_interactions,
                    "is_reliable": p.is_reliable,
                    "activity_level": round(p.activity_level, 4),
                }
                for agent_id, p in self._profiles.items()
            }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize trust state for durable persistence."""
        with self._lock:
            return {
                "profiles": {
                    agent_id: {
                        "global_trust": profile.global_trust,
                        "domain_trust": profile.domain_trust,
                        "total_entries": profile.total_entries,
                        "total_confirmations": profile.total_confirmations,
                        "total_contradictions": profile.total_contradictions,
                        "total_interactions": profile.total_interactions,
                        "first_seen": profile.first_seen,
                        "last_active": profile.last_active,
                        "events": [
                            {
                                "timestamp": event.timestamp,
                                "event_type": event.event_type,
                                "other_agent": event.other_agent,
                                "domain": event.domain,
                                "magnitude": event.magnitude,
                            }
                            for event in profile.events
                        ],
                    }
                    for agent_id, profile in self._profiles.items()
                },
                "trust_graph": self._trust_graph,
            }

    def load_dict(self, payload: Optional[Dict[str, Any]]) -> None:
        """Restore trust state from persisted data."""
        with self._lock:
            self._profiles = {}
            self._trust_graph = {}

            if not payload:
                return

            for agent_id, profile_payload in payload.get("profiles", {}).items():
                if not isinstance(profile_payload, dict):
                    continue
                profile = AgentProfile(
                    agent_id=agent_id,
                    global_trust=float(profile_payload.get("global_trust", DEFAULT_TRUST)),
                    domain_trust={
                        domain: float(score)
                        for domain, score in profile_payload.get("domain_trust", {}).items()
                    },
                    total_entries=int(profile_payload.get("total_entries", 0)),
                    total_confirmations=int(profile_payload.get("total_confirmations", 0)),
                    total_contradictions=int(profile_payload.get("total_contradictions", 0)),
                    total_interactions=int(profile_payload.get("total_interactions", 0)),
                    first_seen=float(profile_payload.get("first_seen", time.time())),
                    last_active=float(profile_payload.get("last_active", time.time())),
                )
                profile.events = [
                    TrustEvent(
                        timestamp=float(event.get("timestamp", time.time())),
                        event_type=str(event.get("event_type", "")),
                        other_agent=str(event.get("other_agent", "")),
                        domain=str(event.get("domain", "")),
                        magnitude=float(event.get("magnitude", 0.0)),
                    )
                    for event in profile_payload.get("events", [])
                    if isinstance(event, dict)
                ]
                self._profiles[agent_id] = profile

            self._trust_graph = {
                agent: {other: float(weight) for other, weight in targets.items()}
                for agent, targets in payload.get("trust_graph", {}).items()
                if isinstance(targets, dict)
            }
