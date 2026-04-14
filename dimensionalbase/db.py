"""
DimensionalBase — the public API.

v0.3: Performance re-architecture.
  - Unified VectorStore (single float32 array, pre-normalized, BLAS-fast)
  - Single prefetch in reasoning (1 query instead of 3)
  - LSH-accelerated contradiction detection
  - Bloom filter novelty skip
  - Vectorized context scoring
  - LRU query embedding cache
  - Incremental reference graph
  - All norms eliminated from hot path (pre-normalized embeddings)

Same surface: put(), get(), subscribe(), unsubscribe(). Deeper engine.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from dimensionalbase.algebra.fingerprint import SemanticFingerprint, BloomFilter
from dimensionalbase.algebra.operations import compose, relate, analogy, subspace_alignment
from dimensionalbase.algebra.space import DimensionalSpace
from dimensionalbase.channels.manager import ChannelManager
from dimensionalbase.config import DimensionalBaseConfig
from dimensionalbase.context.compression import SemanticCompressor
from dimensionalbase.context.engine import ContextEngine
from dimensionalbase.context.reranker import CrossEncoderReranker, Reranker
from dimensionalbase.core.entry import KnowledgeEntry
from dimensionalbase.core.types import (
    ChannelLevel, EntryType, Event, EventType, QueryResult,
    ScoringWeights, Subscription, TTL,
)
from dimensionalbase.embeddings.provider import EmbeddingProvider, NullEmbeddingProvider
from dimensionalbase.events.bus import EventBus
from dimensionalbase.reasoning.active import ActiveReasoning
from dimensionalbase.reasoning.confidence import ConfidenceEngine
from dimensionalbase.reasoning.provenance import ProvenanceTracker
from dimensionalbase.security.encryption import EncryptionProvider
from dimensionalbase.security.validation import (
    validate_confidence,
    validate_metadata,
    validate_owner,
    validate_path,
    validate_value,
)
from dimensionalbase.trust.agent_trust import AgentTrustEngine
from dimensionalbase import __version__ as _version

logger = logging.getLogger("dimensionalbase")

_CONFIDENCE_STATE_KEY = "confidence_engine"
_TRUST_STATE_KEY = "trust_engine"
_PROVENANCE_STATE_KEY = "provenance_tracker"


class DimensionalBase:
    """The protocol and database for AI communication.

    Four methods. Three channels. Under the hood: unified vector store,
    dimensional algebra, Bayesian confidence, provenance DAG, agent trust,
    LSH fingerprinting, bloom filter novelty detection, vectorized scoring.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        embedding_provider: Optional[EmbeddingProvider] = None,
        prefer_embedding: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        encryption_provider: Optional[EncryptionProvider] = None,
        encryption_key: Optional[str] = None,
        encryption_passphrase: Optional[str] = None,
        scoring_weights: Optional[ScoringWeights] = None,
        staleness_threshold: float = 3600.0,
        auto_reasoning: bool = True,
        config: Optional[DimensionalBaseConfig] = None,
        rerank: bool = False,
    ):
        cfg = config or DimensionalBaseConfig()
        self._config = cfg

        # ── Channel layer (creates shared VectorStore) ───────
        self._channels = ChannelManager(
            db_path=db_path,
            embedding_provider=embedding_provider,
            prefer_embedding=prefer_embedding,
            openai_api_key=openai_api_key,
            encryption_provider=encryption_provider,
            encryption_key=encryption_key,
            encryption_passphrase=encryption_passphrase,
        )

        # ── Event system ─────────────────────────────────────
        self._event_bus = EventBus(max_history=cfg.event_history_max)

        # ── Re-ranker (cross-encoder for retrieval accuracy) ─
        _reranker: Optional[Reranker] = None
        if rerank or cfg.rerank_enabled:
            _reranker = CrossEncoderReranker(model_name=cfg.rerank_model)

        # ── Context engine ───────────────────────────────────
        self._context = ContextEngine(
            channel_manager=self._channels,
            weights=scoring_weights,
            reranker=_reranker,
        )

        # ── Compression ──────────────────────────────────────
        self._compressor = SemanticCompressor()

        # ── Dimensional algebra (uses shared VectorStore) ────
        self._space: Optional[DimensionalSpace] = None
        self._fingerprint: Optional[SemanticFingerprint] = None
        self._novelty_filter: Optional[BloomFilter] = None

        if self._channels.has_embeddings:
            dim = self._channels.embedding_provider.dimension()
            # DimensionalSpace wraps the SAME VectorStore the channel uses
            self._space = DimensionalSpace(
                vector_store=self._channels.vector_store,
                merge_threshold=cfg.cluster_merge_threshold,
            )
            self._fingerprint = SemanticFingerprint(dimension=dim)
            self._novelty_filter = BloomFilter(capacity=cfg.bloom_capacity)

        # ── Active reasoning (with LSH fingerprint ref) ──────
        self._reasoning: Optional[ActiveReasoning] = None
        self._auto_reasoning = auto_reasoning
        if auto_reasoning:
            self._reasoning = ActiveReasoning(
                channel_manager=self._channels,
                event_bus=self._event_bus,
                staleness_threshold=cfg.staleness_seconds if staleness_threshold == 3600.0 else staleness_threshold,
                contradiction_threshold=cfg.contradiction_threshold,
                summary_threshold=cfg.summary_threshold,
                fingerprint=self._fingerprint,
            )

        # ── Bayesian confidence ──────────────────────────────
        self._confidence = ConfidenceEngine(
            temporal_decay_half_life=cfg.confidence_decay_half_life,
            propagation_depth=cfg.propagation_depth,
            confirmation_weight=cfg.confirmation_weight,
            contradiction_weight=cfg.contradiction_weight,
        )

        # ── Provenance tracking ──────────────────────────────
        self._provenance = ProvenanceTracker()

        # ── Agent trust model ────────────────────────────────
        self._trust = AgentTrustEngine(
            default_trust=cfg.default_trust,
            k_factor=cfg.trust_k_factor,
            decay_half_life=cfg.trust_decay_half_life,
            min_interactions=cfg.min_interactions_reliable,
            pagerank_max_iterations=cfg.pagerank_max_iterations,
            pagerank_epsilon=cfg.pagerank_epsilon,
        )

        # ── Counters ─────────────────────────────────────────
        self._created_at = time.time()
        self._total_puts = 0
        self._total_gets = 0

        self._rehydrate_runtime_state()

        logger.info(
            f"DimensionalBase v{_version} | channel={self._channels.best_channel_level.name} | "
            f"embeddings={'yes' if self._channels.has_embeddings else 'no'} | "
            f"reasoning={'on' if auto_reasoning else 'off'} | "
            f"algebra={'on' if self._space else 'off'}"
        )

    # ==================================================================
    # PUBLIC API — 4 methods
    # ==================================================================

    def put(
        self,
        path: str,
        value: str,
        owner: str,
        type: str = "fact",
        confidence: float = 1.0,
        refs: Optional[List[str]] = None,
        ttl: str = "session",
        metadata: Optional[Dict[str, str]] = None,
        derived_from: Optional[List[str]] = None,
    ) -> KnowledgeEntry:
        """Write knowledge to the shared dimensional space."""
        self._total_puts += 1
        validate_path(path)
        validate_value(value)
        validate_owner(owner)
        validate_confidence(confidence)
        validate_metadata(metadata)
        refs = refs or []

        entry = KnowledgeEntry(
            path=path, value=value, owner=owner,
            type=EntryType(type), confidence=confidence,
            refs=refs, ttl=TTL(ttl), metadata=metadata or {},
        )

        # ── Check for update ─────────────────────────────────
        existing = self._channels.retrieve(path)
        is_update = existing is not None
        if is_update:
            entry.id = existing.id
            entry.created_at = existing.created_at
            entry.version = existing.version + 1

        # ── Store (embedding generated + pre-normalized in channel) ──
        channel_used = self._channels.store(entry)

        # ── Novelty via bloom filter fast-path ───────────────
        novelty_metrics = {}
        if self._space and entry.has_embedding:
            skip_full = False
            if self._novelty_filter and self._fingerprint:
                fp = self._fingerprint.hash(entry.embedding)
                if self._novelty_filter.might_contain(fp):
                    # Known topic — skip O(n) novelty scan
                    novelty_metrics = {"novelty": 0.1, "info_gain": 0.05}
                    self._space.add_fast(path, entry.embedding)
                    skip_full = True
                self._fingerprint.index(path, entry.embedding)
                self._novelty_filter.add(fp)

            if not skip_full:
                novelty_metrics = self._space.add(path, entry.embedding)

            if novelty_metrics:
                entry.metadata["_novelty"] = str(round(novelty_metrics.get("novelty", 0), 4))

        # ── Incremental reference graph update ───────────────
        self._context.update_refs(entry)

        # ── Post-write: provenance + confidence + trust ──────
        self._post_write_updates(entry, existing, is_update)

        # ── Fire CHANGE event ────────────────────────────────
        self._event_bus.emit(Event(
            type=EventType.CHANGE, path=path,
            data={"owner": owner, "type": type, "version": entry.version,
                  "channel": channel_used.name, "is_update": is_update},
            source_owner=owner, timestamp=time.time(),
        ))

        # ── Active reasoning (single prefetch inside) ────────
        if self._reasoning:
            reasoning_events = self._reasoning.on_write(entry)
            for evt in reasoning_events:
                if evt.type == EventType.CONFLICT:
                    self._handle_conflict_event(evt)

        self._persist_runtime_state()

        return entry

    def get(
        self,
        scope: str = "**",
        budget: int = 2000,
        query: Optional[str] = None,
        owner: Optional[str] = None,
        type: Optional[str] = None,
        reader: Optional[str] = None,
        diversity: float = 0.0,
    ) -> QueryResult:
        """Read relevant knowledge within a token budget."""
        self._total_gets += 1

        result = self._context.query(
            scope=scope, budget=budget, query=query,
            owner=owner, entry_type=type,
            confidence_signal_resolver=self._confidence_signal,
        )

        # Compression (delta encoding for known readers)
        if reader and result.entries:
            compressed = self._compressor.compress(
                entries=result.entries, reader_agent=reader, budget=budget,
            )
            result = QueryResult(
                entries=compressed.entries, total_matched=result.total_matched,
                tokens_used=sum(e.token_estimate for e in compressed.entries),
                budget_remaining=budget - sum(e.token_estimate for e in compressed.entries),
                channel_used=result.channel_used,
            )

        return result

    def subscribe(self, pattern: str, subscriber: str,
                  callback: Callable[[Event], None]) -> Subscription:
        return self._event_bus.subscribe(pattern, subscriber, callback)

    def unsubscribe(self, subscription: Subscription) -> bool:
        return self._event_bus.unsubscribe(subscription)

    # ==================================================================
    # DIMENSIONAL ALGEBRA
    # ==================================================================

    def encode(self, text: str) -> Optional[np.ndarray]:
        if not self._channels.has_embeddings:
            return None
        return self._channels.embedding_provider.embed(text)

    def relate(self, path_a: str, path_b: str) -> Optional[Dict[str, float]]:
        if not self._space:
            return None
        a, b = self._space.get(path_a), self._space.get(path_b)
        if a is None or b is None:
            return None
        return relate(a, b)

    def compose(self, paths: List[str], mode: str = "attentive") -> Optional[np.ndarray]:
        if not self._space:
            return None
        vectors = [self._space.get(p) for p in paths]
        vectors = [v for v in vectors if v is not None]
        if len(vectors) < 2:
            return vectors[0] if vectors else None
        confidences = [self._confidence.get_confidence(p) for p in paths]
        return compose(vectors, weights=confidences, mode=mode)

    def materialize(self, vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        if not self._space:
            return []
        return self._space.search(vector, k=k)

    # ==================================================================
    # CONVENIENCE METHODS
    # ==================================================================

    def delete(self, path: str) -> bool:
        removed = self._remove_paths([path], emit_delete=True)
        return removed > 0

    def exists(self, path: str) -> bool:
        return self._channels.retrieve(path) is not None

    def retrieve(self, path: str) -> Optional[KnowledgeEntry]:
        return self._channels.retrieve(path)

    def clear_turn(self) -> int:
        return self._clear_by_ttl(TTL.TURN)

    def clear_session(self) -> int:
        return self._clear_by_ttl(TTL.TURN) + self._clear_by_ttl(TTL.SESSION)

    # ==================================================================
    # INTROSPECTION
    # ==================================================================

    @property
    def channel(self) -> ChannelLevel:
        return self._channels.best_channel_level

    @property
    def has_embeddings(self) -> bool:
        return self._channels.has_embeddings

    @property
    def entry_count(self) -> int:
        return self._channels.count()

    @property
    def capabilities(self):
        return self._channels.capabilities()

    @property
    def events(self) -> EventBus:
        return self._event_bus

    @property
    def trust(self) -> AgentTrustEngine:
        return self._trust

    @property
    def provenance(self) -> ProvenanceTracker:
        return self._provenance

    @property
    def confidence(self) -> ConfidenceEngine:
        return self._confidence

    @property
    def space(self) -> Optional[DimensionalSpace]:
        return self._space

    def status(self) -> Dict[str, Any]:
        caps = self._channels.capabilities()
        vs = self._channels.vector_store
        s: Dict[str, Any] = {
            "entries": self._channels.count(),
            "channel": self._channels.best_channel_level.name,
            "embeddings": self._channels.has_embeddings,
            "embedding_provider": (
                self._channels.embedding_provider.name
                if self._channels.embedding_provider is not None else "none"
            ),
            "embedding_dimension": (
                self._channels.embedding_provider.dimension()
                if self._channels.embedding_provider is not None else 0
            ),
            "vector_entries": vs.count if vs else 0,
            "semantic_index_ready": self._channels.semantic_index_ready,
            "reindexed_on_startup": self._channels.reindexed_on_startup,
            "encryption_enabled": self._channels.encryption_enabled,
            "channels": {c.level.name: {"available": c.available, "description": c.description} for c in caps},
            "subscriptions": self._event_bus.subscription_count,
            "reasoning": self._auto_reasoning,
            "uptime_seconds": round(time.time() - self._created_at, 2),
            "total_puts": self._total_puts,
            "total_gets": self._total_gets,
            "agents": self._trust.summary(),
            "provenance_nodes": self._provenance.node_count,
        }
        if self._space:
            m = self._space.metrics()
            s["space"] = {
                "total_points": m.total_points,
                "intrinsic_dimension": round(m.intrinsic_dimension_estimate, 1),
                "mean_pairwise_similarity": round(m.mean_pairwise_similarity, 4),
                "coverage": round(m.coverage, 4),
                "cluster_count": m.cluster_count,
            }
        if vs:
            s["vector_store_memory_mb"] = round(vs.memory_bytes / (1024 * 1024), 2)
        if self._fingerprint:
            s["fingerprint_index_size"] = self._fingerprint.indexed_count
        return s

    def agent_trust_report(self) -> Dict[str, Any]:
        report = self._trust.summary()
        pagerank = self._trust.compute_pagerank_trust()
        for aid in report:
            report[aid]["pagerank_trust"] = round(pagerank.get(aid, 0), 4)
        return report

    def lineage(self, path: str) -> List:
        return self._provenance.get_lineage(path)

    def knowledge_topology(self) -> Dict[str, Any]:
        if not self._space:
            return {"available": False}
        m = self._space.metrics()
        clusters = self._space.detect_clusters()
        voids = self._space.find_voids()
        return {
            "available": True, "total_points": m.total_points,
            "intrinsic_dimension": round(m.intrinsic_dimension_estimate, 1),
            "coverage": round(m.coverage, 4),
            "clusters": [{"count": c.count, "radius": round(c.radius, 4), "paths": c.paths[:5]} for c in clusters],
            "void_count": len(voids),
        }

    def close(self) -> None:
        self._persist_runtime_state()
        self._channels.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return (f"DimensionalBase(entries={self.entry_count}, "
                f"channel={self.channel.name}, embeddings={self.has_embeddings}, "
                f"agents={self._trust.agent_count})")

    # ==================================================================
    # INTERNAL
    # ==================================================================

    def _post_write_updates(self, entry: KnowledgeEntry,
                            existing: Optional[KnowledgeEntry], is_update: bool):
        """Batched provenance + confidence + trust updates."""
        value_hash = hashlib.md5(entry.value.encode()).hexdigest()[:16]

        # Provenance
        if is_update:
            self._provenance.record_update(
                path=entry.path, owner=entry.owner,
                value_hash=value_hash, version=entry.version,
            )
        else:
            self._provenance.record_creation(
                path=entry.path, owner=entry.owner,
                value_hash=value_hash, derived_from=entry.refs,
            )

        # Confidence
        if is_update and existing and existing.owner != entry.owner:
            self._handle_cross_agent_write(entry, existing)
        elif is_update:
            self._confidence.refresh(entry.path, entry.confidence, entry.owner)
        else:
            self._confidence.register(entry.path, entry.confidence, entry.owner)

        # Trust
        self._trust.record_entry(entry.owner)

    def _handle_cross_agent_write(self, new_entry: KnowledgeEntry,
                                   existing: KnowledgeEntry):
        domain = self._get_domain(new_entry.path)
        if self._channels.has_embeddings and new_entry.has_embedding and existing.has_embedding:
            # Pre-normalized: dot = cosine
            sim = float(np.dot(new_entry.embedding, existing.embedding))
            is_confirmation = sim > 0.8
        else:
            is_confirmation = new_entry.value.strip().lower() == existing.value.strip().lower()

        if is_confirmation:
            self._confidence.confirm(existing.path, new_entry.owner)
            self._trust.record_confirmation(new_entry.owner, existing.owner, domain)
            self._provenance.record_confirmation(existing.path, new_entry.owner)
        else:
            self._confidence.contradict(existing.path, new_entry.owner)
            self._trust.record_contradiction(
                new_entry.owner, existing.owner, domain,
                confidence_of_contradicting=new_entry.confidence,
                confidence_of_contradicted=existing.confidence,
            )
            vh = hashlib.md5(new_entry.value.encode()).hexdigest()[:16]
            self._provenance.record_contradiction(existing.path, new_entry.owner, vh)

    def _handle_conflict_event(self, event: Event):
        data = event.data
        new_owner = data.get("new_entry_owner", "")
        existing_owner = data.get("existing_entry_owner", "")
        if new_owner and existing_owner:
            self._trust.record_contradiction(
                new_owner, existing_owner, self._get_domain(event.path),
                confidence_of_contradicting=data.get("new_confidence", 0.5),
                confidence_of_contradicted=data.get("existing_confidence", 0.5),
            )

    @staticmethod
    def _get_domain(path: str) -> str:
        parts = path.split("/")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
        return parts[0] if parts else ""

    def _confidence_signal(self, entry: KnowledgeEntry) -> float:
        bayesian_conf = self._confidence.get_confidence(entry.path)
        domain_trust = self._trust.get_trust(entry.owner, domain=self._get_domain(entry.path))
        return bayesian_conf * (0.6 + 0.4 * domain_trust)

    def _clear_by_ttl(self, ttl: TTL) -> int:
        paths = self._channels.text_channel.delete_paths_by_ttl(ttl)
        if not paths:
            return 0
        self._remove_runtime_state(paths)
        return len(paths)

    def _remove_paths(self, paths: List[str], emit_delete: bool = False) -> int:
        removed_paths: List[str] = []
        for path in paths:
            deleted = self._channels.text_channel.delete(path)
            if deleted:
                removed_paths.append(path)

        if not removed_paths:
            return 0

        self._remove_runtime_state(removed_paths)

        if emit_delete:
            now = time.time()
            for path in removed_paths:
                self._event_bus.emit(Event(type=EventType.DELETE, path=path, timestamp=now))

        return len(removed_paths)

    def _remove_runtime_state(self, paths: List[str]) -> None:
        for path in paths:
            self._confidence.remove(path)
            self._context.remove_path(path)
            if self._channels.vector_store is not None:
                self._channels.vector_store.remove(path)
            if self._fingerprint:
                self._fingerprint.remove(path)
        if self._space:
            self._space.refresh_from_store()
        self._persist_runtime_state()

    def _rehydrate_runtime_state(self) -> None:
        """Restore persisted runtime state and rebuild derived indexes."""
        for entry in self._channels.all_entries():
            self._context.update_refs(entry)

        text_channel = self._channels.text_channel
        self._confidence.load_dict(text_channel.load_system_state(_CONFIDENCE_STATE_KEY))
        self._trust.load_dict(text_channel.load_system_state(_TRUST_STATE_KEY))
        self._provenance.load_dict(text_channel.load_system_state(_PROVENANCE_STATE_KEY))

    def _persist_runtime_state(self) -> None:
        """Persist runtime state that must survive process restarts."""
        text_channel = self._channels.text_channel
        text_channel.save_system_state(_CONFIDENCE_STATE_KEY, self._confidence.to_dict())
        text_channel.save_system_state(_TRUST_STATE_KEY, self._trust.to_dict())
        text_channel.save_system_state(_PROVENANCE_STATE_KEY, self._provenance.to_dict())

    @staticmethod
    def tool_definitions() -> List[Dict[str, Any]]:
        return [
            {"name": "db_put", "description": "Write knowledge to the shared store.",
             "parameters": {"type": "object", "properties": {
                 "path": {"type": "string"}, "value": {"type": "string"},
                 "type": {"type": "string", "enum": ["fact", "decision", "plan", "observation"], "default": "fact"},
                 "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 1.0},
                 "refs": {"type": "array", "items": {"type": "string"}},
             }, "required": ["path", "value"]}},
            {"name": "db_get", "description": "Read relevant knowledge within a token budget.",
             "parameters": {"type": "object", "properties": {
                 "scope": {"type": "string", "default": "**"},
                 "budget": {"type": "integer", "default": 500},
                 "query": {"type": "string"},
             }}},
            {"name": "db_subscribe", "description": "Watch for changes matching a pattern.",
             "parameters": {"type": "object", "properties": {
                 "pattern": {"type": "string"},
             }, "required": ["pattern"]}},
        ]
