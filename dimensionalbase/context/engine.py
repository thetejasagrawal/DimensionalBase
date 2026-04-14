"""
ContextEngine — budget-aware knowledge retrieval.

v0.3 optimizations:
  - LRU embedding cache: same query = one embedding call, not N
  - Vectorized batch scoring: single BLAS matmul, not per-entry loop
  - heapq.nlargest: O(n log k) instead of O(n log n) full sort
  - Incremental reference graph: updated on write, not rebuilt on read
  - Pre-normalized embeddings: dot product = cosine, no norms
"""

from __future__ import annotations

import hashlib
import heapq
import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from dimensionalbase.channels.manager import ChannelManager
from dimensionalbase.context.reranker import Reranker
from dimensionalbase.core.entry import KnowledgeEntry
from dimensionalbase.core.types import ChannelLevel, QueryResult, ScoringWeights

logger = logging.getLogger("dimensionalbase.context")

DEFAULT_BUDGET = 2000
MAX_CANDIDATES = 500


class ContextEngine:
    """Budget-aware context retrieval with vectorized scoring."""

    def __init__(self, channel_manager: ChannelManager, weights: Optional[ScoringWeights] = None,
                 reranker: Optional[Reranker] = None):
        self._channels = channel_manager
        self._weights = weights or ScoringWeights()
        self._reranker = reranker

        # LRU embedding cache: hash(query) -> (embedding, timestamp)
        self._emb_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._cache_ttl = 60.0
        self._cache_max = 256

        # Incremental reference graph (updated on writes, not rebuilt on reads)
        self._ref_counts: Dict[str, int] = {}
        self._entry_refs: Dict[str, List[str]] = {}

    # ── Called from db.put() to keep ref graph current ────────

    def update_refs(self, entry: KnowledgeEntry) -> None:
        """Incrementally update the reference graph on each write."""
        previous_refs = self._entry_refs.get(entry.path, [])
        for ref in previous_refs:
            current = self._ref_counts.get(ref, 0)
            if current <= 1:
                self._ref_counts.pop(ref, None)
            else:
                self._ref_counts[ref] = current - 1

        for ref in entry.refs:
            self._ref_counts[ref] = self._ref_counts.get(ref, 0) + 1

        self._entry_refs[entry.path] = list(entry.refs)
        self._ref_counts.setdefault(entry.path, 0)

    def remove_path(self, path: str) -> None:
        """Remove reference bookkeeping for a deleted path."""
        refs = self._entry_refs.pop(path, [])
        for ref in refs:
            current = self._ref_counts.get(ref, 0)
            if current <= 1:
                self._ref_counts.pop(ref, None)
            else:
                self._ref_counts[ref] = current - 1
        self._ref_counts.pop(path, None)

    # ── Adaptive Weight Logic ────────────────────────────────

    def _adapt_weights(self, has_query: bool) -> ScoringWeights:
        """Adapt scoring weights based on query presence.

        Inspired by Ramp Labs' Latent Briefing: the reader's task should
        influence what information is considered relevant.
        """
        w = self._weights
        if not w.adaptive:
            return w
        if has_query:
            # Query provided — boost similarity (most relevant to the task)
            return ScoringWeights(
                recency=max(0.0, w.recency - 0.10),
                confidence=w.confidence,
                similarity=min(1.0, w.similarity + 0.10),
                reference_distance=w.reference_distance,
                adaptive=w.adaptive,
            )
        else:
            # No query — boost recency (freshest information)
            return ScoringWeights(
                recency=min(1.0, w.recency + 0.15),
                confidence=w.confidence,
                similarity=max(0.0, w.similarity - 0.15),
                reference_distance=w.reference_distance,
                adaptive=w.adaptive,
            )

    @staticmethod
    def _variance_reweight(
        signals: Dict[str, np.ndarray],
        base_weights: Dict[str, float],
        min_variance: float = 0.001,
    ) -> Dict[str, float]:
        """Redistribute weight from uniform (useless) signals to discriminative ones.

        When a scoring signal has near-zero variance (e.g. all entries have the
        same confidence, same recency, same ref count), that signal cannot
        distinguish good entries from bad ones — its weight is wasted.  This
        method detects uniform signals and shifts their weight proportionally
        to signals that *do* vary.

        This naturally handles:
          - Single-document QA: recency/confidence/refs uniform → similarity dominates
          - Multi-agent coordination: all signals vary → original weights preserved
        """
        variances = {name: float(np.var(arr)) for name, arr in signals.items()}

        # Identify which signals are discriminative (have meaningful spread)
        discriminative = {
            name: var for name, var in variances.items() if var > min_variance
        }

        if not discriminative:
            # Nothing varies at all — keep base weights as-is
            return dict(base_weights)

        # Collect weight from non-discriminative (uniform) signals
        dead_weight = sum(
            base_weights[name] for name in base_weights
            if name not in discriminative
        )

        # Redistribute proportionally to discriminative signals
        disc_total = sum(base_weights[name] for name in discriminative)
        new_weights = {}
        for name in base_weights:
            if name in discriminative and disc_total > 0:
                share = base_weights[name] / disc_total
                new_weights[name] = base_weights[name] + dead_weight * share
            else:
                new_weights[name] = 0.0

        return new_weights

    # ── Core Query ────────────────────────────────────────────

    def query(
        self,
        scope: str,
        budget: int = DEFAULT_BUDGET,
        query: Optional[str] = None,
        owner: Optional[str] = None,
        entry_type: Optional[str] = None,
        reader: Optional[str] = None,
        confidence_signal_resolver: Optional[Callable[[KnowledgeEntry], float]] = None,
    ) -> QueryResult:
        # 0. Adapt weights based on query presence
        self._active_weights = self._adapt_weights(has_query=query is not None)

        # 1. Collect candidates
        candidates = self._channels.query_by_path(scope)

        if owner:
            candidates = [e for e in candidates if e.owner == owner]
        if entry_type:
            candidates = [e for e in candidates if e.type.value == entry_type]

        # 1b. Cap candidates to avoid O(n) scoring on huge knowledge bases.
        # Candidates are already sorted by updated_at DESC from SQL, so
        # slicing keeps the most recent entries.
        if len(candidates) > MAX_CANDIDATES:
            candidates = candidates[:MAX_CANDIDATES]

        if not candidates:
            return QueryResult(
                entries=[], total_matched=0, tokens_used=0,
                budget_remaining=budget, channel_used=self._channels.best_channel_level,
            )

        # 2. Get query embedding (cached)
        query_embedding = self._get_query_embedding(query) if query else None

        # 3. Vectorized scoring
        scored = self._score_candidates_vectorized(
            candidates,
            query_embedding,
            confidence_signal_resolver=confidence_signal_resolver,
        )

        # 4. Top-k via heapq (O(n log k) not O(n log n))
        avg_tokens = max(1, sum(e.token_estimate for e in candidates) // len(candidates))
        estimated_k = min(len(scored), max(10, (budget // avg_tokens) * 2))

        # MMR diversity: only useful when entries come from diverse sources
        # (multi-agent). For single-source data (document QA), adjacent chunks
        # provide essential supporting context — diversity hurts.
        owners = set(e.owner for e, _ in scored)
        use_mmr = query_embedding is not None and len(scored) > estimated_k and len(owners) > 1
        if use_mmr:
            top_k = self._mmr_select(scored, query_embedding, estimated_k, lam=0.7)
        else:
            top_k = heapq.nlargest(estimated_k, scored, key=lambda x: x[1])

        # 4b. Cross-encoder re-ranking (if enabled)
        # Feed ALL candidates (up to MAX_CANDIDATES) to the re-ranker.
        # The cross-encoder handles 500 entries in ~300ms — fast enough.
        if self._reranker is not None and query:
            top_k = self._reranker.rerank(query, scored, estimated_k)

        # 4c. Context window expansion: if a retrieved entry has
        # sequential neighbours (chunk/41 → chunk/42 → chunk/43),
        # pull them in for supporting context. This matters for
        # document QA where answers span chunk boundaries.
        packed_entries = [entry for entry, _ in top_k]
        if query and len(owners) <= 1:
            packed_entries = self._expand_neighbours(packed_entries, candidates)

        # 5. Pack within budget
        packed, tokens_used = self._pack_within_budget(packed_entries, budget)

        return QueryResult(
            entries=packed, total_matched=len(candidates),
            tokens_used=tokens_used, budget_remaining=budget - tokens_used,
            channel_used=self._channels.best_channel_level,
        )

    # ── Vectorized Scoring ────────────────────────────────────

    def _score_candidates_vectorized(
        self,
        candidates: List[KnowledgeEntry],
        query_embedding: Optional[np.ndarray] = None,
        confidence_signal_resolver: Optional[Callable[[KnowledgeEntry], float]] = None,
    ) -> List[Tuple[KnowledgeEntry, float]]:
        """Score ALL candidates in batch numpy operations."""
        n = len(candidates)
        now = time.time()
        w = getattr(self, "_active_weights", self._weights)

        # Recency (vectorized)
        ages = np.array([max(0.0, now - e.updated_at) for e in candidates], dtype=np.float32)
        recency = np.power(2.0, -ages / 3600.0)

        # Confidence (vectorized)
        if confidence_signal_resolver is None:
            confidence = np.array([e.confidence for e in candidates], dtype=np.float32)
        else:
            confidence = np.array(
                [max(0.0, min(1.0, confidence_signal_resolver(e))) for e in candidates],
                dtype=np.float32,
            )

        # Similarity (vectorized BLAS)
        similarity = np.full(n, 0.5, dtype=np.float32)
        if query_embedding is not None:
            emb_indices = [i for i, e in enumerate(candidates) if e.has_embedding]
            if emb_indices:
                idx_arr = np.array(emb_indices)
                emb_matrix = np.stack([candidates[i].embedding for i in emb_indices])
                # Pre-normalized: dot product = cosine similarity
                sims = emb_matrix @ query_embedding.astype(np.float32)
                similarity[idx_arr] = np.maximum(0.0, sims)

        # Reference distance (vectorized via batch dict lookup)
        ref_raw = np.array(
            [self._ref_counts.get(e.path, 0) for e in candidates],
            dtype=np.float32,
        )
        ref_scores = np.where(ref_raw > 0, np.minimum(1.0, 0.3 + 0.3 * np.log1p(ref_raw)), 0.1)

        # Variance-aware reweighting: shift weight from uniform signals
        # to discriminative ones (e.g. when all entries have same owner/time,
        # similarity should dominate)
        if w.adaptive:
            signals = {
                "recency": recency,
                "confidence": confidence,
                "similarity": similarity,
                "reference_distance": ref_scores,
            }
            base = {
                "recency": w.recency,
                "confidence": w.confidence,
                "similarity": w.similarity,
                "reference_distance": w.reference_distance,
            }
            aw = self._variance_reweight(signals, base)
            total = (aw["recency"] * recency + aw["confidence"] * confidence
                     + aw["similarity"] * similarity + aw["reference_distance"] * ref_scores)
        else:
            total = (w.recency * recency + w.confidence * confidence
                     + w.similarity * similarity + w.reference_distance * ref_scores)

        # Only write scores on top candidates (defer the rest)
        top_indices = np.argpartition(total, -min(n, MAX_CANDIDATES))[-min(n, MAX_CANDIDATES):]
        for i in top_indices:
            candidates[i]._raw_score = float(total[i])
            candidates[i]._score = float(total[i])

        return list(zip(candidates, total.tolist()))

    # ── Context Window Expansion ───────────────────────────────

    @staticmethod
    def _expand_neighbours(
        selected: List[KnowledgeEntry],
        all_candidates: List[KnowledgeEntry],
    ) -> List[KnowledgeEntry]:
        """Pull adjacent chunks to give the LLM more supporting context.

        If chunk/42 is selected, also include chunk/41 and chunk/43
        (if they exist in the candidate pool but weren't selected).
        Entries are re-ordered by path so the context reads naturally.
        """
        import re
        candidate_map = {e.path: e for e in all_candidates}
        selected_paths = {e.path for e in selected}
        expanded_paths = set(selected_paths)

        for entry in selected:
            # Try to extract a numeric suffix: "doc/chunk/42" → ("doc/chunk/", 42)
            match = re.match(r"^(.*/)(\d+)$", entry.path)
            if not match:
                continue
            prefix, idx = match.group(1), int(match.group(2))
            for neighbour_idx in [idx - 1, idx + 1]:
                neighbour_path = f"{prefix}{neighbour_idx}"
                if neighbour_path not in expanded_paths and neighbour_path in candidate_map:
                    expanded_paths.add(neighbour_path)

        # Build the expanded list, sorted by path for natural reading order
        result = [candidate_map[p] for p in expanded_paths if p in candidate_map]
        result.sort(key=lambda e: e.path)
        return result

    # ── MMR Diversity ─────────────────────────────────────────

    @staticmethod
    def _mmr_select(
        scored: List[Tuple[KnowledgeEntry, float]],
        query_emb: np.ndarray,
        k: int,
        lam: float = 0.7,
    ) -> List[Tuple[KnowledgeEntry, float]]:
        """Maximal Marginal Relevance: balance relevance with diversity.

        lam=1.0 → pure relevance (same as top-k).
        lam=0.0 → pure diversity (most different from each other).
        lam=0.7 → good balance for document retrieval.
        """
        # Filter to entries with embeddings
        with_emb = [(e, s) for e, s in scored if e.has_embedding]
        without_emb = [(e, s) for e, s in scored if not e.has_embedding]

        if not with_emb:
            return heapq.nlargest(k, scored, key=lambda x: x[1])

        selected: List[Tuple[KnowledgeEntry, float]] = []
        remaining = list(with_emb)

        for _ in range(min(k, len(with_emb))):
            best_score = -1.0
            best_idx = 0

            for i, (entry, rel_score) in enumerate(remaining):
                # Relevance term: original score
                relevance = rel_score

                # Diversity term: max similarity to already-selected entries
                if selected:
                    max_sim = max(
                        float(np.dot(entry.embedding, sel_e.embedding))
                        for sel_e, _ in selected
                        if sel_e.has_embedding
                    )
                else:
                    max_sim = 0.0

                mmr = lam * relevance - (1 - lam) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        # Add non-embedding entries at the end if space remains
        if len(selected) < k and without_emb:
            selected.extend(without_emb[:k - len(selected)])

        return selected

    # ── Embedding Cache ───────────────────────────────────────

    def _get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Cached query embedding. Same query in 60s = one API call, not N."""
        if not self._channels.has_embeddings:
            return None

        cache_key = hashlib.md5(query.encode()).hexdigest()
        now = time.time()

        if cache_key in self._emb_cache:
            emb, ts = self._emb_cache[cache_key]
            if now - ts < self._cache_ttl:
                return emb

        try:
            raw = self._channels.embedding_provider.embed(query)
            # Normalize for consistency with VectorStore
            emb = np.asarray(raw, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 1e-12:
                emb = emb / norm

            # Evict oldest if cache full
            if len(self._emb_cache) >= self._cache_max:
                oldest = min(self._emb_cache, key=lambda k: self._emb_cache[k][1])
                del self._emb_cache[oldest]

            self._emb_cache[cache_key] = (emb, now)
            return emb
        except Exception:
            return None

    # ── Reference Scoring ─────────────────────────────────────

    def _reference_score(self, path: str) -> float:
        """Score based on reference connectivity. Uses incremental graph."""
        count = self._ref_counts.get(path, 0)
        if count == 0:
            return 0.1
        return min(1.0, 0.3 + 0.3 * np.log1p(count))

    # ── Budget Packing ────────────────────────────────────────

    @staticmethod
    def _pack_within_budget(
        entries: List[KnowledgeEntry], budget: int,
    ) -> Tuple[List[KnowledgeEntry], int]:
        """Hierarchical fallback: full → compact → path-only."""
        packed = []
        tokens_used = 0

        for entry in entries:
            full = entry.token_estimate
            if tokens_used + full <= budget:
                packed.append(entry)
                tokens_used += full
                continue

            compact = entry.compact_token_estimate
            if tokens_used + compact <= budget:
                packed.append(entry)
                tokens_used += compact
                continue

            path_only = entry.path_only_token_estimate
            if tokens_used + path_only <= budget:
                packed.append(entry)
                tokens_used += path_only
                continue

            break  # Budget exhausted
        return packed, tokens_used
