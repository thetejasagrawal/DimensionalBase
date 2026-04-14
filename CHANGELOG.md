# Changelog

All notable changes to DimensionalBase will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.5.0] - 2026-04-13

### Added
- Central `DimensionalBaseConfig` dataclass — all tuneable thresholds in one place, passed via `DimensionalBase(config=...)`
- Cross-encoder re-ranking (`rerank=True`) — optional second-pass scoring using `cross-encoder/ms-marco-MiniLM-L-6-v2` for document QA accuracy; lazy-loaded, CPU-only, ~50-100ms for 50 candidates
- Variance-aware scoring reweighting — automatically detects uniform (useless) signals and redistributes weight to discriminative ones; single-document QA shifts to pure similarity, multi-agent keeps balanced weights
- Context window expansion — when a retrieved chunk has sequential neighbours (chunk/41 → chunk/42 → chunk/43), adjacent chunks are pulled in for supporting context
- MMR diversity for multi-agent retrieval — Maximal Marginal Relevance prevents returning redundant entries from the same source; automatically disabled for single-owner (document QA) workloads
- `QueryResult.raw_text` property — clean text output without metadata prefix, maximum signal for LLM context windows
- Circuit breaker for embedding providers — automatic text-only fallback when the embedding API goes down, with recovery probing
- `db demo` CLI command — interactive 30-second terminal experience showing multi-agent coordination, contradiction detection, and budget-aware retrieval
- Web dashboard at `/dashboard/` — dark-themed single-page app with entry browser, real-time event feed (WebSocket), agent trust visualization, and provenance explorer
- `--seed-demo` flag on `db serve` — populates demo data on startup for dashboard exploration
- `/api/v1/events` REST endpoint — event history retrieval for the dashboard
- Rich-formatted CLI output for `status`, `get`, `trust-report`, and `lineage` commands (graceful plain-text fallback)
- Head-to-head LongBench v2 benchmark (`benchmarks/standard/head_to_head.py`) — apples-to-apples comparison of DimensionalBase vs Naive RAG vs Full Context vs Latent Briefing using GPT-4o-mini as the answering LLM
- Conflict detection example (`examples/conflict_detection.py`)
- Test suites for config propagation and circuit breaker (17 new tests, 291 total)

### Changed
- All hardcoded thresholds (contradiction similarity, staleness, trust K-factor, confidence decay, cluster merge, bloom capacity, PageRank iterations, event history max) now configurable via `DimensionalBaseConfig`
- LSH pre-filter now bypassed for small candidate sets (< 50 entries) — fixes missed contradiction detection with real OpenAI embeddings
- Candidate retrieval capped at `MAX_CANDIDATES` (500) — read latency at 10K entries dropped from 561ms to 110ms (5× faster), growth factor improved from 383× to 75× (beats TextPassing's 83×)
- SQL queries in TextChannel now support optional `LIMIT` clause to avoid materialising thousands of unused rows
- Vectorized reference scoring replaces per-entry Python function calls
- Similarity array scatter uses numpy fancy indexing instead of Python loop
- PageRank now uses epsilon convergence check — exits early when scores stabilise
- Event history buffer logs a warning and increments a counter when events are dropped
- `AgentTrustEngine`, `DimensionalSpace`, `EventBus` all accept config values at construction
- Version bumped to 0.5.0

### Fixed
- Contradiction detection with real embeddings — LSH was too aggressive on small knowledge bases, filtering out valid candidates before cosine comparison; now bypassed when < 50 candidates (0/5 → 5/5 on real OpenAI embeddings)
- `setup.py` reduced to a minimal shim — version now sourced exclusively from `pyproject.toml` dynamic config

## [0.4.0] - 2026-04-11

### Added
- Durable SQLite-backed `embeddings` table with semantic index reload on startup
- Optional encrypted value storage via pluggable encryption providers
- Secured REST and WebSocket auth flow using `X-DMB-API-Key` / `api_key`
- Internal synthetic benchmark dependency group (`bench`)
- Hardening tests for semantic durability, ranking, encrypted storage, and secured server flows
- Durable persistence for confidence, trust, and provenance runtime state across restart

### Changed
- `DimensionalBase.get()` now applies Bayesian confidence and trust before ranking and budget packing
- Same-owner updates preserve confidence evidence instead of resetting it
- Domain extraction now uses the second path segment when present
- Status output now reports semantic index health, embedding provider info, vector counts, and encryption status
- Project positioning updated to reflect a hardened alpha OSS core rather than production-ready claims
- Packaged server and CLI bootstrap secure mode by default when an API key is configured, with unauthenticated `/healthz` for readiness checks

### Fixed
- File-backed embedding indexes now survive close/reopen cycles
- TTL cleanup removes semantic state as well as text rows
- Topology stats rebuild correctly after startup, updates, and deletes
- Secured writes no longer allow non-admin owner spoofing
- FastAPI routes no longer bypass the secure wrapper when one is provided
- `compose` responses no longer leak unauthorized nearest-neighbor paths through `materialize()`
- CLI runtime config now correctly instantiates the database instead of returning the Click context object

## [0.3.0] - 2026-04-11

### Added
- Custom exception hierarchy (`DimensionalBaseError` and subclasses)
- Schema migration framework for safe SQLite upgrades
- Task-adaptive scoring (query presence influences scoring weights)
- MCP server for Claude Code / Cursor integration
- LangChain integration (`DimensionalBaseMemory`, `DimensionalBaseTool`)
- CrewAI integration (`DimensionalBaseTool`)
- REST API server (FastAPI + WebSocket)
- CLI tool (`db put`, `db get`, `db status`, etc.)
- Security layer (API key auth, path-based ACL, encryption at rest)
- Standard benchmarks (LongBench v2, HotPotQA)
- Pre-commit hooks (black, ruff, mypy)
- GitHub Actions CI/CD pipeline
- Comprehensive test suite (concurrency, persistence, errors, embeddings)
- README, LICENSE, CONTRIBUTING, CHANGELOG

### Changed
- Version is now single-sourced from `dimensionalbase.__version__`
- `ScoringWeights` now supports adaptive mode (default on)
- TextChannel uses migration framework instead of raw SQL schema

### Fixed
- Version inconsistency across pyproject.toml, __init__.py, and db.py

## [0.2.0] - 2026-04-10

### Added
- Unified VectorStore (single float32 array, pre-normalized, BLAS-fast)
- LSH-accelerated contradiction detection via SemanticFingerprint
- Bloom filter novelty skip for known topics
- Vectorized context scoring (single BLAS matmul)
- LRU query embedding cache
- Incremental reference graph (updated on write, not rebuilt on read)
- Bayesian confidence engine (Beta distributions)
- Agent trust engine (Elo + PageRank)
- Provenance tracking (full DAG with 7 derivation types)
- Semantic compression (delta encoding + deduplication)
- Dimensional algebra (compose, relate, analogy, and 7 more operations)
- Published to PyPI

## [0.1.0] - 2026-03-15

### Added
- Initial implementation
- Four-method API: put(), get(), subscribe(), unsubscribe()
- TextChannel (SQLite) + EmbeddingChannel
- Basic context engine with budget packing
- Event bus with glob pattern matching
