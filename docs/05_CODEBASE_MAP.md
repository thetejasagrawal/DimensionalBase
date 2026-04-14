# DimensionalBase — Complete Codebase Map

## Root Directory

The root directory of this repository. All paths below are relative to it.

---

## Top-Level Files

| File | Purpose |
|------|---------|
| `README.md` | Quick-start guide, installation, 5-minute intro |
| `DimensionalBase-Final-Vision.md` | 🔑 **Comprehensive design vision** — read this for the full picture |
| `BENCHMARK_REPORT.md` | Internal synthetic token benchmark report (6 models, strong token savings) |
| `AGENT_COMMS_BENCHMARK_REPORT.md` | Internal synthetic communication pattern benchmark |
| `MULTI_AGENT_BENCHMARK_REPORT.md` | Internal synthetic multi-agent environment benchmark |
| `CHANGELOG.md` | Version history: v0.1 → v0.5 with what changed |
| `CONTRIBUTING.md` | Development workflow, code style, PR guidelines |
| `LICENSE` | MIT |
| `pyproject.toml` | Python package config: name=dimensionalbase, dynamic version, extras |
| `setup.py` | Legacy installation script (for pip install -e .) |
| `Dockerfile` | Container image (Python 3.11, installs all extras) |
| `docker-compose.yml` | Orchestration: REST API + MCP server |
| `mcp.json` | MCP server configuration for Claude Code/Cursor |
| `.pre-commit-config.yaml` | Hooks: black (100 char), ruff (linting), mypy (types) |
| `.gitignore` | Standard Python gitignore |

---

## `dimensionalbase/` — Main Source (20 modules)

### `dimensionalbase/__init__.py`
Public exports. What gets imported when you do `from dimensionalbase import ...`:
- `DimensionalBase` (main class)
- `KnowledgeEntry`, `QueryResult`, `Event`, `Subscription`
- `EntryType`, `TTL`, `ChannelLevel`, `EventType`
- `ScoringWeights`
- `__version__ = "0.5.0"`

### `dimensionalbase/db.py` (~517 LOC)
**The heart of the system.** Single `DimensionalBase` class that:
- Wires all subsystems together in `__init__`
- Implements `put()`, `get()`, `subscribe()`, `unsubscribe()`
- Delegates to ChannelManager, ContextEngine, ActiveReasoning, etc.
- Exposes algebra methods (`encode`, `relate`, `compose`, `materialize`)
- Exposes introspection methods (`status`, `agent_trust_report`, `lineage`, `knowledge_topology`)

---

### `dimensionalbase/core/` — Type System

| File | Classes/Contents |
|------|-----------------|
| `core/types.py` | `EntryType` enum (fact/decision/plan/observation), `TTL` enum (turn/session/persistent), `ChannelLevel` enum (TEXT/EMBEDDING/TENSOR), `EventType` enum (change/delete/conflict/gap/stale/summary) |
| `core/entry.py` | `KnowledgeEntry` dataclass — the atomic unit of knowledge. Has path, value, owner, type, confidence, refs, embedding, version, ttl, id, created_at, updated_at, metadata |
| `core/matching.py` | Glob pattern matching for paths. `match(pattern, path) -> bool`. Used for scoped `get()` and `subscribe()` pattern matching. |

---

### `dimensionalbase/channels/` — 3-Channel Communication

| File | Class | Purpose |
|------|-------|---------|
| `channels/base.py` | `Channel` (ABC) | Abstract base: `write(entry)`, `query(scope, query, filters)`, `channel_level` |
| `channels/text.py` | `TextChannel` | Channel 1. SQLite backend. Always active. Text-only (no embeddings). Supports glob path queries. |
| `channels/embedding.py` | `EmbeddingChannel` | Channel 2. Vector search. Active when embedding provider available. Delegates vector ops to shared VectorStore. |
| `channels/tensor.py` | `TensorChannel` | Channel 3. Placeholder for Phase 4 (raw KV cache). Currently raises `NotImplementedError`. |
| `channels/manager.py` | `ChannelManager` | Creates shared VectorStore, initializes channels, and routes put/get based on explicitly configured embedding support. |

---

### `dimensionalbase/storage/` — Storage Layer

| File | Class | Purpose |
|------|-------|---------|
| `storage/vectors.py` | `VectorStore` | Unified float32 contiguous array. Pre-normalized. Thread-safe (RLock). `add`, `remove`, `get`, `search`, `all_similarities`. |
| `storage/migrations.py` | `ensure_schema_current()` + migration registry | SQLite schema versioning. Applied automatically on init. Currently on schema version 3. |

**VectorStore internal layout:**
```
_matrix: float32 array, shape (capacity, dimension)
_paths:  list of str, length = current fill level
_path_to_idx: dict str → int

When full: resize to capacity * 2 (amortized O(1))
On delete: swap last element into deleted slot (keeps array contiguous)
```

---

### `dimensionalbase/algebra/` — Dimensional Algebra

| File | Class/Functions | Purpose |
|------|----------------|---------|
| `algebra/operations.py` | `compose()`, `relate()`, `project()`, `interpolate()`, `decompose()`, `centroid()`, `analogy()`, `subspace_alignment()` | 8 algebraic operations on embedding vectors |
| `algebra/space.py` | `DimensionalSpace` | Analytics layer: running stats (Welford), intrinsic dimensionality, cluster detection, coverage, novelty scoring |
| `algebra/fingerprint.py` | `LSHFingerprint`, `BloomFilter` | Fast contradiction pre-filtering (narrows O(N) → O(20)) and novelty skip |

---

### `dimensionalbase/context/` — Context Engine

| File | Class | Purpose |
|------|-------|---------|
| `context/engine.py` | `ContextEngine` | Vectorized 4-signal scoring, LRU embedding cache, budget packing (hierarchical fallback) |
| `context/compression.py` | `SemanticCompressor` | Delta encoding, deduplication for tight budget scenarios |

**Scoring weights (default):**
- recency: 0.30
- confidence: 0.20
- similarity: 0.30
- ref_distance: 0.20

**LRU cache:** Same query string within 60 seconds reuses the cached embedding. Avoids repeated API calls for the same semantic query.

---

### `dimensionalbase/reasoning/` — Active Reasoning

| File | Class | Purpose |
|------|-------|---------|
| `reasoning/active.py` | `ActiveReasoning` | Contradiction detection (similarity threshold 0.75), gap detection (plan refs without observations), staleness detection, auto-summarization |
| `reasoning/confidence.py` | `ConfidenceEngine` | Bayesian Beta distributions for confidence. `register()`, `confirm()`, `contradict()`, `get_confidence()` |
| `reasoning/provenance.py` | `ProvenanceTracker` | DAG of derivation history. 7 derivation types. `record()`, `lineage()`, `full_dag()` |

---

### `dimensionalbase/trust/` — Agent Trust Model

| File | Class | Purpose |
|------|-------|---------|
| `trust/agent_trust.py` | `AgentTrustEngine`, `AgentProfile` | Elo-like trust tracking. `record_write()`, `record_confirmation()`, `record_contradiction()`, `get_profile()`, `report()` |

**Domain extraction:** Path `"task/auth/status"` → domain `"auth"`. Domain trust updated separately from global trust.

---

### `dimensionalbase/events/` — Event Bus

| File | Class | Purpose |
|------|-------|---------|
| `events/bus.py` | `EventBus` | Pub/sub. `emit()`, `subscribe()`, `unsubscribe()`. Ring buffer of last 100 events. Thread-safe. |

---

### `dimensionalbase/embeddings/` — Embedding Providers

| File | Class | Purpose |
|------|-------|---------|
| `embeddings/provider.py` | `EmbeddingProvider` (ABC), `LocalEmbeddingProvider`, `OpenAIEmbeddingProvider`, `NullEmbeddingProvider` | Explicit provider resolution, `encode()`, `encode_batch()`, `dimension` |
| `embeddings/__init__.py` | Exports | `auto_detect_provider(prefer, openai_key)` |

**Provider resolution order:**
1. Explicit custom provider or `prefer_embedding="local"`
2. Explicit `prefer_embedding="openai"` or `openai_api_key`
3. NullEmbeddingProvider (text-only mode, no embeddings)

---

### `dimensionalbase/mcp/` — MCP Server

| File | Contents |
|------|---------|
| `mcp/server.py` | 6 MCP tools (db_put, db_get, db_relate, db_compose, db_status, db_subscribe) + 3 resources (recent entries, recent events, status). Wraps a singleton `DimensionalBase` instance. |
| `mcp/__main__.py` | Entry point: `python -m dimensionalbase.mcp` |

---

### `dimensionalbase/server/` — REST API

| File | Contents |
|------|---------|
| `server/app.py` | FastAPI app. 11 REST endpoints + WebSocket. CORS configured. Auth middleware hooked. |
| `server/models.py` | Pydantic request/response models: `PutRequest`, `GetResponse`, `EntryResponse`, `StatusResponse`, etc. |
| `server/ws.py` | `WebSocketManager` — maintains active WS connections, broadcasts events from EventBus |
| `server/__main__.py` | Entry point: `python -m dimensionalbase.server` (uvicorn on 0.0.0.0:8000) |

---

### `dimensionalbase/cli/` — CLI

| File | Contents |
|------|---------|
| `cli/main.py` | Click app. Commands: `db put <path> <value> --owner`, `db get <scope> --budget --query`, `db status`, `db delete <path>`, `db lineage <path>` |
| `cli/formatters.py` | Rich-based table and text formatters for CLI output |

---

### `dimensionalbase/integrations/` — Framework Adapters

| File | Class | Purpose |
|------|-------|---------|
| `integrations/langchain/memory.py` | `DimensionalBaseMemory` | LangChain `BaseMemory` adapter. `load_memory_variables()` calls `db.get()`, `save_context()` calls `db.put()`. |
| `integrations/langchain/tool.py` | `DimensionalBaseTool` | LangChain tool definition (BaseTool subclass) |
| `integrations/crewai/tool.py` | `DimensionalBaseTool` | CrewAI tool (BaseTool subclass, same interface) |
| `integrations/crewai/__init__.py` | Exports | |

---

### `dimensionalbase/security/` — Security Layer

| File | Class | Purpose |
|------|-------|---------|
| `security/auth.py` | `APIKeyAuth` | Generate, validate, revoke API keys. Stores hashed keys in SQLite. |
| `security/encryption.py` | `EntryEncryptor` | Fernet symmetric encryption for entry values at rest. Key from env var `DMB_ENCRYPTION_KEY`. |
| `security/acl.py` | `ACLEngine` | Path-based access control. `grant(agent, path, perms)`, `revoke()`, `check(agent, path, perm)`. Glob-based matching. |
| `security/validation.py` | `EntryValidator` | Input validation: path format (no leading slash, valid chars), confidence range (0.0–1.0), value length, owner format. |
| `security/middleware.py` | `SecurityMiddleware` | FastAPI middleware. Applied to all `/api/v1/` routes. Calls auth + ACL before handler. |

---

## `tests/` — Test Suite (13 files)

| File | Tests |
|------|-------|
| `tests/conftest.py` | Fixtures: `db()` factory (in-memory DimensionalBase), `populated_db()`, `agent_names` |
| `tests/test_core.py` | `put()`, `get()`, `subscribe()`, `unsubscribe()`, TTL behavior, path glob matching |
| `tests/test_algebra.py` | `compose()`, `relate()`, `project()`, `interpolate()`, `decompose()`, `analogy()` |
| `tests/test_intelligence.py` | Contradiction detection, gap detection, staleness detection, auto-summarization |
| `tests/test_embeddings.py` | Explicit embedding-provider integration, text-only fallback, semantic methods |
| `tests/test_adaptive_scoring.py` | Weight adaptation when query provided vs. not, recency vs. similarity boosting |
| `tests/test_concurrency.py` | 100 concurrent writes, concurrent reads during writes, no data corruption |
| `tests/test_persistence.py` | SQLite persistence across `db.close()` + reopen, version survival, TTL cleanup |
| `tests/test_e2e.py` | End-to-end: multi-agent deployment scenario, conflict resolution, budget constraints |
| `tests/test_errors.py` | `EntryValidationError`, `BudgetExhaustedError`, invalid paths, channel errors |
| `tests/test_security.py` | API key auth, at-rest encryption, ACL enforcement, input validation |
| `tests/test_mcp.py` | All 6 MCP tools, 3 MCP resources, error handling |
| `tests/__init__.py` | |

---

## `spec/` — Formal Protocol Specification (DBPS v1.0)

| File | Contents |
|------|---------|
| `spec/dbps-v1.0.md` | **95KB, 22 sections.** Formal spec for the Dimensional Base Protocol Standard. Covers: entry schema, path syntax, dimensional model, storage requirements, channel specs, ingestion pipeline, embedding, similarity search, confidence model, trust model, conflict resolution, REST API, WebSocket protocol, MCP integration, security requirements, error codes, versioning, conformance criteria |
| `spec/schemas/handshake.json` | JSON Schema for MCP handshake messages |
| `spec/schemas/knowledge-entry.json` | JSON Schema for entry serialization |
| `spec/schemas/event.json` | JSON Schema for event messages |
| `spec/grammar/path-syntax.abnf` | ABNF grammar for valid path strings |
| `spec/grammar/glob-matching.abnf` | ABNF grammar for glob patterns |
| `spec/test-vectors/glob-matching.json` | Test vectors: `(pattern, path) → bool` |
| `spec/test-vectors/confidence.json` | Test vectors: Bayesian confidence update sequences |
| `spec/test-vectors/trust.json` | Test vectors: Elo trust calculation sequences |

---

## `conformance/` — Spec Compliance Tests

| File | Tests |
|------|-------|
| `conformance/core/test_entry_serialization.py` | Entry to/from dict, to/from JSON, all fields preserved |
| `conformance/core/test_put_get_semantics.py` | API contract: put returns entry, get returns QueryResult, versions increment |
| `conformance/core/test_glob_matching.py` | All test vectors from `spec/test-vectors/glob-matching.json` |
| `conformance/embedding/test_normalization.py` | All stored embeddings must be L2-normalized (norm ≈ 1.0) |
| `conformance/reasoning/test_confidence_updates.py` | Bayesian update sequences match `spec/test-vectors/confidence.json` |
| `conformance/reasoning/test_trust_updates.py` | Elo trust calculations match `spec/test-vectors/trust.json` |
| `conformance/scoring/test_budget_packing.py` | Budget packing fills ≤budget tokens, highest-scored entries first |

---

## `benchmarks/` — Performance Benchmarks

| File | Purpose |
|------|---------|
| `benchmarks/definitive.py` | Main token comparison benchmark: 210 entries, 6 models, text-passing vs. DB |
| `benchmarks/agent_comms_bench.py` | Communication pattern benchmarks: sequential relay, fan-out, round-table, hierarchical |
| `benchmarks/multi_agent_bench.py` | Large-scale multi-agent test: 269 entries, 12 agents, 6 domains |
| `benchmarks/baselines.py` | Baseline `TextPasser` class used as Method A in all benchmarks |
| `benchmarks/real_world.py` | Real-world task scenario v1 |
| `benchmarks/real_world_v2.py` | Real-world task scenario v2 (improved methodology) |
| `benchmarks/run_all.py` | Runs all benchmarks sequentially |
| `benchmarks/standard/` | HotPotQA, LongBench v2, and other standardized eval suites |

---

## `examples/` — Usage Examples

| File | Demonstrates |
|------|-------------|
| `examples/basic_usage.py` | Simple 3-agent put/get/subscribe flow |
| `examples/langchain_integration.py` | DimensionalBaseMemory in a LangChain chain |
| `examples/crewai_integration.py` | DimensionalBaseTool in a CrewAI crew |
| `examples/mcp_usage.md` | How to configure Claude Code with mcp.json |
| `examples/contradiction_detection.py` | Deliberately create contradictions and watch CONFLICT events |
| `examples/budget_packing.py` | Show how budget packing works at different token limits |

---

## `papers/` — Research Papers

Background research and formalization documents supporting the DBPS v1.0 spec. Contains mathematical proofs for the Bayesian confidence model and Elo trust calculations.

---

## Dependency Tree

```
DimensionalBase.__init__
├── EmbeddingProvider (explicitly configured or null)
│   ├── LocalEmbeddingProvider → sentence-transformers
│   ├── OpenAIEmbeddingProvider → openai
│   └── NullEmbeddingProvider
│
├── VectorStore (shared across channels)
│
├── ChannelManager
│   ├── TextChannel → SQLite (migrations applied)
│   ├── EmbeddingChannel → VectorStore reference
│   └── TensorChannel (placeholder)
│
├── ContextEngine → VectorStore + EmbeddingProvider
│   └── LRU embedding cache
│
├── ActiveReasoning → ContextEngine + EventBus
│   ├── LSHFingerprint
│   └── BloomFilter
│
├── ConfidenceEngine
│
├── ProvenanceTracker
│
├── AgentTrustEngine
│
├── EventBus
│
└── DimensionalSpace → VectorStore
```
