# DimensionalBase — AI Agent Working Guide

> This file is specifically for AI agents (Claude, GPT, etc.) who are actively working on this codebase. It explains conventions, common tasks, gotchas, and patterns you need to follow to contribute correctly.

---

## Before You Touch Anything

Read these files first, in order:
1. `docs/00_START_HERE.md` — What this is
2. `docs/02_ARCHITECTURE.md` — How it's built
3. `dimensionalbase/db.py` — The main facade (how everything connects)

Then read the specific module you'll work on.

---

## Getting Started

Clone the repository and install in development mode:
```bash
git clone https://github.com/txtgrey/DimensionalBase.git
cd DimensionalBase
pip install -e ".[dev]"
pre-commit install
```

---

## Code Style Rules

**Formatter:** Black with 100-char line length (configured in `pyproject.toml`).
**Linter:** Ruff (configured in `pyproject.toml`).
**Type checker:** MyPy (strict mode for core modules).
**Pre-commit hooks:** Run Black → Ruff → MyPy on every commit.

To run manually before committing:
```bash
black dimensionalbase/ tests/ --line-length 100
ruff check dimensionalbase/ tests/
mypy dimensionalbase/
```

**Docstrings:** Use Google-style docstrings for all public methods:
```python
def put(self, path: str, value: str, owner: str) -> KnowledgeEntry:
    """Write a knowledge entry to the shared space.

    Args:
        path: Hierarchical path, e.g. "task/auth/status".
        value: Human-readable content. Also used as embedding source.
        owner: Agent identifier for trust tracking.

    Returns:
        The created KnowledgeEntry with generated id and embedding.

    Raises:
        EntryValidationError: If path format or confidence range is invalid.
    """
```

---

## Testing Rules

**Run tests:**
```bash
pytest tests/ -v
pytest tests/ --cov=dimensionalbase --cov-report=term-missing
```

**Coverage target:** 80%+ on `dimensionalbase/` core modules.

**Test patterns:**

All tests use the `db` fixture from `tests/conftest.py`:
```python
@pytest.fixture
def db():
    d = DimensionalBase(db_path=":memory:")
    yield d
    d.close()
```

Always use `:memory:` in tests (never file paths). Tests must be deterministic and isolated.

When testing async callbacks (subscribe), use threading.Event to wait for callback:
```python
received = threading.Event()
events = []

def callback(event):
    events.append(event)
    received.set()

sub = db.subscribe("task/**", "test", callback)
db.put("task/x", "value", owner="tester")
received.wait(timeout=1.0)
assert len(events) == 1
```

**For conformance tests** (`conformance/`), always test against the spec test vectors in `spec/test-vectors/`. These are the ground truth — implementation must match exactly.

---

## Adding a New Feature

### Step 1: Identify the right module

Use the module map in `docs/05_CODEBASE_MAP.md`. New features almost always touch:
1. A subsystem module (reasoning, trust, algebra, etc.)
2. `dimensionalbase/db.py` — expose it through the main API if needed
3. `dimensionalbase/__init__.py` — export any new public types
4. A test file in `tests/`

### Step 2: Check if it belongs in the spec

If the feature is fundamental to the protocol (changes how `put`/`get` work, changes entry schema, changes confidence model, etc.), it should be reflected in `spec/dbps-v1.0.md` and a conformance test should be added to `conformance/`.

If it's an implementation detail (optimization, new algebra operation, new integration), no spec change needed.

### Step 3: Don't break the 4-method API

The public agent API is exactly 4 methods: `put`, `get`, `subscribe`, `unsubscribe`. New capabilities are always:
- Added as additional methods on `DimensionalBase` (introspection, algebra)
- Added to the MCP server as new tools
- Exposed in the REST API as new endpoints
- **Never** by changing the signature of `put`/`get`/`subscribe`/`unsubscribe`

### Step 4: Update CHANGELOG.md

Every meaningful change goes in `CHANGELOG.md` under the appropriate version section.

---

## Common Tasks

### Add a new algebra operation

1. Add the function to `dimensionalbase/algebra/operations.py`
2. Call it from `dimensionalbase/db.py` (add a wrapper method)
3. Export from `dimensionalbase/__init__.py` if needed
4. Add a test to `tests/test_algebra.py`
5. Add a tool to `dimensionalbase/mcp/server.py` if it makes sense as an MCP tool
6. Add an endpoint to `dimensionalbase/server/app.py`

Example pattern:
```python
# In algebra/operations.py
def my_new_op(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Compute something interesting between two vectors."""
    # Both vectors are pre-normalized (L2 norm = 1.0)
    # Use numpy operations only — no scipy unless it's in dependencies
    return float(np.dot(vector_a, vector_b))

# In db.py
def my_new_op(self, path_a: str, path_b: str) -> float:
    """Public API wrapper."""
    entry_a = self.retrieve(path_a)
    entry_b = self.retrieve(path_b)
    if entry_a is None or entry_b is None:
        raise EntryValidationError(f"One or both paths not found")
    return operations.my_new_op(entry_a.embedding, entry_b.embedding)
```

### Add a new event type

1. Add to `EventType` enum in `dimensionalbase/core/types.py`
2. Emit from the relevant subsystem via `self._event_bus.emit(Event(type=NEW_TYPE, ...))`
3. Update `spec/schemas/event.json` with the new type
4. Add test to `tests/test_core.py` (subscribe and verify event fires)

### Add a new REST endpoint

1. Add route to `dimensionalbase/server/app.py`
2. Add request/response models to `dimensionalbase/server/models.py`
3. Add to the REST endpoint table in `docs/03_API_REFERENCE.md`

### Add a new framework integration

1. Create `dimensionalbase/integrations/{framework}/` directory
2. Add `tool.py` and/or `memory.py` following existing patterns
3. Add `{framework}` to extras in `pyproject.toml`
4. Add tests to `tests/`

---

## Architecture Invariants — Never Violate These

### 1. All stored embeddings must be L2-normalized
```python
# CORRECT: normalize before storing
embedding = embedding / np.linalg.norm(embedding)
vector_store.add(path, embedding)

# WRONG: store unnormalized
vector_store.add(path, embedding)  # dot product ≠ cosine similarity
```

This invariant is tested in `conformance/embedding/test_normalization.py`.

### 2. VectorStore is the single source of truth for embeddings
Never create a second numpy array to store embeddings. All embeddings go through `VectorStore`. The channels (TextChannel, EmbeddingChannel) share the same VectorStore instance.

### 3. The 4-method API signatures are frozen
Do not change the signatures of `put()`, `get()`, `subscribe()`, `unsubscribe()`. These are the protocol — changing them breaks every agent using the system.

### 4. ActiveReasoning must not block `put()`
`ActiveReasoning.on_write()` is called synchronously on every `put()`. It must complete quickly. Any operation that could be slow (full scan, API calls, complex math) must be pre-filtered with LSH fingerprinting or deferred to a background thread.

### 5. Thread safety via VectorStore's RLock
All reads/writes to the VectorStore go through its `RLock`. Do not access `_matrix`, `_paths`, or `_path_to_idx` directly from outside `VectorStore`. Use the public methods.

### 6. Budget must always be respected
`get()` must never return more tokens than `budget`. The budget packing in `ContextEngine` enforces this with the hierarchical fallback. If even path-only doesn't fit, skip the entry.

---

## Key Design Patterns

### Pattern: Vectorized batch operations
Always prefer one matrix operation over N individual operations:
```python
# CORRECT: one BLAS call
sims = self._vector_store._matrix[:n] @ query_embedding  # shape (N,)

# WRONG: N individual calls
sims = [np.dot(self._vector_store.get(path), query_embedding) for path in paths]
```

### Pattern: Optional subsystem
When a subsystem is optional (e.g., embedding provider may not be available), check before calling:
```python
if self._embedding_provider is not None and self._embedding_provider.is_available:
    embedding = self._embedding_provider.encode(value)
else:
    embedding = None
```

### Pattern: Emit events after writes
Every `put()` must emit a `CHANGE` event. Deletes must emit `DELETE`. Never emit before the write is committed.

### Pattern: Hierarchical path domains
Domain extraction is used by the trust engine. The convention is that the second segment of a path is the domain:
- `task/auth/status` → domain `auth`
- `task/deploy/log` → domain `deploy`
- `config/db/host` → domain `db`

If a path has only one segment, the domain is the segment itself.

### Pattern: Confidence as float, Beta as internal state
The `confidence` field on a `KnowledgeEntry` is the **self-reported** confidence (0.0–1.0), set by the writing agent. The `ConfidenceEngine` maintains internal Beta distribution state separately. When you call `get()`, the scoring uses the Beta-posterior mean, not the raw self-reported value. This is intentional — it discounts agents that frequently get contradicted.

---

## Gotchas and Known Issues

### Gotcha 1: `:memory:` databases don't persist
If you create a `DimensionalBase(db_path=":memory:")`, all data is lost when `db.close()` is called or the process exits. For persistent data across sessions, use a file path: `DimensionalBase(db_path="/path/to/knowledge.db")`.

### Gotcha 2: Embeddings are explicit; plain startup is text-only
Plain `DimensionalBase()` starts in text-only mode (`NullEmbeddingProvider`). To enable semantic search, pass `embedding_provider=...`, `prefer_embedding="local"`, or `openai_api_key=...`. In text-only mode:
- `get()` returns entries based on recency + confidence only (no semantic similarity)
- `relate()` raises `ChannelError`
- `compose()` raises `ChannelError`
- `channel_used` in `QueryResult` will be `ChannelLevel.TEXT`

Check `db.status()["embedding_provider"]` to confirm which provider is active.

### Gotcha 3: TensorChannel always raises NotImplementedError
`channels/tensor.py` is a Phase 4 placeholder. Don't try to use it. Startup does not probe or select TensorChannel unless tensor probing is explicitly enabled.

### Gotcha 4: TTL cleanup is not automatic during operation
`TTL.TURN` entries must be explicitly cleaned up by calling `db.clear_turn()` at the end of a turn. `TTL.SESSION` entries must be explicitly cleaned with `db.clear_session()`. `db.close()` does not delete session state in `v0.4`. `TTL.PERSISTENT` entries are never automatically cleaned.

### Gotcha 5: Glob patterns in `get()` scope vs. exact paths
`get(scope="task/auth/status")` with no wildcards matches ONLY that exact path. To get all entries under a prefix, you must use `task/auth/**` or `task/auth/*`.

### Gotcha 6: `refs` creates graph edges but not automatic retrieval
Setting `refs=["task/auth/status"]` in a `put()` call creates a graph edge. It increases the ref_proximity score of `task/auth/status` when any related entry is returned. But it does NOT automatically cause `task/auth/status` to be fetched — it still needs to be in scope.

### Gotcha 8: Re-ranking requires sentence-transformers
`rerank=True` lazy-loads a cross-encoder model (~22MB). First query takes 2-5 seconds for model load. Subsequent queries add ~50-100ms. Only useful when embeddings are enabled. Falls back gracefully if sentence-transformers is not installed.

### Gotcha 9: Variance reweighting changes scoring automatically
When all entries have the same owner/confidence/recency (e.g. single-document ingestion), the scoring engine automatically shifts weight to similarity. This is by design — uniform signals can't differentiate entries. In multi-agent scenarios with diverse sources, balanced weights are preserved.

### Gotcha 10: Context window expansion adds adjacent chunks
For single-owner workloads, `get()` automatically pulls in adjacent sequential chunks (chunk/41 alongside chunk/42). This improves document QA but increases token usage slightly. Only triggers when entries follow a `prefix/N` path pattern.

### Gotcha 7: Concurrent writes to the same path
SQLite serializes concurrent writes. If two agents try to `put()` the same path simultaneously, one will block on the SQLite write lock. The winner increments the version; the loser's write completes after. Both entries are stored in the version history. No data is lost, but the final version will be whichever write completed last.

---

## Running the Full Stack Locally

```bash
# Install everything
pip install -e ".[dev,embeddings-local,mcp,server,cli,langchain,crewai,security]"

# Run tests
pytest tests/ -v

# Run REST API
python -m dimensionalbase.server
# API now at http://localhost:8000

# Run MCP server
python -m dimensionalbase.mcp
# MCP server ready on stdio

# Use CLI
db put task/test "hello world" --owner agent1
db get task/** --budget 100
db status
```

---

## Version Increment Checklist

When bumping the version (e.g., 0.3.0 → 0.4.0):

1. `dimensionalbase/__init__.py` — update `__version__`
2. `pyproject.toml` — update `version`
3. `CHANGELOG.md` — add new version section
4. If any spec changes: `spec/dbps-v1.0.md` — update version references
5. Run full test suite + conformance suite
6. Run all benchmarks to verify no regression
