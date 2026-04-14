# DimensionalBase — Technical Architecture

## System Overview

DimensionalBase is composed of 20 Python modules organized into 10 subsystems. All subsystems are accessed through a single facade class: `DimensionalBase` in `dimensionalbase/db.py`.

```
┌─────────────────────────────────────────────────────────────┐
│                     Agent Code                              │
│         db.put() / db.get() / db.subscribe()                │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              DimensionalBase (db.py)                        │
│              Public API Facade                              │
└──┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬────────┘
   │      │      │      │      │      │      │      │
   ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼
Channel Context Algebra Reason  Trust Events Embed Security
Manager Engine  Ops    Engine  Engine  Bus  Provider  Layer
```

---

## Module-by-Module Reference

### `dimensionalbase/db.py` — Main Facade (517 LOC)

The only class agents interact with. Instantiates and wires together all subsystems.

**Constructor parameters:**
```python
DimensionalBase(
    db_path=":memory:",           # SQLite path, or ":memory:" for in-process
    embedding_provider=None,      # Custom embedding provider
    prefer_embedding=None,        # "local" or "openai" to explicitly enable a provider
    openai_api_key=None,          # Explicit OpenAI key for embedding mode
    scoring_weights=None,         # Override default context scoring weights
    staleness_threshold=3600.0,   # Seconds before an entry is considered stale
    auto_reasoning=True,          # Whether to run ActiveReasoning on every write
    config=None,                  # DimensionalBaseConfig for all tuneable thresholds
    rerank=False,                 # Cross-encoder re-ranking for document QA accuracy
)
```

**What happens on init:**
1. EmbeddingProvider resolved explicitly (custom → local/openai preference → text-only)
2. VectorStore initialized (shared float32 contiguous array)
3. ChannelManager initialized (wires TextChannel + EmbeddingChannel to shared VectorStore)
4. ContextEngine initialized (budget-aware retrieval)
5. ActiveReasoning initialized (contradiction/gap/staleness detection)
6. ConfidenceEngine initialized (Bayesian Beta distributions)
7. AgentTrustEngine initialized (Elo-like trust tracking)
8. ProvenanceTracker initialized (full derivation DAG)
9. EventBus initialized (pub/sub)
10. DimensionalSpace initialized (manifold analytics)

---

### `dimensionalbase/core/` — Type System

**`core/types.py`** — All enums:
```python
class EntryType(str, Enum):
    FACT = "fact"
    DECISION = "decision"
    PLAN = "plan"
    OBSERVATION = "observation"

class TTL(str, Enum):
    TURN = "turn"        # Deleted after one agent turn
    SESSION = "session"  # Deleted when session ends
    PERSISTENT = "persistent"  # Never automatically deleted

class ChannelLevel(str, Enum):
    TEXT = "text"           # Channel 1: SQLite text
    EMBEDDING = "embedding" # Channel 2: Semantic vectors
    TENSOR = "tensor"       # Channel 3: Raw KV cache (future)

class EventType(str, Enum):
    CHANGE = "change"     # Entry written/updated
    DELETE = "delete"     # Entry deleted
    CONFLICT = "conflict" # Contradiction detected
    GAP = "gap"           # Plan step missing observation
    STALE = "stale"       # Entry exceeded staleness threshold
    SUMMARY = "summary"   # Prefix auto-summarized
```

**`core/entry.py`** — The atomic unit of knowledge:
```python
@dataclass
class KnowledgeEntry:
    path: str                           # e.g. "task/auth/status"
    value: str                          # Human-readable text
    owner: str                          # Which agent wrote this
    type: EntryType = EntryType.FACT
    confidence: float = 1.0             # Self-reported, 0.0–1.0
    refs: List[str] = field(default_factory=list)  # Linked paths
    embedding: Optional[np.ndarray] = None
    version: int = 1
    ttl: TTL = TTL.SESSION
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, str] = field(default_factory=dict)
```

The `refs` field creates the knowledge graph. Entries that reference each other score higher in retrieval when any of them is returned.

**`core/matching.py`** — Glob pattern matching for paths:
- `task/**` matches `task/auth/status`, `task/deploy/error`, etc.
- `task/*/status` matches `task/auth/status`, `task/db/status`, etc.
- Used by both `get()` scope filtering and `subscribe()` pattern matching

---

### `dimensionalbase/channels/` — 3-Channel Communication

The channel layer abstracts how knowledge is transmitted. The same `put`/`get` API works regardless of which channel is active.

**`channels/manager.py`** — ChannelManager:

Channel selection logic (runs at init):
```python
if embedding_provider or prefer_embedding or openai_api_key:
    → use EmbeddingChannel
else:
    → use TextChannel
```

The ChannelManager:
- Creates the shared VectorStore
- Initializes all available channels
- Routes writes to all channels (text always + best available)
- Routes reads to the highest-quality available channel
- Returns `ChannelLevel` in every `QueryResult` so agents know which channel served them

**`channels/text.py`** — TextChannel (Channel 1):
- Always available, always active
- Stores to SQLite: path, value, owner, type, confidence, refs, version, ttl, timestamps
- Supports glob path queries
- No embeddings — text matching only

**`channels/embedding.py`** — EmbeddingChannel (Channel 2):
- Available when an embedding provider is detected
- Stores embeddings in the shared VectorStore
- On `get()`, performs semantic similarity search against the query embedding
- Returns entries sorted by similarity score (combined with recency/confidence/refs)

**`channels/tensor.py`** — TensorChannel (Channel 3):
- Placeholder for Phase 4
- Will transmit raw KV cache slices over RDMA
- Zero information loss — no compression at all
- Requires agents on shared GPU hardware

---

### `dimensionalbase/storage/` — Storage Layer

**`storage/vectors.py`** — Unified VectorStore (v0.3 key innovation):

The single source of truth for all embeddings. Before v0.3, each channel had its own array. Now there's one.

```python
class VectorStore:
    _matrix: np.ndarray      # Shape (capacity, dim), float32, pre-normalized
    _paths: List[str]        # Index → path
    _path_to_idx: Dict[str, int]  # path → index
    _lock: threading.RLock   # Thread-safe, re-entrant

    def add(path: str, vector: np.ndarray) -> int    # O(1) amortized
    def remove(path: str) -> bool                    # O(1) swap-with-last
    def get(path: str) -> Optional[np.ndarray]       # O(1)
    def search(query: np.ndarray, k: int) -> List[Tuple[str, float]]  # O(n) + O(n log k)
    def all_similarities(query: np.ndarray) -> np.ndarray  # O(n) single BLAS
```

Key properties:
- **All vectors L2-normalized on write**: `dot(a, b) = cosine_similarity(a, b)`. No normalization in the search hot path.
- **Contiguous float32 array**: `_matrix @ query` is a single BLAS matrix-vector multiply scoring all N entries simultaneously.
- **Swap-on-delete**: When an entry is removed, the last row swaps into its slot. No gaps, no compaction needed.
- **Thread-safe with RLock**: Multiple readers can hold the lock simultaneously; writers get exclusive access.

**`storage/migrations.py`** — SQLite Schema Migrations:
- Versioned schema evolution
- Safe upgrades without data loss
- Applied automatically on `DimensionalBase()` init

---

### `dimensionalbase/algebra/` — Dimensional Algebra

Mathematical operations on the vector space. These let agents query the geometry of the shared knowledge, not just its contents.

**`algebra/operations.py`** — 8 operations:

```python
# 1. COMPOSE — Synthesize multiple vectors into one
compose(
    entries: List[KnowledgeEntry],
    mode: Literal["weighted_mean", "principal", "grassmann", "attentive"] = "attentive"
) -> np.ndarray
# weighted_mean: simple weighted average (fast)
# principal: PCA first component (captures main direction)
# grassmann: Karcher mean on Riemannian manifold (mathematically exact)
# attentive: soft attention weights (best for conflicting beliefs)

# 2. RELATE — Discover relationship between two vectors
relate(a: np.ndarray, b: np.ndarray) -> Dict[str, float]
# Returns: {cosine, angular_dist, projection, residual, parallelism, opposition, independence}
# Use: "How do these two concepts relate?"

# 3. PROJECT — Map into subspace
project(vector: np.ndarray, basis: np.ndarray) -> Tuple[np.ndarray, float]
# Returns: (projected_vector, fraction_of_information_retained)
# Use: "What does agent A's knowledge look like from agent B's perspective?"

# 4. INTERPOLATE — Semantic midpoint
interpolate(a: np.ndarray, b: np.ndarray, t: float = 0.5) -> np.ndarray
# Returns: normalized vector at position t between a and b (spherical lerp)
# Use: "Find a concept halfway between X and Y"

# 5. DECOMPOSE — Factor into components
decompose(vector: np.ndarray, n_components: int = 3) -> List[np.ndarray]
# Returns: list of orthogonal component vectors
# Use: "What are the main aspects of this concept?"

# 6. CENTROID — Weighted geometric mean
centroid(vectors: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray

# 7. ANALOGY — Vector arithmetic
analogy(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray
# Returns: c + (b - a), normalized
# "king - man + woman ≈ queen"

# 8. SUBSPACE_ALIGNMENT — How aligned are two knowledge sets?
subspace_alignment(set_a: List[np.ndarray], set_b: List[np.ndarray]) -> float
# Returns: 0.0 (orthogonal) to 1.0 (identical)
```

**`algebra/space.py`** — DimensionalSpace:
Analytics layer over the VectorStore. Computes running statistics on the knowledge manifold:
- Running mean and variance (Welford incremental algorithm — no full recompute on every write)
- Intrinsic dimensionality estimation (participation ratio)
- Cluster detection and density analysis
- Coverage computation (what fraction of concept space is covered?)
- Novelty scoring (how much new information does a candidate entry add?)

**`algebra/fingerprint.py`** — LSH Fingerprinting:
- Locality Sensitive Hashing for fast approximate nearest-neighbor pre-filtering
- Used by ActiveReasoning to narrow contradiction candidates from O(N) to O(20)
- Bloom filter for novelty skip — if an entry is very similar to an existing one, skip it

---

### `dimensionalbase/context/` — Budget-Aware Context Engine

**`context/engine.py`** — ContextEngine:

The retrieval layer. Called on every `get()`.

**Scoring formula (vectorized):**
```python
score = (
    weights.recency     × recency_score(entry)     +  # 30% default
    weights.confidence  × entry.confidence          +  # 20% default
    weights.similarity  × cosine_sim(query, entry)  +  # 30% default
    weights.ref_dist    × ref_graph_proximity(entry)   # 20% default
)
```

All candidates scored in **one BLAS call**:
```python
sims = vector_matrix @ query_embedding  # Shape (N,) — all N entries scored simultaneously
```

**Adaptive weighting:**
- Query provided → similarity weight boosted (most relevant to task)
- No query → recency weight boosted (freshest information)

**Budget packing (hierarchical fallback):**
```
For each entry (sorted by score, highest first):
1. Try full:     "[path] (owner, conf=0.92, type=fact): value"
2. Try compact:  "[path]: truncated_value..."
3. Try path:     "path"
4. Skip:         if even path-only doesn't fit budget
```

**Optimizations (v0.3):**
- `heapq.nlargest(k, scores)` — O(n log k) partial sort, not O(n log n) full sort
- LRU embedding cache (TTL=60s) — same query string = one embedding API call
- Incremental reference graph — updated on write, not rebuilt on read
- Pre-normalized embeddings — no norm computation in hot path

**`context/compression.py`** — SemanticCompressor:
- Delta encoding: if two entries are very similar, only store the diff
- Deduplication: hash-based exact duplicate removal
- Used when packing context hits tight budget constraints

---

### `dimensionalbase/reasoning/` — Active Reasoning

The "fourth participant" — a module that watches every write and fires events when it detects problems.

**`reasoning/active.py`** — ActiveReasoning:

Runs on every `put()` call (when `auto_reasoning=True`):

**Contradiction Detection:**
```python
candidates = lsh_fingerprint.query(new_entry.embedding, k=20)  # Fast pre-filter
for existing in candidates:
    similarity = dot(new_entry.embedding, existing.embedding)  # Already normalized
    if similarity >= 0.75 and new_entry.value != existing.value:
        event_bus.emit(Event(type=CONFLICT, path=new_entry.path, ...))
```
Threshold 0.75: semantically very similar, but different values = contradiction.

**Gap Detection:**
```python
if new_entry.type == PLAN:
    for ref in new_entry.refs:
        observations = get_entries(scope=ref, type=OBSERVATION)
        if not observations:
            event_bus.emit(Event(type=GAP, path=ref, ...))
```

**Staleness Detection:**
```python
for entry in all_entries:
    if (now - entry.updated_at) > staleness_threshold:  # default 3600s
        event_bus.emit(Event(type=STALE, path=entry.path, ...))
```

**Auto-Summarization:**
```python
entries_at_prefix = get_entries(scope=f"{prefix}/**")
if len(entries_at_prefix) > SUMMARY_THRESHOLD:  # default 10
    summary = generate_summary(entries_at_prefix)
    put(f"{prefix}/_summary", summary, owner="system")
    event_bus.emit(Event(type=SUMMARY, path=prefix, ...))
```

**`reasoning/confidence.py`** — ConfidenceEngine:

Tracks confidence as Bayesian Beta distributions. Each entry has a `ConfidenceState(alpha, beta)`.

```python
# Initial state: weak prior (alpha=1, beta=1), shifted by agent's self-reported confidence
alpha = 1 + (agent_confidence * 2)
beta = 1 + ((1 - agent_confidence) * 2)

# Mean confidence
mean = alpha / (alpha + beta)

# Variance (uncertainty about the confidence itself)
variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))

# When another agent confirms this entry:
alpha += 1

# When another agent contradicts this entry:
beta += 2  # Conservative — doubt weighs more than confirmation
```

**`reasoning/provenance.py`** — ProvenanceTracker:

Full DAG tracking of knowledge derivation. 7 derivation types:
1. `DIRECT` — Original source, no derivation
2. `CONFIRMED` — Another agent confirmed this
3. `CONTRADICTED` — Another agent contradicted this
4. `COMPOSED` — Synthesized from multiple entries via `compose()`
5. `PROJECTED` — Dimensionally reduced via `project()`
6. `INFERRED` — Derived from patterns in the knowledge space
7. `MATERIALIZED` — Projected back to text from an embedding

Every `put()` records a `ProvenanceNode` with: entry_id, derivation_type, parent_ids, agent_id, timestamp.

---

### `dimensionalbase/trust/` — Agent Trust Model

**`trust/agent_trust.py`** — AgentTrustEngine:

Tracks which agents are reliable in which domains.

**AgentProfile:**
```python
@dataclass
class AgentProfile:
    agent_id: str
    global_trust: float = 0.5              # 0.0–1.0 (initialized at 0.5)
    domain_trust: Dict[str, float] = {}    # "database" → 0.8, "ui" → 0.3
    total_entries: int = 0
    total_confirmations: int = 0
    total_contradictions: int = 0
    confirmation_rate: float = 0.0         # confirmations / (confirmations + contradictions)
    is_reliable: bool = False              # True after ≥5 interactions
    last_activity: float                   # For decay calculation
```

**Trust update (Elo-like):**
```python
K_FACTOR = 32  # Same as chess Elo

# Agent B confirms Agent A's entry:
trust_diff = trust_A - trust_B
expected_confirmation_prob = 1.0 / (1.0 + exp(-trust_diff * 4))
surprise = 1.0 - expected_confirmation_prob
delta = K_FACTOR * surprise / 100

trust_A += delta   # Confirmed entries increase confirmer's trust
# (if unexpected confirmation → bigger delta → bigger update)

# Agent B contradicts Agent A's entry:
# Same formula but negative delta, weighted heavier
```

Domain extraction from path: `"task/auth/status"` → domain `"auth"`. Domain trust updated independently from global trust.

---

### `dimensionalbase/events/` — Event Bus

**`events/bus.py`** — EventBus:

Pub/sub system. Subscribers register glob patterns. Events are broadcast to matching subscribers.

```python
@dataclass
class Event:
    type: EventType
    path: str
    entry: Optional[KnowledgeEntry]
    payload: Dict[str, Any] = {}
    timestamp: float = field(default_factory=time.time)
```

**Subscription matching:**
- Pattern `"task/**"` matches any event at a path under `task/`
- Pattern `"task/*/status"` matches only status events
- Pattern `"**"` matches everything

**Implementation:**
- In-memory subscriber registry (dict of pattern → callback)
- Event buffer for clients without persistent listeners (ring buffer, last 100 events)
- Thread-safe emit (locks subscriber list before iterating)
- WebSocket broadcast when REST API is running (`server/ws.py`)

---

### `dimensionalbase/embeddings/` — Embedding Provider

**`embeddings/provider.py`** — EmbeddingProvider:

Provider resolution at init:
```python
if prefer == "local":
    # → LocalEmbeddingProvider
elif prefer == "openai" or openai_api_key:
    # → OpenAIEmbeddingProvider
else:
    # → NullEmbeddingProvider (text-only mode, Channel 1 only)
```

**Interface:**
```python
class EmbeddingProvider:
    def encode(text: str) -> np.ndarray           # Single string
    def encode_batch(texts: List[str]) -> np.ndarray  # Batch (more efficient)
    @property
    def dimension() -> int                         # Vector dimensionality
    @property
    def provider_name() -> str                     # "local" or "openai"
```

The LRU cache in ContextEngine wraps `encode()` — repeated queries with the same text reuse the cached vector (60s TTL).

---

### `dimensionalbase/mcp/` — MCP Server

**`mcp/server.py`** — Exposes DimensionalBase to Claude Code, Cursor, Windsurf via the Model Context Protocol.

**6 Tools:**
```
db_put        — Write a knowledge entry
db_get        — Read with scope + budget
db_relate     — Compute relationship between two paths
db_compose    — Merge multiple entries into one vector
db_status     — System status (entry count, channel, trust summary)
db_subscribe  — Watch a path pattern (returns subscription ID)
```

**3 Resources (readable context):**
```
dimensionalbase://entries/recent   — Last 20 entries
dimensionalbase://events/recent    — Last 20 events (conflicts, gaps, etc.)
dimensionalbase://status           — Full system status object
```

**`mcp/mcp.json` (project root):**
```json
{
  "mcpServers": {
    "dimensionalbase": {
      "command": "python",
      "args": ["-m", "dimensionalbase.mcp"]
    }
  }
}
```

---

### `dimensionalbase/server/` — REST API

**`server/app.py`** — FastAPI application.

**Endpoints:**
```
POST   /api/v1/entries              db.put()
GET    /api/v1/entries?scope=&query= db.get()
GET    /api/v1/entries/{path}       db.retrieve()
DELETE /api/v1/entries/{path}       db.delete()
POST   /api/v1/relate               db.relate()
POST   /api/v1/compose              db.compose()
GET    /api/v1/status               db.status()
GET    /api/v1/trust                db.agent_trust_report()
GET    /api/v1/topology             db.knowledge_topology()
GET    /api/v1/lineage/{path}       db.lineage()
WS     /ws/subscribe                Real-time event stream
```

**`server/models.py`** — Pydantic request/response models (validated on every request).

**`server/ws.py`** — WebSocket manager: maintains active connections, broadcasts events from EventBus.

---

### `dimensionalbase/integrations/` — Framework Adapters

**`integrations/langchain/memory.py`** — DimensionalBaseMemory:
Implements LangChain's `BaseMemory` interface.
```python
def load_memory_variables(inputs: Dict) -> Dict:
    # Calls db.get() with inputs["input"] as query
    # Returns {"history": result.text}

def save_context(inputs: Dict, outputs: Dict) -> None:
    # Calls db.put() with the AI's output as an observation entry
```

**`integrations/langchain/tool.py`** — DimensionalBaseTool:
LangChain tool that lets agents explicitly read/write from within chain steps.

**`integrations/crewai/tool.py`** — DimensionalBaseTool (CrewAI):
CrewAI-specific tool adapter with the same `put`/`get` interface.

---

### `dimensionalbase/security/` — Auth & Encryption

**`security/auth.py`** — API key authentication for REST endpoints.

**`security/encryption.py`** — At-rest encryption for entry values (SQLite column-level).

**`security/acl.py`** — Path-based access control:
```python
# Allow agent "backend" to write to "task/auth/**"
# Block agent "ui" from reading "task/secrets/**"
acl.grant(agent="backend", path="task/auth/**", permissions=["read", "write"])
```

**`security/validation.py`** — Input validation (path format, value length, confidence range).

**`security/middleware.py`** — FastAPI middleware that enforces auth + ACL on every request.

---

## Data Flow: `put()` call

```
db.put("task/auth/status", "JWT expired", owner="agent-backend")

1. validation.validate(path, value, owner, ...)
2. embedding = embedding_provider.encode("JWT expired")  # or cache hit
3. embedding = l2_normalize(embedding)
4. entry = KnowledgeEntry(path=..., value=..., embedding=..., ...)
5. text_channel.write(entry)           # → SQLite
6. vector_store.add(path, embedding)   # → contiguous float32 array
7. event_bus.emit(Event(CHANGE, path)) # → all matching subscribers
8. if auto_reasoning:
    active_reasoning.on_write(entry)   # Contradiction/gap/staleness check
    confidence_engine.register(entry)
    provenance_tracker.record(entry)
    trust_engine.record_write(entry.owner)
9. return entry
```

## Data Flow: `get()` call

```
db.get("task/**", budget=500, query="what is blocking deployment?")

1. q_embedding = embedding_provider.encode(query)  # or LRU cache hit
2. candidates = text_channel.list(scope="task/**") # All entries under task/
3. sims = vector_store.all_similarities(q_embedding)  # ONE BLAS call → (N,) array
4. scores = recency*0.3 + confidence*0.2 + sims*0.3 + ref_graph*0.2
5. top_k = heapq.nlargest(k, scores)              # O(n log k)
6. packed = budget_pack(top_k, budget=500)         # hierarchical fallback
7. return QueryResult(entries=packed, text=format(packed), tokens_used=..., channel=EMBEDDING)
```
