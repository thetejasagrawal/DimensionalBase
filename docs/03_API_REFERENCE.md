# DimensionalBase — Complete API Reference

## Import

```python
from dimensionalbase import DimensionalBase
from dimensionalbase.core.types import EntryType, TTL, ChannelLevel, EventType
from dimensionalbase.core.entry import KnowledgeEntry
```

---

## `DimensionalBase` — Main Class

### Constructor

```python
db = DimensionalBase(
    db_path: str = ":memory:",
    embedding_provider: Optional[EmbeddingProvider] = None,
    prefer_embedding: Optional[Literal["local", "openai"]] = None,
    openai_api_key: Optional[str] = None,
    scoring_weights: Optional[ScoringWeights] = None,
    staleness_threshold: float = 3600.0,
    auto_reasoning: bool = True,
    config: Optional[DimensionalBaseConfig] = None,
    rerank: bool = False,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | `str` | `":memory:"` | SQLite database path. Use `":memory:"` for tests, a file path for persistence. |
| `embedding_provider` | `EmbeddingProvider` | `None` | Custom embedding provider. Use this when you want deterministic semantic behavior. |
| `prefer_embedding` | `"local"` or `"openai"` | `None` | Explicitly enable the built-in local or OpenAI provider. |
| `openai_api_key` | `str` | `None` | Explicit key for OpenAI embeddings. Plain `DimensionalBase()` does not auto-enable OpenAI from the environment. |
| `scoring_weights` | `ScoringWeights` | `None` | Override default 30/20/30/20 (recency/confidence/similarity/refs). |
| `staleness_threshold` | `float` | `3600.0` | Seconds after which entries are considered stale. |
| `auto_reasoning` | `bool` | `True` | Whether to run ActiveReasoning on every write. Set False for batch imports. |
| `config` | `DimensionalBaseConfig` | `None` | Central config for all tuneable thresholds. Overrides individual params. |
| `rerank` | `bool` | `False` | Enable cross-encoder re-ranking for higher retrieval accuracy. Adds ~50-100ms per query. |

---

## The 4-Method Agent API

### `put()` — Write Knowledge

```python
entry: KnowledgeEntry = db.put(
    path: str,
    value: str,
    owner: str,
    type: Union[EntryType, str] = "fact",
    confidence: float = 1.0,
    refs: Optional[List[str]] = None,
    ttl: Union[TTL, str] = "session",
    metadata: Optional[Dict[str, str]] = None,
) -> KnowledgeEntry
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Hierarchical path, e.g. `"task/auth/status"`. Use `/` as separator. No leading slash. |
| `value` | `str` | Human-readable content. Also used as the text to encode into an embedding. |
| `owner` | `str` | Agent identifier. Used for trust tracking and filtering. |
| `type` | `EntryType` or `str` | `"fact"`, `"decision"`, `"plan"`, or `"observation"`. |
| `confidence` | `float` | Self-reported confidence, 0.0–1.0. Used in context scoring. |
| `refs` | `List[str]` | List of related paths. Creates graph edges — referenced entries score higher in retrieval. |
| `ttl` | `TTL` or `str` | `"turn"` (one agent turn), `"session"` (default), `"persistent"` (never auto-deleted). |
| `metadata` | `Dict[str, str]` | Arbitrary string key-value pairs for extensibility. |

**Returns:** The complete `KnowledgeEntry` with generated `id`, `embedding`, `version=1`, timestamps.

**Side effects:**
- Writes to SQLite (TextChannel)
- Writes embedding to VectorStore (EmbeddingChannel, when embeddings are explicitly enabled)
- Fires `CHANGE` event on EventBus
- Runs ActiveReasoning (if `auto_reasoning=True`): may fire `CONFLICT`, `GAP`, `STALE`, `SUMMARY` events
- Updates ConfidenceEngine, ProvenanceTracker, AgentTrustEngine

**If the path already exists:** Creates a new version. `entry.version` is incremented. Old version retained in SQLite for lineage.

**Examples:**
```python
# Basic fact
db.put("task/auth/status", "JWT signing key expired. Auth returning 401.", owner="agent-backend")

# High-confidence decision with refs
db.put(
    "task/decision/rollback",
    "Rolling back to v2.1.3. Auth failure rate exceeded 50%.",
    owner="agent-planner",
    type="decision",
    confidence=0.97,
    refs=["task/auth/status", "task/metrics/error-rate"],
    ttl="persistent",
)

# Observation
db.put(
    "task/deploy/observation",
    "Deployment completed in 4m 32s. No errors.",
    owner="agent-deploy",
    type="observation",
    confidence=1.0,
)
```

---

### `get()` — Read Knowledge

```python
result: QueryResult = db.get(
    scope: str,
    budget: int = 2000,
    query: Optional[str] = None,
    owner: Optional[str] = None,
    type: Optional[Union[EntryType, str]] = None,
) -> QueryResult
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `scope` | `str` | Glob pattern for paths. `"task/**"` returns all entries under `task/`. `"**"` returns all entries. |
| `budget` | `int` | Maximum tokens to return. Default 2000. The system packs the most relevant entries that fit. |
| `query` | `str` or `None` | Semantic query string. When provided, similarity to this query boosts entry scores. When absent, recency is boosted. |
| `owner` | `str` or `None` | Filter by agent. Only returns entries written by this agent. |
| `type` | `EntryType` or `None` | Filter by entry type. e.g. `"fact"` or `"decision"`. |

**Returns:** `QueryResult`

```python
@dataclass
class QueryResult:
    entries: List[KnowledgeEntry]  # Scored entries, highest first
    text: str                      # Pre-formatted text (ready to paste in prompt)
    tokens_used: int               # Actual tokens consumed (≤ budget)
    channel_used: ChannelLevel     # Which channel served this (TEXT, EMBEDDING, TENSOR)
    query_embedding: Optional[np.ndarray]  # The query's embedding (if computed)
```

**`result.text` format:**
```
[task/auth/status] (agent-backend, conf=0.92, fact): JWT signing key expired. Auth returning 401.
[task/decision/rollback] (agent-planner, conf=0.97, decision): Rolling back to v2.1.3...
```

**Scoring (default weights):**
```
score = 0.30 × recency + 0.20 × confidence + 0.30 × similarity + 0.20 × ref_proximity
```
When `query` is not provided, recency weight increases to 0.45 and similarity drops to 0.15.

**Budget packing fallback:**
If an entry doesn't fit at full representation, the system tries compact → path-only → skips.

**Examples:**
```python
# All facts under task/, semantic query, tight budget
result = db.get("task/**", budget=500, query="what is blocking deployment?")
print(result.text)         # Pre-formatted context
print(result.tokens_used)  # Actual tokens (≤500)

# Only decisions, no semantic query (recency-ranked)
result = db.get("task/**", type="decision", budget=1000)

# Specific agent's observations
result = db.get("task/deploy/**", owner="agent-deploy", type="observation")

# Entire knowledge base, large budget
result = db.get("**", budget=10000)
```

---

### `subscribe()` — Watch for Changes

```python
subscription: Subscription = db.subscribe(
    pattern: str,
    subscriber: str,
    callback: Callable[[Event], None],
) -> Subscription
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `pattern` | `str` | Glob pattern. Events at matching paths trigger the callback. |
| `subscriber` | `str` | Identifier for this subscription (for logging). |
| `callback` | `Callable[[Event], None]` | Function called when a matching event fires. Called synchronously on the thread that emitted the event. |

**Returns:** `Subscription` — opaque handle, pass to `unsubscribe()`.

**Event types delivered:**
- `CHANGE` — An entry was written or updated at a matching path
- `CONFLICT` — Two entries at matching paths contradict each other
- `GAP` — A plan at a matching path references a step with no observation
- `STALE` — An entry at a matching path exceeded the staleness threshold
- `SUMMARY` — A matching path was auto-summarized

**`Event` object:**
```python
@dataclass
class Event:
    type: EventType              # CHANGE, CONFLICT, GAP, STALE, SUMMARY
    path: str                    # Path that triggered the event
    entry: Optional[KnowledgeEntry]  # The entry involved (if applicable)
    payload: Dict[str, Any]      # Extra context (e.g., conflicting entries for CONFLICT)
    timestamp: float             # Unix timestamp
```

**Examples:**
```python
def on_conflict(event):
    print(f"CONFLICT at {event.path}")
    print(f"Entry A: {event.payload['entry_a'].value}")
    print(f"Entry B: {event.payload['entry_b'].value}")

sub = db.subscribe("task/**", subscriber="conflict-monitor", callback=on_conflict)

def on_any(event):
    print(f"{event.type.value}: {event.path}")

sub_all = db.subscribe("**", subscriber="audit-logger", callback=on_any)
```

---

### `unsubscribe()` — Stop Watching

```python
db.unsubscribe(subscription: Subscription) -> None
```

Removes the callback. No more events will be delivered for this subscription.

---

## Algebra Methods (Advanced)

### `encode()` — Embed Text

```python
vector: np.ndarray = db.encode(text: str) -> np.ndarray
```

Encode any string into a normalized embedding using the active provider. Useful for custom similarity computations.

---

### `relate()` — Discover Relationship

```python
relationship: Dict[str, float] = db.relate(path_a: str, path_b: str) -> Dict
```

Returns a dictionary of geometric relationships between two stored entries:
```python
{
    "cosine": 0.82,        # Cosine similarity
    "angular_dist": 0.34,  # Angular distance (radians)
    "projection": 0.71,    # How much of A is in B's direction
    "residual": 0.29,      # How much of A is NOT in B's direction
    "parallelism": 0.82,   # Same as cosine for normalized vectors
    "opposition": 0.02,    # Cosine similarity with -B (how opposite?)
    "independence": 0.12,  # Cross-variance (how unrelated?)
}
```

---

### `compose()` — Merge Entries

```python
merged_vector: np.ndarray = db.compose(
    paths: List[str],
    mode: Literal["weighted_mean", "principal", "grassmann", "attentive"] = "attentive"
) -> np.ndarray
```

Synthesizes multiple knowledge entries into a single vector representation.

| Mode | Description | Use Case |
|------|-------------|----------|
| `weighted_mean` | Weighted average by confidence | Fast, good for agreement |
| `principal` | PCA first component | Captures dominant direction |
| `grassmann` | Karcher mean (Riemannian) | Mathematically exact, slow |
| `attentive` | Soft attention weights | Best for conflicting beliefs |

---

### `materialize()` — Vector → Nearest Entries

```python
matches: List[Tuple[str, float]] = db.materialize(
    vector: np.ndarray,
    k: int = 5
) -> List[Tuple[str, float]]
```

Given a vector (e.g., output from `compose()`), find the k nearest entries in the knowledge space. Returns `(path, similarity)` tuples.

---

## Introspection Methods

### `retrieve()` — Get Single Entry

```python
entry: Optional[KnowledgeEntry] = db.retrieve(path: str)
```

Exact path lookup. Returns `None` if not found.

---

### `delete()` — Remove Entry

```python
deleted: bool = db.delete(path: str) -> bool
```

Removes an entry. Returns `True` if found and deleted, `False` if not found. Fires `DELETE` event.

---

### `exists()` — Check Path

```python
found: bool = db.exists(path: str) -> bool
```

---

### `status()` — System Status

```python
status: Dict = db.status()
# Returns:
{
    "entries": 47,
    "channel": "EMBEDDING",
    "embeddings": True,
    "embedding_provider": "local",
    "embedding_dimension": 384,
    "vector_entries": 47,
    "semantic_index_ready": True,
    "reindexed_on_startup": False,
    "encryption_enabled": False,
    "subscriptions": 3,
    "reasoning": True,
    "total_puts": 128,
    "total_gets": 64,
    "provenance_nodes": 52,
}
```

---

### `agent_trust_report()` — Trust Scores

```python
report: Dict = db.agent_trust_report()
# Returns:
{
    "agents": {
        "agent-backend": {
            "global_trust": 0.73,
            "domain_trust": {"auth": 0.89, "deploy": 0.65},
            "total_entries": 12,
            "confirmation_rate": 0.83,
            "is_reliable": True
        },
        ...
    },
    "total_agents": 4,
    "most_trusted": "agent-backend",
}
```

---

### `lineage()` — Provenance Chain

```python
chain: List[ProvenanceNode] = db.lineage(path: str) -> List[ProvenanceNode]
```

Returns the full derivation history of an entry — who wrote it, what confirmed or contradicted it, what it was composed from.

---

### `knowledge_topology()` — Graph Topology

```python
topology: Dict = db.knowledge_topology()
```

Returns the structure of the knowledge graph — clusters, centrality, coverage, density.

---

### `tool_definitions()` — LLM Tool Schemas

```python
tools: List[Dict] = DimensionalBase.tool_definitions()
```

Returns JSON tool definitions for `put`, `get`, and `subscribe` in the format expected by OpenAI/Anthropic function calling. Use this to include DimensionalBase tools in a model's tool list.

---

### `close()` — Cleanup

```python
db.close()
```

Commits pending writes, closes SQLite connection, releases resources. Call this when done.

---

## Types Reference

### `KnowledgeEntry`

```python
@dataclass
class KnowledgeEntry:
    path: str
    value: str
    owner: str
    type: EntryType = EntryType.FACT
    confidence: float = 1.0
    refs: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    version: int = 1
    ttl: TTL = TTL.SESSION
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, str] = field(default_factory=dict)
```

---

### `QueryResult`

```python
@dataclass
class QueryResult:
    entries: List[KnowledgeEntry]
    text: str
    tokens_used: int
    channel_used: ChannelLevel
    query_embedding: Optional[np.ndarray] = None
```

---

### `ScoringWeights`

```python
@dataclass
class ScoringWeights:
    recency: float = 0.30
    confidence: float = 0.20
    similarity: float = 0.30
    ref_distance: float = 0.20
    # Must sum to 1.0
```

---

### `Subscription`

Opaque handle returned by `subscribe()`. Pass it to `unsubscribe()`.

---

## Exception Hierarchy

```
DimensionalBaseError (base)
├── EntryValidationError   — Invalid path, value, confidence, etc.
├── StorageError           — SQLite write/read failure
├── ChannelError           — Channel unavailable or misconfigured
├── EmbeddingError         — Embedding provider failure (network, model not loaded)
├── SchemaVersionError     — SQLite schema migration needed
└── BudgetExhaustedError   — Could not fit even one entry within the budget
```

All exceptions include a `message` and an optional `context` dict with details.

---

## Path Syntax

Paths use `/` as a separator. They are hierarchical — `task/auth/status` is a child of `task/auth`, which is a child of `task`.

**Rules:**
- No leading slash: `"task/auth"` ✓, `"/task/auth"` ✗
- No trailing slash: `"task/auth"` ✓, `"task/auth/"` ✗
- Lowercase alphanumeric + hyphens recommended: `"task/api-v2/status"` ✓
- Underscores are allowed: `"task/api_v2"` ✓

**Glob patterns (for `scope` and `subscribe`):**
| Pattern | Matches |
|---------|---------|
| `task/**` | Any path under `task/`, including `task/a`, `task/a/b`, `task/a/b/c` |
| `task/*/status` | `task/auth/status`, `task/db/status` (one segment wildcard) |
| `task/?uth/status` | `task/auth/status` (single character wildcard) |
| `**` | Everything |
| `task/auth/status` | Exact match only |

---

## Complete Example: Multi-Agent Coordination

```python
from dimensionalbase import DimensionalBase

db = DimensionalBase(db_path="session.db")

# --- Agent Backend writes a fact ---
db.put(
    "task/auth/status",
    "JWT signing key expired. Auth returning 401 for all users.",
    owner="agent-backend",
    type="fact",
    confidence=0.95,
    ttl="session",
)

# --- Agent Monitor detects conflicts automatically ---
# (ActiveReasoning fires CONFLICT event if another agent writes a contradictory fact)
conflict_log = []
def on_conflict(event):
    conflict_log.append(event)

sub = db.subscribe("task/**", subscriber="monitor", callback=on_conflict)

# --- Agent Planner reads context and makes a decision ---
context = db.get(
    scope="task/**",
    budget=800,
    query="what auth problems exist and what should we do?",
)

print(context.text)
# [task/auth/status] (agent-backend, conf=0.95, fact): JWT signing key expired...

db.put(
    "task/decision/fix-auth",
    "Rotate JWT signing key using secrets manager. ETA: 5 minutes.",
    owner="agent-planner",
    type="decision",
    confidence=0.90,
    refs=["task/auth/status"],  # Links to the fact that motivated this decision
    ttl="persistent",
)

# --- Agent Deploy reads only what it needs ---
deploy_context = db.get(
    scope="task/decision/**",
    budget=300,
    query="what do I need to deploy?",
)

print(deploy_context.tokens_used)  # Far less than 300 — only relevant entries
print(deploy_context.channel_used) # ChannelLevel.EMBEDDING

# --- Check agent trust ---
trust = db.agent_trust_report()
print(trust["agents"]["agent-planner"]["global_trust"])

# --- Cleanup ---
db.unsubscribe(sub)
db.close()
```
