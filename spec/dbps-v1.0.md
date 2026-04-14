# DimensionalBase Protocol Specification (DBPS) Version 1.0.0

**Status:** Draft Standard
**Date:** 2026-04-10
**Authors:** DimensionalBase Core Team
**License:** Apache 2.0
**RFC 2119 Compliance:** The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in RFC 2119.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Terminology](#2-terminology)
3. [Architecture Overview](#3-architecture-overview)
4. [Entry Schema](#4-entry-schema)
5. [Dimensional Model](#5-dimensional-model)
6. [Storage Layer](#6-storage-layer)
7. [Channel Negotiation](#7-channel-negotiation)
8. [Ingestion Pipeline](#8-ingestion-pipeline)
9. [Embedding Generation](#9-embedding-generation)
10. [Similarity Search](#10-similarity-search)
11. [Confidence Scoring](#11-confidence-scoring)
12. [Trust Model](#12-trust-model)
13. [Conflict Resolution](#13-conflict-resolution)
14. [Temporal Reasoning](#14-temporal-reasoning)
15. [Composite Scoring](#15-composite-scoring)
16. [REST API](#16-rest-api)
17. [WebSocket Protocol](#17-websocket-protocol)
18. [Model Context Protocol (MCP) Integration](#18-model-context-protocol-mcp-integration)
19. [Security](#19-security)
20. [Error Handling](#20-error-handling)
21. [Versioning and Migration](#21-versioning-and-migration)
22. [Conformance and Test Vectors](#22-conformance-and-test-vectors)

---

## 1. Introduction

### 1.1 Purpose

This document defines the DimensionalBase Protocol Specification (DBPS), a formal protocol
for structured knowledge management that combines semantic embeddings, confidence tracking,
trust propagation, and temporal reasoning into a unified memory substrate for AI systems.

### 1.2 Scope

DBPS specifies the wire formats, algorithmic requirements, storage semantics, and API
contracts that compliant implementations MUST support. The protocol is designed to be
transport-agnostic, though reference bindings are provided for HTTP/REST, WebSocket,
and the Model Context Protocol (MCP).

### 1.3 Design Goals

1. **Dimensional Fidelity** — Every knowledge entry MUST be representable across multiple
   orthogonal dimensions (semantic, temporal, confidence, trust, relational).
2. **Epistemic Honesty** — The system MUST track and expose uncertainty rather than
   presenting all knowledge with equal authority.
3. **Source Accountability** — Every assertion MUST be traceable to its originating source,
   and sources MUST carry quantified trust scores.
4. **Temporal Awareness** — The system MUST distinguish between "known at time T" and
   "true at time T", supporting both event-time and ingestion-time semantics.
5. **Interoperability** — Implementations MUST expose functionality through standardized
   protocols (REST, WebSocket, MCP) with well-defined schemas.

### 1.4 Notation Conventions

Mathematical formulas in this specification use standard notation. Vectors are denoted
in boldface (e.g., **v**). Scalars are italicized (e.g., *α*). Set membership is denoted
with ∈. The operator ‖·‖ denotes the L2 (Euclidean) norm unless otherwise specified.

### 1.5 Normative References

- RFC 2119 — Key words for use in RFCs to Indicate Requirement Levels
- RFC 7159 — The JavaScript Object Notation (JSON) Data Interchange Format
- RFC 6455 — The WebSocket Protocol
- RFC 7519 — JSON Web Token (JWT)
- RFC 8259 — The JavaScript Object Notation (JSON) Data Interchange Format (supersedes 7159)

---

## 2. Terminology

### 2.1 Core Definitions

- **Entry**: The atomic unit of knowledge in DimensionalBase. Each entry represents a single
  assertion, fact, observation, or relationship. Entries are immutable once committed;
  updates produce new entry versions linked to their predecessors.

- **Dimension**: An orthogonal axis along which entries can be measured, compared, or
  filtered. The core dimensions are: semantic, temporal, confidence, trust, and relational.

- **Channel**: A namespaced communication pathway between a client and the DimensionalBase
  server. Channels carry typed messages and maintain session state.

- **Source**: An identified origin of knowledge. Sources have associated trust scores that
  propagate to entries they produce. Sources MAY be human users, AI agents, APIs, documents,
  or sensor feeds.

- **Confidence Score**: A probability measure in the range [0.0, 1.0] representing the
  system's belief in the truth of an entry. Confidence is modeled as a Beta distribution
  and decays over time.

- **Trust Score**: A composite measure in the range [0.0, 1.0] representing the reliability
  of a source. Trust is computed via Elo rating and PageRank propagation.

- **Embedding**: A dense vector representation of an entry's semantic content. Embeddings
  MUST be L2-normalized and MUST have dimensionality matching the configured model.

- **Composite Score**: The final ranking score for an entry, computed as a weighted
  combination of relevance, confidence, trust, and recency.

### 2.2 Dimensional Axes

| Axis       | Type       | Range         | Unit               |
|------------|------------|---------------|---------------------|
| Semantic   | Vector     | [-1.0, 1.0]^d | Cosine similarity  |
| Temporal   | Scalar     | [0, ∞)        | Unix epoch seconds  |
| Confidence | Distribution | [0.0, 1.0]  | Probability         |
| Trust      | Scalar     | [0.0, 1.0]   | Dimensionless       |
| Relational | Graph      | N/A           | Edge weight [0,1]   |

### 2.3 Abbreviations

| Abbreviation | Expansion                              |
|-------------|----------------------------------------|
| DBPS        | DimensionalBase Protocol Specification  |
| MCP         | Model Context Protocol                  |
| JWT         | JSON Web Token                          |
| ELO         | Elo rating system                       |
| PR          | PageRank                                |
| HNSW        | Hierarchical Navigable Small World      |
| TTL         | Time To Live                            |
| UUID        | Universally Unique Identifier           |

---

## 3. Architecture Overview

### 3.1 System Layers

A conforming DBPS implementation MUST provide the following layers:

```
┌─────────────────────────────────────────────────┐
│              Client Layer (MCP / REST / WS)      │
├─────────────────────────────────────────────────┤
│              Protocol Layer (DBPS v1.0)          │
├─────────────────────────────────────────────────┤
│              Processing Layer                    │
│  ┌──────────┬──────────┬──────────┬──────────┐  │
│  │ Ingestion│ Embedding│ Scoring  │ Conflict │  │
│  │ Pipeline │ Generator│ Engine   │ Resolver │  │
│  └──────────┴──────────┴──────────┴──────────┘  │
├─────────────────────────────────────────────────┤
│              Storage Layer                       │
│  ┌──────────┬──────────┬──────────┐             │
│  │ Vector   │ Document │ Graph    │             │
│  │ Index    │ Store    │ Store    │             │
│  └──────────┴──────────┴──────────┘             │
└─────────────────────────────────────────────────┘
```

### 3.2 Data Flow

1. Knowledge enters the system through one of the client interfaces (MCP tool call,
   REST API request, or WebSocket message).
2. The Protocol Layer validates the incoming message against the DBPS schema and
   authenticates the request.
3. The Ingestion Pipeline normalizes, deduplicates, and enriches the entry.
4. The Embedding Generator produces a semantic vector for the entry's content.
5. The Scoring Engine computes initial confidence (from source trust and content analysis)
   and composite scores.
6. The Conflict Resolver checks for contradictions with existing entries and triggers
   resolution workflows if needed.
7. The entry is persisted across all three storage backends (vector index, document store,
   graph store) atomically.

### 3.3 Consistency Model

DBPS employs an eventual consistency model with the following guarantees:

- **Read-after-write consistency** — A client that writes an entry MUST be able to read
  it back immediately within the same channel session.
- **Monotonic reads** — Once a client reads a version of an entry, subsequent reads
  MUST NOT return older versions.
- **Causal consistency** — If entry B was created with knowledge of entry A, then any
  client that sees B MUST also be able to see A.

### 3.4 Concurrency

Implementations MUST support concurrent access from multiple channels. Write operations
on the same entry MUST be serialized. Implementations SHOULD use optimistic concurrency
control with version vectors. When a write conflict is detected, the implementation
MUST reject the later write with error code `CONFLICT_VERSION_MISMATCH` (see Section 20).

### 3.5 Persistence Guarantees

All committed entries MUST survive process restart. Implementations MUST flush writes
to durable storage before acknowledging success to the client. Write-ahead logging (WAL)
is RECOMMENDED for crash recovery.

---

## 4. Entry Schema

### 4.1 Overview

An Entry is the fundamental unit of knowledge in DBPS. Every entry MUST conform to the
schema defined in this section. Implementations MUST reject entries that violate any
constraint specified herein.

### 4.2 Field Definitions

An Entry consists of exactly 13 fields. All fields are REQUIRED unless explicitly marked
OPTIONAL.

#### Field 1: `id`

- **Type:** `string` (UUID v4)
- **Format:** RFC 4122 UUID, lowercase hexadecimal with hyphens
- **Constraints:** MUST be globally unique. MUST be generated by the server upon ingestion.
  Clients MUST NOT set this field.
- **Example:** `"550e8400-e29b-41d4-a716-446655440000"`

#### Field 2: `content`

- **Type:** `string`
- **Format:** UTF-8 encoded text
- **Constraints:** MUST NOT be empty. MUST NOT exceed 65,536 bytes when encoded as UTF-8.
  Leading and trailing whitespace SHOULD be trimmed by the ingestion pipeline.
- **Example:** `"The speed of light in a vacuum is approximately 299,792,458 meters per second."`

#### Field 3: `source`

- **Type:** `string`
- **Format:** URI or structured identifier
- **Constraints:** MUST NOT be empty. MUST identify the origin of the knowledge. For human
  sources, the format SHOULD be `user:<username>`. For AI sources, `agent:<agent_id>`.
  For document sources, `doc:<document_uri>`. For API sources, `api:<endpoint>`.
- **Example:** `"doc:https://physics.nist.gov/cgi-bin/cuu/Value?c"`

#### Field 4: `confidence`

- **Type:** `number` (float64)
- **Format:** Decimal in range [0.0, 1.0]
- **Constraints:** MUST be a finite number. MUST NOT be NaN or Infinity. Initial value
  is set during ingestion based on source trust and content analysis. Subsequent updates
  follow the Beta distribution model defined in Section 11.
- **Default:** Computed as `source_trust * content_quality_factor`
- **Example:** `0.87`

#### Field 5: `confidence_distribution`

- **Type:** `object`
- **Format:** `{ "alpha": number, "beta": number }`
- **Constraints:** Both `alpha` and `beta` MUST be positive real numbers (> 0). The mean
  of the distribution (`alpha / (alpha + beta)`) MUST equal the `confidence` field value
  within a tolerance of ±0.001.
- **Default:** Computed from initial confidence *c* as `{ "alpha": 1 + c, "beta": 1 + (1 - c) }`
- **Example:** `{ "alpha": 1.87, "beta": 1.13 }`

#### Field 6: `embedding`

- **Type:** `array<number>` (float32[])
- **Format:** Dense vector of floating-point values
- **Constraints:** Length MUST match the configured embedding dimensionality (default: 1536
  for text-embedding-3-small, 3072 for text-embedding-3-large). Each component MUST be a
  finite float32 value. The vector MUST be L2-normalized such that ‖**v**‖₂ = 1.0 ± 0.0001.
- **Example:** `[0.0123, -0.0456, 0.0789, ...]` (truncated)

#### Field 7: `dimensions`

- **Type:** `object`
- **Format:** Key-value map of dimension names to dimension values
- **Constraints:** MUST contain at least the keys `"semantic"`, `"temporal"`, and
  `"confidence"`. Additional custom dimensions MAY be included. Keys MUST be lowercase
  alphanumeric strings with optional underscores, not exceeding 64 characters. Values
  MUST be JSON-serializable.
- **Example:** `{ "semantic": "physics.constants", "temporal": "2024-01-15T00:00:00Z", "confidence": 0.87, "domain": "science" }`

#### Field 8: `created_at`

- **Type:** `string` (ISO 8601 datetime)
- **Format:** `YYYY-MM-DDTHH:mm:ss.sssZ` (UTC)
- **Constraints:** MUST be set by the server at ingestion time. Clients MUST NOT set this
  field. MUST represent the wall-clock time at which the entry was committed to storage.
- **Example:** `"2025-06-15T14:30:00.000Z"`

#### Field 9: `updated_at`

- **Type:** `string` (ISO 8601 datetime)
- **Format:** `YYYY-MM-DDTHH:mm:ss.sssZ` (UTC)
- **Constraints:** MUST be set by the server. Initially equals `created_at`. Updated
  whenever the entry's confidence, trust, or metadata changes. MUST be monotonically
  non-decreasing.
- **Example:** `"2025-06-15T15:45:00.000Z"`

#### Field 10: `event_time`

- **Type:** `string` (ISO 8601 datetime) — OPTIONAL
- **Format:** `YYYY-MM-DDTHH:mm:ss.sssZ` (UTC)
- **Constraints:** Represents the time at which the knowledge was true or the event
  occurred, as opposed to when it was ingested. MAY be in the past or future relative
  to `created_at`. If omitted, temporal queries SHOULD fall back to `created_at`.
- **Example:** `"1905-06-30T00:00:00Z"` (Einstein's special relativity paper)

#### Field 11: `tags`

- **Type:** `array<string>`
- **Format:** List of lowercase alphanumeric strings with optional hyphens and underscores
- **Constraints:** Each tag MUST NOT exceed 128 characters. The array MUST NOT contain
  more than 64 tags. Duplicate tags MUST be silently deduplicated. Tags MUST be stored
  in sorted lexicographic order.
- **Example:** `["physics", "constants", "speed-of-light", "nist"]`

#### Field 12: `relations`

- **Type:** `array<object>` — OPTIONAL
- **Format:** List of relation objects: `{ "type": string, "target_id": string, "weight": number }`
- **Constraints:** `type` MUST be one of: `"supports"`, `"contradicts"`, `"extends"`,
  `"supersedes"`, `"related_to"`, `"derived_from"`, `"part_of"`. `target_id` MUST be a
  valid entry UUID that exists in the store (referential integrity). `weight` MUST be in
  range [0.0, 1.0] and defaults to 1.0 if omitted.
- **Example:** `[{ "type": "supports", "target_id": "661f9511-f30c-42e5-b817-557766550011", "weight": 0.95 }]`

#### Field 13: `metadata`

- **Type:** `object` — OPTIONAL
- **Format:** Arbitrary JSON object for implementation-specific extensions
- **Constraints:** MUST NOT exceed 16,384 bytes when serialized as JSON. Keys MUST NOT
  begin with the prefix `_dbps_` which is reserved for protocol-level metadata. Nested
  depth MUST NOT exceed 8 levels.
- **Example:** `{ "language": "en", "extraction_method": "manual", "review_status": "verified" }`

### 4.3 Schema Validation

Implementations MUST validate all 13 fields upon ingestion. Validation failures MUST
result in a `SCHEMA_VALIDATION_ERROR` (error code 4001) with a human-readable description
of which field(s) failed and why.

### 4.4 Serialization

Entries MUST be serialized as JSON (RFC 8259) for transport. The embedding field MAY be
omitted from responses when the client sets the `exclude_embeddings=true` query parameter,
to reduce payload size. When omitted, the field MUST be set to `null` (not absent).

### 4.5 Size Limits

| Field                  | Maximum Size         |
|------------------------|----------------------|
| `content`              | 65,536 bytes (UTF-8) |
| `embedding`            | 12,288 float32 values|
| `tags`                 | 64 elements          |
| `relations`            | 1,024 elements       |
| `metadata`             | 16,384 bytes (JSON)  |
| Total entry (serialized) | 262,144 bytes      |

---

## 5. Dimensional Model

### 5.1 Overview

The Dimensional Model is the mathematical foundation of DBPS. Each entry exists as a point
in a multi-dimensional space where each axis represents an independently measurable property.
Queries project into this space and retrieve entries based on proximity along one or more axes.

### 5.2 Semantic Dimension

The semantic dimension is defined by the entry's embedding vector. Similarity between two
entries *A* and *B* along this dimension is computed as cosine similarity:

```
sim_semantic(A, B) = (A · B) / (‖A‖₂ × ‖B‖₂)
```

Since all embeddings are L2-normalized (per Section 4, Field 6), this simplifies to:

```
sim_semantic(A, B) = A · B
```

The result is in the range [-1.0, 1.0]. Values closer to 1.0 indicate higher semantic
similarity. Implementations MUST use this formula for all semantic comparisons.

### 5.3 Temporal Dimension

The temporal dimension maps entries onto a continuous timeline. Temporal proximity between
entries is computed using an exponential decay function:

```
sim_temporal(A, B) = exp(-|t_A - t_B| / τ)
```

Where:
- `t_A` and `t_B` are the timestamps of entries A and B (in Unix epoch seconds)
- `τ` (tau) is the decay constant, default value 86400 (24 hours)

Implementations MUST support configurable τ values in the range [60, 31536000] (1 minute
to 1 year). The temporal dimension uses `event_time` when available, falling back to
`created_at`.

### 5.4 Confidence Dimension

The confidence dimension represents epistemic certainty. Entries with higher confidence
scores are considered more reliable. Confidence proximity between entries is computed as:

```
sim_confidence(A, B) = 1 - |c_A - c_B|
```

Where `c_A` and `c_B` are the confidence scores of entries A and B. This yields a value
in [0.0, 1.0]. The full confidence model is defined in Section 11.

### 5.5 Trust Dimension

The trust dimension propagates from sources to entries. Trust proximity is defined as:

```
sim_trust(A, B) = 1 - |trust(source_A) - trust(source_B)|
```

The full trust model is defined in Section 12.

### 5.6 Relational Dimension

The relational dimension captures graph structure. Relational proximity between entries
is defined by their shortest-path distance in the knowledge graph, weighted by edge weights:

```
sim_relational(A, B) = 1 / (1 + d_weighted(A, B))
```

Where `d_weighted(A, B)` is the shortest weighted path distance between A and B. If no
path exists, `sim_relational(A, B) = 0`.

### 5.7 Custom Dimensions

Implementations MAY support custom dimensions beyond the five core dimensions. Custom
dimensions MUST be registered through the configuration API before use. Each custom
dimension MUST define:
- A name (unique, lowercase alphanumeric with underscores)
- A value type (`scalar`, `vector`, `categorical`)
- A similarity function
- A default value

### 5.8 Dimensional Projection

Queries specify which dimensions to include and their relative weights. The combined
multi-dimensional similarity is:

```
sim_combined(A, Q) = Σᵢ wᵢ × sim_dimensionᵢ(A, Q) / Σᵢ wᵢ
```

Where `wᵢ` are the dimension weights specified in the query. Dimensions with weight 0
MUST be excluded from computation entirely (not merely zeroed).

---

## 6. Storage Layer

### 6.1 Overview

DBPS requires three complementary storage backends that MUST maintain consistency:

1. **Vector Index** — For efficient approximate nearest-neighbor (ANN) search over embeddings
2. **Document Store** — For full entry persistence and field-level queries
3. **Graph Store** — For relational traversal and trust propagation

### 6.2 Vector Index Requirements

The vector index MUST support:
- L2-normalized vectors of configurable dimensionality (768, 1024, 1536, 3072)
- Approximate nearest-neighbor search with configurable recall targets
- HNSW (Hierarchical Navigable Small World) indexing is RECOMMENDED
- Minimum throughput: 100 queries per second at p99 < 50ms for 1M vectors
- Index parameters:
  - `M` (max connections per layer): default 16, range [4, 64]
  - `ef_construction`: default 200, range [100, 500]
  - `ef_search`: default 100, range [10, 500]

### 6.3 Document Store Requirements

The document store MUST support:
- CRUD operations on full entry documents
- Secondary indexes on: `source`, `created_at`, `updated_at`, `tags`, `confidence`
- Range queries on numeric and datetime fields
- Full-text search on `content` field (RECOMMENDED)
- Atomic read-modify-write operations for confidence and trust updates
- Bulk operations (batch insert, batch update) for ingestion throughput

### 6.4 Graph Store Requirements

The graph store MUST support:
- Directed, weighted edges between entries (as defined by the `relations` field)
- Breadth-first and depth-first traversal
- Shortest-path computation (Dijkstra's algorithm with edge weights)
- PageRank computation (see Section 12)
- Minimum traversal throughput: 10,000 edges per second

### 6.5 Consistency Between Stores

All three stores MUST be updated atomically for each write operation. Implementations
MUST use one of the following strategies:
- Two-phase commit (2PC) across all three stores
- Write-ahead log (WAL) with replay on failure
- Single-store-of-record with synchronous replication

If any store fails during a write, the entire operation MUST be rolled back, and the
client MUST receive an error response. Partial writes are NOT permitted.

### 6.6 Compaction and Maintenance

Implementations SHOULD periodically compact storage to reclaim space from soft-deleted
entries. Compaction MUST NOT affect the availability of non-deleted entries. The vector
index SHOULD be rebuilt periodically to maintain query performance as the dataset grows.

### 6.7 Backup and Recovery

Implementations MUST support point-in-time recovery. Backups MUST capture a consistent
snapshot across all three stores. The maximum acceptable data loss (RPO) MUST be
configurable, with a default of 0 (synchronous durability).

---

## 7. Channel Negotiation

### 7.1 Overview

Channel negotiation establishes a typed, authenticated communication session between a
client and the DBPS server. Channels carry all subsequent protocol messages and maintain
session state including authentication context, active subscriptions, and query cursors.

### 7.2 Handshake Initiation

A client initiates channel negotiation by sending a `channel.open` message. This message
MUST be the first message on any new connection. The handshake uses JSON encoding over
the underlying transport (HTTP for REST, WebSocket frame for WS, tool call for MCP).

#### Request Format

```json
{
  "jsonrpc": "2.0",
  "method": "channel.open",
  "id": 1,
  "params": {
    "protocol_version": "1.0.0",
    "client_id": "client-uuid-here",
    "client_name": "MyApp/2.1.0",
    "capabilities": {
      "streaming": true,
      "batch_operations": true,
      "subscriptions": true,
      "max_embedding_dimensions": 3072,
      "supported_similarity_functions": ["cosine", "dot_product", "euclidean"],
      "compression": ["gzip", "none"]
    },
    "auth": {
      "type": "bearer",
      "token": "eyJhbGciOiJIUzI1NiIs..."
    }
  }
}
```

### 7.3 Handshake Response

The server MUST respond with a `channel.opened` result or an error within 5 seconds.

#### Success Response Format

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "channel_id": "ch-550e8400-e29b-41d4-a716-446655440000",
    "protocol_version": "1.0.0",
    "server_name": "DimensionalBase/1.0.0",
    "negotiated_capabilities": {
      "streaming": true,
      "batch_operations": true,
      "subscriptions": true,
      "embedding_dimensions": 1536,
      "similarity_function": "cosine",
      "compression": "gzip",
      "max_batch_size": 1000,
      "max_concurrent_queries": 50,
      "session_ttl_seconds": 3600
    },
    "server_time": "2025-06-15T14:30:00.000Z"
  }
}
```

### 7.4 Capability Negotiation Rules

The following rules govern capability negotiation:

1. **Protocol Version** — The server MUST support the exact version requested or respond
   with error `UNSUPPORTED_VERSION`. Implementations MUST NOT silently downgrade.
2. **Streaming** — If both client and server support streaming, it MUST be enabled. If
   the client requests streaming but the server does not support it, the server MUST set
   `streaming: false` and the client MUST fall back to polling.
3. **Batch Operations** — Negotiated to the minimum of client and server capabilities.
   `max_batch_size` indicates the maximum number of entries per batch operation.
4. **Embedding Dimensions** — The server MUST select the highest dimensionality that does
   not exceed the client's `max_embedding_dimensions` and is supported by the configured
   embedding model.
5. **Compression** — The server MUST select the first mutually supported compression
   algorithm from the client's preference list.

### 7.5 Channel Lifecycle

```
Client                          Server
  |                               |
  |--- channel.open ------------->|
  |                               |-- validate auth
  |                               |-- negotiate capabilities
  |<-- channel.opened ------------|
  |                               |
  |--- (protocol messages) ------>|
  |<-- (protocol messages) -------|
  |                               |
  |--- channel.close ------------>|
  |<-- channel.closed ------------|
  |                               |
```

### 7.6 Keep-Alive

Channels MUST implement keep-alive to detect stale connections. The client MUST send a
`channel.ping` message at least once every `session_ttl_seconds / 3` seconds. The server
MUST respond with `channel.pong` within 5 seconds. If the server does not receive a ping
within `session_ttl_seconds`, it MUST close the channel with reason `SESSION_TIMEOUT`.

```json
{ "jsonrpc": "2.0", "method": "channel.ping", "id": 42, "params": { "timestamp": 1718458200000 } }
{ "jsonrpc": "2.0", "id": 42, "result": { "timestamp": 1718458200015 } }
```

### 7.7 Channel Resumption

If a channel is closed unexpectedly, the client MAY attempt resumption by including the
previous `channel_id` in a new `channel.open` request. The server SHOULD restore session
state (subscriptions, cursors) if the channel was closed less than `session_ttl_seconds`
ago. If resumption fails, the server MUST respond with a fresh channel.

### 7.8 Error During Handshake

If authentication fails, the server MUST respond with error code `AUTH_FAILED` (4010).
If the protocol version is unsupported, error code `UNSUPPORTED_VERSION` (4020).
If the server is overloaded, error code `SERVER_OVERLOADED` (5030) with a `Retry-After`
header or field.

---

## 8. Ingestion Pipeline

### 8.1 Overview

The ingestion pipeline transforms raw knowledge inputs into fully-qualified DBPS entries.
It is a multi-stage pipeline where each stage MUST complete successfully before the next
begins.

### 8.2 Pipeline Stages

```
Input → Validation → Normalization → Deduplication → Enrichment → Embedding → Scoring → Storage
```

### 8.3 Stage 1: Validation

The validation stage MUST verify:
- All REQUIRED fields are present
- All field types match the schema (Section 4)
- All field constraints are satisfied
- The total serialized size does not exceed 262,144 bytes
- The `source` field references a registered source

Validation failures MUST be reported with specific field-level error messages.

### 8.4 Stage 2: Normalization

The normalization stage MUST:
- Trim leading and trailing whitespace from `content`
- Normalize Unicode text to NFC form (Unicode Normalization Form C)
- Convert all tags to lowercase
- Sort tags lexicographically
- Deduplicate tags
- Validate and normalize datetime fields to UTC ISO 8601

### 8.5 Stage 3: Deduplication

The deduplication stage checks for near-duplicate entries:

1. Compute a content fingerprint using SimHash (64-bit)
2. Query existing entries with Hamming distance ≤ 3 from the fingerprint
3. For each candidate, compute exact cosine similarity of embeddings
4. If cosine similarity ≥ 0.95, flag as potential duplicate

When a duplicate is detected:
- If the source is the same: reject with `DUPLICATE_ENTRY` error
- If the source is different: create the entry with a `"derived_from"` relation to the
  existing entry, and boost the existing entry's confidence (see Section 11)

### 8.6 Stage 4: Enrichment

The enrichment stage augments the entry with derived metadata:
- Extract named entities from `content` and store in `metadata.entities`
- Classify the entry's domain/topic and store in `dimensions.semantic`
- Detect language and store in `metadata.language`
- Compute content quality score based on length, specificity, and structure

### 8.7 Stage 5: Embedding

See Section 9 for full embedding generation specification.

### 8.8 Stage 6: Scoring

- Compute initial confidence from source trust and content quality (Section 11)
- Compute initial composite score (Section 15)

### 8.9 Stage 7: Storage

- Persist the fully-qualified entry atomically across all three stores (Section 6)
- Update relevant graph edges if relations are specified
- Emit ingestion event for subscribers

### 8.10 Batch Ingestion

Implementations MUST support batch ingestion of up to `max_batch_size` entries per
request (as negotiated in Section 7). Batch ingestion MUST be atomic: either all entries
succeed or none are committed. Individual entry failures within a batch MUST cause the
entire batch to fail with detailed per-entry error reporting.

### 8.11 Throughput Requirements

Implementations MUST achieve a minimum sustained ingestion throughput of:
- Single entry: p99 latency < 200ms
- Batch of 100 entries: p99 latency < 2000ms
- Batch of 1000 entries: p99 latency < 15000ms

---

## 9. Embedding Generation

### 9.1 Overview

Embedding generation converts the textual content of an entry into a dense vector
representation suitable for semantic similarity search. The quality of embeddings directly
affects the quality of semantic retrieval.

### 9.2 Supported Models

Implementations MUST support at least one of the following embedding models:

| Model                    | Dimensionality | Normalization |
|--------------------------|----------------|---------------|
| text-embedding-3-small   | 1536           | L2            |
| text-embedding-3-large   | 3072           | L2            |
| text-embedding-ada-002   | 1536           | L2            |

Implementations MAY support additional models provided they produce L2-normalized vectors.

### 9.3 Normalization Requirement

All embedding vectors MUST be L2-normalized before storage:

```
v_normalized = v / ‖v‖₂
```

Where `‖v‖₂ = sqrt(Σᵢ vᵢ²)`. After normalization, `‖v_normalized‖₂` MUST equal 1.0
within a tolerance of ±0.0001.

### 9.4 Chunking Strategy

For content that exceeds the embedding model's context window:

1. Split content into chunks of at most 512 tokens with 64-token overlap
2. Generate embeddings for each chunk
3. Compute the entry embedding as the mean of all chunk embeddings
4. Re-normalize the mean embedding to unit length

```
v_entry = normalize(mean(v_chunk_1, v_chunk_2, ..., v_chunk_n))
```

### 9.5 Caching

Implementations SHOULD cache embeddings keyed by `(content_hash, model_id)` to avoid
redundant computation. Cache entries SHOULD be invalidated when the embedding model
changes.

### 9.6 Embedding Updates

If the configured embedding model changes, implementations MUST re-embed all existing
entries. This operation MAY be performed asynchronously. During re-embedding, the old
embeddings MUST remain available for queries. Once re-embedding is complete, the old
embeddings MUST be atomically replaced.

### 9.7 Dimensionality Reduction

Implementations MAY support dimensionality reduction via Matryoshka representation
learning (MRL) for models that support it. When using MRL, the `embedding_dimensions`
negotiated in the channel handshake determines the truncation point. The truncated
vector MUST be re-normalized after truncation.

### 9.8 Error Handling

If embedding generation fails (e.g., model API error), the ingestion pipeline MUST
retry up to 3 times with exponential backoff (initial delay 1s, multiplier 2, max 8s).
If all retries fail, the entry MUST be rejected with error `EMBEDDING_GENERATION_FAILED`.

---

## 10. Similarity Search

### 10.1 Overview

Similarity search is the primary query mechanism in DBPS. It retrieves entries that are
most similar to a query along one or more dimensions. The core algorithm combines
approximate nearest-neighbor (ANN) search in the vector index with filtering and
re-ranking from the document and graph stores.

### 10.2 Query Format

```json
{
  "method": "entries.search",
  "params": {
    "query": "What is the speed of light?",
    "query_embedding": [0.0123, -0.0456, ...],
    "dimensions": {
      "semantic": { "weight": 0.6 },
      "temporal": { "weight": 0.1, "reference_time": "2025-06-15T00:00:00Z", "decay_tau": 86400 },
      "confidence": { "weight": 0.2, "min_threshold": 0.5 },
      "trust": { "weight": 0.1, "min_threshold": 0.3 }
    },
    "filters": {
      "tags": { "any_of": ["physics", "constants"] },
      "source": { "one_of": ["doc:nist.gov", "user:physicist"] },
      "created_after": "2024-01-01T00:00:00Z"
    },
    "limit": 10,
    "offset": 0,
    "include_embeddings": false,
    "include_explanations": true
  }
}
```

### 10.3 Search Algorithm

1. **Pre-filter:** Apply all non-semantic filters to narrow the candidate set. If the
   filter selectivity is < 1% of total entries, apply filters before ANN search.
   Otherwise, apply filters after ANN search (post-filtering).

2. **ANN Search:** Query the vector index for the top `limit * oversampling_factor`
   nearest neighbors to `query_embedding`. The oversampling factor MUST be at least 3
   to ensure sufficient candidates survive filtering. Implementations MAY increase the
   oversampling factor when filter selectivity is low.

3. **Re-rank:** Compute the composite score (Section 15) for each candidate and sort
   by descending composite score.

4. **Paginate:** Apply `offset` and `limit` to the re-ranked results.

### 10.4 Explanation Format

When `include_explanations` is true, each result MUST include a breakdown:

```json
{
  "entry": { ... },
  "score": 0.847,
  "explanation": {
    "semantic_similarity": 0.92,
    "temporal_proximity": 0.85,
    "confidence": 0.87,
    "trust": 0.91,
    "weights": { "semantic": 0.6, "temporal": 0.1, "confidence": 0.2, "trust": 0.1 },
    "weighted_components": {
      "semantic": 0.552,
      "temporal": 0.085,
      "confidence": 0.174,
      "trust": 0.091
    },
    "composite_score": 0.847
  }
}
```

### 10.5 Performance Requirements

| Dataset Size | p50 Latency | p99 Latency | Minimum Recall@10 |
|-------------|-------------|-------------|-------------------|
| 10,000      | < 5ms       | < 20ms      | 0.95              |
| 100,000     | < 10ms      | < 50ms      | 0.92              |
| 1,000,000   | < 25ms      | < 100ms     | 0.90              |
| 10,000,000  | < 50ms      | < 200ms     | 0.85              |

### 10.6 Empty Results

If no entries match the query after filtering, the response MUST contain an empty `results`
array with a `total_count` of 0. The response MUST NOT be an error.

### 10.7 Hybrid Search

Implementations SHOULD support hybrid search that combines semantic (vector) search with
keyword (BM25) search. When hybrid mode is enabled, the final score is:

```
score_hybrid = α × score_semantic + (1 - α) × score_keyword
```

Where α is configurable, default 0.7. The keyword score MUST be normalized to [0, 1].

---

## 11. Confidence Scoring

### 11.1 Overview

Confidence scoring quantifies the system's belief in the truth of an entry. Unlike static
scores, DBPS confidence is a living value modeled as a Beta distribution that evolves over
time through confirmation, contradiction, and temporal decay.

### 11.2 Beta Distribution Model

Each entry's confidence is represented by a Beta distribution Beta(α, β), where:

- **α** (alpha) represents the strength of evidence supporting the entry
- **β** (beta) represents the strength of evidence against the entry
- The **mean** of the distribution is the entry's confidence score: `c = α / (α + β)`
- The **variance** indicates uncertainty: `var = (α × β) / ((α + β)² × (α + β + 1))`

### 11.3 Initialization

When an entry is first ingested with initial confidence *c* (derived from source trust
and content analysis), the Beta distribution parameters MUST be initialized as:

```
α = 1 + c
β = 1 + (1 - c)
```

This gives a weakly informative prior centered on *c*. For example:
- c = 0.5 → Beta(1.5, 1.5), mean = 0.5, variance = 0.0625
- c = 0.8 → Beta(1.8, 1.2), mean = 0.6, variance = 0.06
- c = 0.9 → Beta(1.9, 1.1), mean = 0.633, variance = 0.058

The initial α + β = 2 represents low total evidence, ensuring the distribution is
responsive to new evidence.

### 11.4 Evidence Updates

#### 11.4.1 Confirmation

When evidence confirms an entry (e.g., another source asserts the same fact, or the entry
is explicitly verified), the distribution is updated:

```
α_new = α_old + 1
```

β remains unchanged. The confidence increases, and the variance decreases (higher certainty).

Multiple independent confirmations are additive:

```
α_new = α_old + n_confirmations
```

#### 11.4.2 Contradiction

When evidence contradicts an entry (e.g., a conflicting assertion from a trusted source),
the distribution is updated:

```
β_new = β_old + 1
```

α remains unchanged. The confidence decreases, and the distribution shifts toward
uncertainty.

Multiple independent contradictions are additive:

```
β_new = β_old + n_contradictions
```

#### 11.4.3 Weighted Evidence

When the confirming or contradicting source has a known trust score *t*, the update
SHOULD be weighted:

```
Confirmation: α_new = α_old + t
Contradiction: β_new = β_old + t
```

This ensures that evidence from more trusted sources has a proportionally larger impact.

### 11.5 Temporal Decay

Confidence decays over time to reflect the diminishing reliability of stale knowledge.
Decay follows an exponential model with a configurable half-life:

```
c_decayed(t) = c_base × 2^(-(t - t_last_update) / t_half)
```

Where:
- `c_base` = the confidence at the time of last update (`α / (α + β)`)
- `t` = current time (Unix epoch seconds)
- `t_last_update` = time of the last evidence update (Unix epoch seconds)
- `t_half` = decay half-life in seconds

The default half-life MUST be **7200 seconds** (2 hours). Implementations MUST support
configurable half-life values in the range [60, 31536000] (1 minute to 1 year).

### 11.6 Decay Application

Temporal decay is applied lazily at query time, NOT eagerly on a schedule. This is
critical for performance:

1. When an entry is retrieved, compute `t_elapsed = now() - t_last_update`
2. Apply decay: `c_effective = (α / (α + β)) × 2^(-t_elapsed / t_half)`
3. Return `c_effective` as the entry's confidence score
4. Implementations MAY periodically materialize decayed confidence scores for entries
   that have not been accessed recently (background compaction)

### 11.7 Confidence Floor

The confidence score MUST NOT decay below a configurable floor value. The default
floor MUST be **0.01**. This ensures that entries never reach exactly zero confidence,
preserving the ability to recover if new confirming evidence arrives.

```
c_final = max(c_effective, c_floor)
```

### 11.8 Confidence Aggregation

When multiple entries assert the same fact (detected via deduplication), their confidence
distributions MAY be aggregated:

```
α_aggregated = Σᵢ (αᵢ - 1) + 1
β_aggregated = Σᵢ (βᵢ - 1) + 1
```

This combines evidence from all sources while maintaining proper Beta distribution
semantics.

### 11.9 Confidence Thresholds

Implementations MUST support filtering entries by confidence thresholds:

| Level       | Range           | Semantic Meaning                     |
|-------------|-----------------|--------------------------------------|
| Verified    | [0.90, 1.00]   | Highly reliable, multi-source confirmed |
| Confident   | [0.70, 0.90)   | Reliable, single trusted source      |
| Probable    | [0.50, 0.70)   | More likely true than false          |
| Uncertain   | [0.30, 0.50)   | Insufficient evidence                |
| Doubtful    | [0.10, 0.30)   | More likely false than true          |
| Discredited | [0.00, 0.10)   | Contradicted by strong evidence      |

---

## 12. Trust Model

### 12.1 Overview

The Trust Model quantifies the reliability of knowledge sources. Trust is a composite
score derived from two independent signals: Elo rating (pairwise accuracy) and PageRank
(network authority). These signals are blended per-domain to produce a final trust score.

### 12.2 Elo Rating System

Each source maintains an Elo rating that reflects its historical accuracy relative to
other sources. The Elo system is adapted from chess rating with the following parameters:

- **Initial Rating:** 1500 (for all new sources)
- **K-factor:** 32 (update magnitude)
- **Scale Factor:** 400 (performance spread)

#### 12.2.1 Expected Score

The expected score of source A against source B is:

```
E_A = 1 / (1 + 10^((R_B - R_A) / 400))
```

Where `R_A` and `R_B` are the current Elo ratings of sources A and B.

#### 12.2.2 Rating Update

When source A's entry is confirmed as more accurate than source B's conflicting entry:

```
R_A_new = R_A + K × (1 - E_A) = R_A + 32 × (1 - E_A)
R_B_new = R_B + K × (0 - E_B) = R_B + 32 × (0 - E_B)
```

When the outcome is a draw (both entries are valid perspectives):

```
R_A_new = R_A + K × (0.5 - E_A) = R_A + 32 × (0.5 - E_A)
R_B_new = R_B + K × (0.5 - E_B) = R_B + 32 × (0.5 - E_B)
```

#### 12.2.3 Elo to Trust Conversion

The Elo rating is converted to a [0, 1] trust score using a logistic function:

```
trust_elo = 1 / (1 + 10^((1500 - R) / 400))
```

This ensures:
- Rating 1500 → trust 0.5 (average source)
- Rating 1900 → trust 0.909
- Rating 1100 → trust 0.091

### 12.3 PageRank Trust Propagation

Sources form a citation graph where an edge from source A to source B exists when A's
entries reference B's entries. Trust propagates through this graph via PageRank:

```
PR(s) = (1 - d) / N + d × Σ_{u ∈ B(s)} PR(u) / L(u)
```

Where:
- `d` = damping factor = **0.85**
- `N` = total number of sources
- `B(s)` = set of sources that link to source s
- `L(u)` = number of outgoing links from source u

The PageRank algorithm MUST iterate until convergence (L1 norm of change vector < 1e-6)
or for a maximum of 100 iterations, whichever comes first.

#### 12.3.1 PageRank to Trust Conversion

PageRank values are normalized to [0, 1]:

```
trust_pr(s) = PR(s) / max(PR(s) for all s)
```

### 12.4 Domain-Specific Trust

Sources may have different trustworthiness across domains. The trust model maintains
per-domain Elo ratings. When computing trust for an entry, the domain-specific rating
is used if available, falling back to the global rating.

```
trust_domain(s, d) = {
  trust_elo_domain(s, d)  if domain-specific data exists (≥ 5 comparisons in domain d)
  trust_elo_global(s)     otherwise
}
```

### 12.5 Composite Trust Score

The final trust score for a source blends Elo and PageRank:

```
trust(s) = w_elo × trust_elo(s) + w_pr × trust_pr(s)
```

Default weights:
- `w_elo` = **0.7** (accuracy-dominant)
- `w_pr` = **0.3** (authority contribution)

Implementations MUST ensure `w_elo + w_pr = 1.0`.

### 12.6 Trust Propagation to Entries

An entry's trust score is inherited from its source:

```
trust(entry) = trust(source(entry))
```

For entries with multiple sources (e.g., aggregated via deduplication), the trust is:

```
trust(entry) = max(trust(source_i) for all sources i of entry)
```

The `max` aggregation is used because a single highly-trusted source is sufficient to
establish trust.

### 12.7 Trust Bounds

Trust scores MUST be clamped to the range [0.05, 0.99]:

```
trust_final = clamp(trust_raw, 0.05, 0.99)
```

The lower bound of 0.05 ensures that no source is completely distrusted (allowing for
recovery). The upper bound of 0.99 ensures that no source is unconditionally trusted.

### 12.8 Trust Decay

Source trust does NOT decay over time (unlike confidence). Trust is a property of the
source, not the knowledge. However, sources that have not contributed entries in more
than 90 days SHOULD have their Elo rating regressed toward the mean:

```
R_regressed = R_current + 0.1 × (1500 - R_current)
```

This applies a 10% regression toward the initial rating of 1500 for inactive sources.

---

## 13. Conflict Resolution

### 13.1 Overview

Conflict resolution handles the case where two or more entries make contradictory
assertions. DBPS uses a multi-strategy approach that considers confidence, trust,
recency, and semantic analysis to resolve conflicts.

### 13.2 Conflict Detection

Conflicts are detected when:
1. A new entry has cosine similarity ≥ 0.85 with an existing entry AND
2. The new entry has a `contradicts` relation to the existing entry, OR
3. Semantic analysis determines the entries make incompatible assertions
   (e.g., "X is true" vs. "X is false")

Implementations MUST check for conflicts during the ingestion pipeline (Section 8,
Stage 3 or later).

### 13.3 Resolution Strategies

#### Strategy 1: Confidence-Based Resolution

The entry with higher effective confidence (after decay) is preferred:

```
winner = argmax(c_effective(entry_i) for all conflicting entries i)
```

The losing entries are NOT deleted but have their confidence reduced:

```
For each loser: β_new = β_old + (c_winner / c_loser)
```

#### Strategy 2: Trust-Based Resolution

When confidence scores are within 0.1 of each other, trust breaks the tie:

```
winner = argmax(trust(source(entry_i)) for all conflicting entries i)
```

#### Strategy 3: Recency-Based Resolution

When both confidence and trust are tied (within 0.1), the most recent entry wins:

```
winner = argmax(created_at(entry_i) for all conflicting entries i)
```

#### Strategy 4: Manual Resolution

If no automatic strategy produces a clear winner (all scores within 0.1), the conflict
is flagged for manual review. A `conflict.pending` event is emitted and the entries are
tagged with `_dbps_conflict_group: <group_id>`.

### 13.4 Resolution Cascade

Strategies are applied in order: Confidence → Trust → Recency → Manual. The first
strategy that produces a winner with margin > 0.1 terminates the cascade.

### 13.5 Conflict Graph

All conflicts are recorded in the graph store as `contradicts` edges. This allows:
- Querying all known conflicts
- Tracing the resolution history
- Re-resolving conflicts when source trust changes

### 13.6 Conflict Events

The following events MUST be emitted during conflict resolution:

| Event                  | Trigger                        |
|------------------------|--------------------------------|
| `conflict.detected`   | New contradicting entry found  |
| `conflict.resolved`   | Automatic resolution succeeded |
| `conflict.pending`    | Manual resolution required     |
| `conflict.overridden` | Manual override applied        |

---

## 14. Temporal Reasoning

### 14.1 Overview

Temporal reasoning enables queries that are time-aware, distinguishing between "what was
known at time T" (ingestion-time queries) and "what was true at time T" (event-time
queries). This dual-timeline model is essential for historical analysis and audit trails.

### 14.2 Dual Timeline Model

Every entry exists on two timelines:

1. **Ingestion Timeline** — When the entry was created in the system (`created_at`).
   This timeline is monotonically increasing and immutable.
2. **Event Timeline** — When the knowledge was true or the event occurred (`event_time`).
   This timeline MAY be non-monotonic and MAY reference past or future times.

### 14.3 Temporal Query Types

#### 14.3.1 As-Of Query

Returns entries as they were known at a specific ingestion time:

```json
{
  "method": "entries.search",
  "params": {
    "query": "speed of light",
    "temporal_mode": "as_of",
    "as_of_time": "2025-01-01T00:00:00Z"
  }
}
```

This query MUST only return entries with `created_at ≤ as_of_time` and MUST use the
confidence scores as they were at `as_of_time` (i.e., without subsequent updates).

#### 14.3.2 Event-Time Query

Returns entries whose events occurred within a time range:

```json
{
  "method": "entries.search",
  "params": {
    "query": "scientific discoveries",
    "temporal_mode": "event_range",
    "event_time_start": "1900-01-01T00:00:00Z",
    "event_time_end": "1950-12-31T23:59:59Z"
  }
}
```

#### 14.3.3 Temporal Proximity Query

Returns entries sorted by temporal proximity to a reference time:

```json
{
  "method": "entries.search",
  "params": {
    "query": "market events",
    "temporal_mode": "proximity",
    "reference_time": "2025-06-15T14:30:00Z",
    "decay_tau": 3600
  }
}
```

### 14.4 Temporal Indexing

Implementations MUST maintain B-tree indexes on both `created_at` and `event_time` fields
to support efficient temporal range queries. The `event_time` index MUST handle NULL values
(entries without event_time).

### 14.5 Temporal Versioning

When an entry's confidence or metadata is updated, the previous state MUST be preserved
in a version history. Each version records:
- The timestamp of the change
- The fields that changed
- The previous values
- The cause of the change (e.g., "confirmation from source X")

This enables as-of queries to reconstruct historical state.

### 14.6 Temporal Decay Integration

Temporal decay (Section 11.5) is applied relative to the query's temporal context:
- For current-time queries: decay from `updated_at` to `now()`
- For as-of queries: decay from `updated_at` to `as_of_time`
- For event-time queries: no decay is applied (event-time entries are timeless)

### 14.7 Future Events

Entries with `event_time` in the future represent predictions or scheduled events.
These entries SHOULD have their confidence discounted by a prediction penalty:

```
c_prediction = c_base × (1 - 0.1 × log2(1 + days_until_event))
```

This logarithmic penalty increases with the prediction horizon, reflecting the
increasing uncertainty of longer-term predictions.

---

## 15. Composite Scoring

### 15.1 Overview

Composite scoring combines signals from multiple dimensions into a single ranking score
for query results. The composite score determines the order in which entries are presented
to the user.

### 15.2 Scoring Formula

The composite score for an entry *E* given query *Q* is:

```
S(E, Q) = w₁ × relevance(E, Q) + w₂ × confidence(E) + w₃ × trust(E) + w₄ × recency(E, Q)
```

Where the default weights are:

| Weight | Symbol | Default Value | Description                    |
|--------|--------|---------------|--------------------------------|
| w₁     | w_rel  | **0.45**      | Semantic relevance weight      |
| w₂     | w_conf | **0.25**      | Confidence weight              |
| w₃     | w_trust| **0.20**      | Source trust weight             |
| w₄     | w_rec  | **0.10**      | Temporal recency weight         |

These weights MUST sum to 1.0: `w₁ + w₂ + w₃ + w₄ = 1.0`.

### 15.3 Component Definitions

#### 15.3.1 Relevance Component

```
relevance(E, Q) = sim_semantic(E.embedding, Q.embedding)
```

This is the cosine similarity between the entry's embedding and the query's embedding,
as defined in Section 5.2. Range: [-1.0, 1.0], clamped to [0.0, 1.0] for scoring.

#### 15.3.2 Confidence Component

```
confidence(E) = c_effective(E)
```

This is the entry's effective confidence after temporal decay, as defined in Section 11.6.
Range: [c_floor, 1.0].

#### 15.3.3 Trust Component

```
trust(E) = trust(source(E))
```

This is the composite trust score of the entry's source, as defined in Section 12.5.
Range: [0.05, 0.99].

#### 15.3.4 Recency Component

```
recency(E, Q) = exp(-(t_Q - t_E) / τ_recency)
```

Where:
- `t_Q` = query reference time (default: current time)
- `t_E` = entry's `event_time` or `created_at`
- `τ_recency` = recency decay constant (default: 86400 seconds = 24 hours)

Range: (0.0, 1.0].

### 15.4 Custom Weight Overrides

Clients MAY override the default weights in their query:

```json
{
  "scoring": {
    "weights": {
      "relevance": 0.60,
      "confidence": 0.20,
      "trust": 0.15,
      "recency": 0.05
    }
  }
}
```

The server MUST validate that custom weights sum to 1.0 (within tolerance ±0.001).
If weights do not sum to 1.0, the server MUST normalize them: `wᵢ_normalized = wᵢ / Σwᵢ`.

### 15.5 Score Normalization

The final composite score MUST be in the range [0.0, 1.0]. Since all components are
in [0.0, 1.0] and weights sum to 1.0, this is guaranteed by construction. However,
implementations MUST clamp the result as a safety measure:

```
S_final = clamp(S(E, Q), 0.0, 1.0)
```

### 15.6 Tie-Breaking

When two entries have identical composite scores (within ±0.0001), ties MUST be broken
by the following criteria, in order:
1. Higher confidence score
2. Higher trust score
3. More recent `created_at`
4. Lexicographically smaller `id`

### 15.7 Minimum Score Threshold

Implementations MUST support a minimum score threshold parameter. Entries with composite
scores below the threshold MUST be excluded from results. Default threshold: **0.1**.

### 15.8 Score Explanation

When `include_explanations` is true (Section 10.4), each component score and its weighted
contribution MUST be included in the response. This enables clients to understand why
entries were ranked as they were.

### 15.9 Boosting and Penalties

Implementations MAY support query-time boosting and penalties:

- **Tag boost:** Entries matching specified tags receive a multiplicative boost: `S_boosted = S × (1 + boost_factor)`, where boost_factor ∈ [0.0, 1.0]
- **Source penalty:** Entries from specified sources receive a multiplicative penalty: `S_penalized = S × (1 - penalty_factor)`, where penalty_factor ∈ [0.0, 0.5]
- **Freshness boost:** Entries created within a specified window receive a boost

After boosting/penalties, the score MUST be re-clamped to [0.0, 1.0].

---

## 16. REST API

### 16.1 Overview

The REST API provides a stateless HTTP interface to DimensionalBase. All endpoints use
JSON (RFC 8259) for request and response bodies. Authentication is via Bearer tokens
(JWT, RFC 7519) in the `Authorization` header.

### 16.2 Base URL

```
https://{host}:{port}/api/v1
```

All endpoint paths in this section are relative to the base URL.

### 16.3 Common Headers

| Header          | Direction | Required | Description                          |
|-----------------|-----------|----------|--------------------------------------|
| Authorization   | Request   | YES      | `Bearer <jwt_token>`                 |
| Content-Type    | Request   | YES      | `application/json`                   |
| Accept          | Request   | NO       | `application/json` (default)         |
| X-Channel-Id    | Request   | NO       | Channel ID for session affinity       |
| X-Request-Id    | Request   | NO       | Client-generated request trace ID    |
| X-Request-Id    | Response  | YES      | Echoed or server-generated trace ID  |
| X-RateLimit-*   | Response  | YES      | Rate limit headers (see 16.14)       |

### 16.4 Endpoint 1: Create Entry

```
POST /entries
```

Creates a new knowledge entry.

**Request Body:**
```json
{
  "content": "The speed of light is 299,792,458 m/s.",
  "source": "doc:https://physics.nist.gov/constants",
  "tags": ["physics", "constants"],
  "event_time": "1983-01-01T00:00:00Z",
  "relations": [],
  "metadata": { "language": "en" }
}
```

**Response (201 Created):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "content": "The speed of light is 299,792,458 m/s.",
  "source": "doc:https://physics.nist.gov/constants",
  "confidence": 0.85,
  "confidence_distribution": { "alpha": 1.85, "beta": 1.15 },
  "embedding": null,
  "dimensions": { "semantic": "physics.constants", "temporal": "1983-01-01T00:00:00Z", "confidence": 0.85 },
  "created_at": "2025-06-15T14:30:00.000Z",
  "updated_at": "2025-06-15T14:30:00.000Z",
  "event_time": "1983-01-01T00:00:00Z",
  "tags": ["constants", "physics"],
  "relations": [],
  "metadata": { "language": "en" }
}
```

### 16.5 Endpoint 2: Get Entry

```
GET /entries/{id}
```

Retrieves a single entry by ID.

**Path Parameters:**
- `id` (string, required): UUID of the entry

**Query Parameters:**
- `include_embedding` (boolean, default false): Include the embedding vector
- `include_history` (boolean, default false): Include version history

**Response (200 OK):** Full entry object as defined in Section 4.

**Error (404 Not Found):** `{ "error": { "code": "ENTRY_NOT_FOUND", "message": "..." } }`

### 16.6 Endpoint 3: Update Entry

```
PATCH /entries/{id}
```

Updates mutable fields of an entry. Only `tags`, `metadata`, `relations`, and `event_time`
MAY be updated. The `content`, `source`, `id`, and timestamps are immutable.

**Request Body:**
```json
{
  "tags": ["physics", "constants", "verified"],
  "metadata": { "language": "en", "review_status": "verified" }
}
```

**Response (200 OK):** Updated entry object.

### 16.7 Endpoint 4: Delete Entry

```
DELETE /entries/{id}
```

Soft-deletes an entry. The entry is marked as deleted but retained for audit purposes.
Hard deletion MUST require a separate administrative endpoint.

**Response (204 No Content)**

### 16.8 Endpoint 5: Search Entries

```
POST /entries/search
```

Performs a multi-dimensional similarity search as defined in Section 10.

**Request Body:** Search query object as defined in Section 10.2.

**Response (200 OK):**
```json
{
  "results": [ { "entry": { ... }, "score": 0.847, "explanation": { ... } } ],
  "total_count": 42,
  "has_more": true,
  "next_offset": 10
}
```

### 16.9 Endpoint 6: Batch Create Entries

```
POST /entries/batch
```

Creates multiple entries in a single atomic operation.

**Request Body:**
```json
{
  "entries": [ { "content": "...", "source": "...", ... }, ... ]
}
```

**Constraints:** Maximum batch size is `max_batch_size` (negotiated or default 1000).

**Response (201 Created):**
```json
{
  "created": [ { "id": "...", "status": "created" }, ... ],
  "count": 100
}
```

### 16.10 Endpoint 7: Confirm Entry

```
POST /entries/{id}/confirm
```

Registers a confirmation for an entry, updating its confidence distribution (Section 11.4.1).

**Request Body:**
```json
{
  "source": "user:physicist",
  "evidence": "Independently verified using cesium-133 frequency standard."
}
```

**Response (200 OK):** Updated entry with new confidence values.

### 16.11 Endpoint 8: Contradict Entry

```
POST /entries/{id}/contradict
```

Registers a contradiction against an entry, updating its confidence distribution
(Section 11.4.2) and triggering conflict resolution (Section 13).

**Request Body:**
```json
{
  "source": "user:reviewer",
  "evidence": "The stated value is incorrect for the 2019 SI redefinition.",
  "alternative_entry_id": "661f9511-f30c-42e5-b817-557766550011"
}
```

**Response (200 OK):** Updated entry with new confidence values and conflict status.

### 16.12 Endpoint 9: Get Source Trust

```
GET /sources/{source_id}/trust
```

Returns the trust profile for a source.

**Response (200 OK):**
```json
{
  "source_id": "user:physicist",
  "trust_score": 0.87,
  "elo_rating": 1823,
  "elo_trust": 0.89,
  "pagerank_trust": 0.82,
  "domain_trust": {
    "physics": 0.94,
    "chemistry": 0.71
  },
  "total_entries": 342,
  "total_confirmations": 289,
  "total_contradictions": 12
}
```

### 16.13 Endpoint 10: List Conflicts

```
GET /conflicts
```

Returns pending and resolved conflicts.

**Query Parameters:**
- `status` (string): `pending`, `resolved`, `all` (default: `pending`)
- `limit` (integer, default 20, max 100)
- `offset` (integer, default 0)

**Response (200 OK):**
```json
{
  "conflicts": [
    {
      "id": "conflict-uuid",
      "entries": ["entry-id-1", "entry-id-2"],
      "status": "pending",
      "detected_at": "2025-06-15T14:30:00.000Z",
      "resolution": null
    }
  ],
  "total_count": 5
}
```

### 16.14 Endpoint 11: Health Check

```
GET /health
```

Returns the health status of the server and its dependencies.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 86400,
  "stores": {
    "vector_index": { "status": "healthy", "entry_count": 1000000 },
    "document_store": { "status": "healthy", "entry_count": 1000000 },
    "graph_store": { "status": "healthy", "node_count": 1000000, "edge_count": 2500000 }
  }
}
```

### 16.15 Rate Limiting

All endpoints are rate-limited. Rate limit information is conveyed via response headers:

| Header                | Description                      |
|-----------------------|----------------------------------|
| X-RateLimit-Limit     | Maximum requests per window      |
| X-RateLimit-Remaining | Remaining requests in window     |
| X-RateLimit-Reset     | Unix timestamp when window resets|

Default limits:
- Read endpoints: 1000 requests per minute
- Write endpoints: 100 requests per minute
- Search endpoint: 200 requests per minute

When rate-limited, the server MUST respond with HTTP 429 and a `Retry-After` header.

### 16.16 Pagination

List endpoints MUST support cursor-based pagination via `offset` and `limit` parameters.
The `limit` parameter MUST NOT exceed 100. Responses MUST include `total_count` and
`has_more` fields.

---

## 17. WebSocket Protocol

### 17.1 Overview

The WebSocket protocol provides a persistent, bidirectional communication channel for
real-time interactions with DimensionalBase. It is the RECOMMENDED transport for
applications requiring streaming results or event subscriptions.

### 17.2 Connection Establishment

Clients connect via the WebSocket upgrade handshake at:

```
wss://{host}:{port}/ws/v1
```

Authentication MUST be provided via the `Sec-WebSocket-Protocol` subprotocol header
or as a query parameter `?token=<jwt>`.

### 17.3 Message Format

All WebSocket messages MUST be JSON-RPC 2.0 compliant:

```json
{
  "jsonrpc": "2.0",
  "method": "method.name",
  "id": 1,
  "params": { ... }
}
```

Notifications (server-initiated, no response expected) omit the `id` field.

### 17.4 Supported Methods

| Method                   | Direction       | Description                      |
|--------------------------|-----------------|----------------------------------|
| `channel.open`           | Client → Server | Initiate channel negotiation     |
| `channel.close`          | Client → Server | Close the channel                |
| `channel.ping`           | Client → Server | Keep-alive ping                  |
| `entries.create`         | Client → Server | Create a new entry               |
| `entries.get`            | Client → Server | Retrieve an entry by ID          |
| `entries.search`         | Client → Server | Search entries                   |
| `entries.search.stream`  | Client → Server | Search with streaming results    |
| `entries.confirm`        | Client → Server | Confirm an entry                 |
| `entries.contradict`     | Client → Server | Contradict an entry              |
| `subscriptions.create`   | Client → Server | Subscribe to events              |
| `subscriptions.cancel`   | Client → Server | Cancel a subscription            |

### 17.5 Streaming Search

The `entries.search.stream` method returns results incrementally:

```json
// Request
{ "jsonrpc": "2.0", "method": "entries.search.stream", "id": 5, "params": { "query": "speed of light" } }

// Response: streaming results
{ "jsonrpc": "2.0", "id": 5, "result": { "type": "result", "entry": { ... }, "score": 0.92 } }
{ "jsonrpc": "2.0", "id": 5, "result": { "type": "result", "entry": { ... }, "score": 0.87 } }
{ "jsonrpc": "2.0", "id": 5, "result": { "type": "done", "total_count": 2 } }
```

### 17.6 Event Subscriptions

Clients can subscribe to real-time events:

```json
{ "jsonrpc": "2.0", "method": "subscriptions.create", "id": 10, "params": {
  "event_types": ["entry.created", "entry.updated", "conflict.detected"],
  "filters": { "tags": ["physics"] }
}}
```

Subscription events are delivered as notifications:

```json
{ "jsonrpc": "2.0", "method": "subscription.event", "params": {
  "subscription_id": "sub-uuid",
  "event_type": "entry.created",
  "data": { "entry_id": "...", "content_preview": "..." },
  "timestamp": "2025-06-15T14:30:00.000Z"
}}
```

### 17.7 Connection Limits

- Maximum concurrent WebSocket connections per client: 10
- Maximum subscriptions per connection: 50
- Maximum message size: 1 MiB
- Server MUST close connections that exceed these limits with WebSocket close code 1008
  (Policy Violation)

### 17.8 Reconnection

Clients SHOULD implement automatic reconnection with exponential backoff:
- Initial delay: 1 second
- Maximum delay: 60 seconds
- Backoff multiplier: 2
- Jitter: ±25%

---

## 18. Model Context Protocol (MCP) Integration

### 18.1 Overview

DimensionalBase integrates with the Model Context Protocol (MCP) to provide AI models
with structured access to its knowledge store. MCP integration exposes DBPS functionality
through MCP tools and resources, enabling AI agents to query, store, and manage knowledge
within their reasoning loops.

### 18.2 MCP Server Configuration

The DBPS MCP server MUST be configurable via a standard MCP server declaration:

```json
{
  "mcpServers": {
    "dimensionalbase": {
      "command": "node",
      "args": ["./dist/mcp-server.js"],
      "env": {
        "DBPS_HOST": "localhost",
        "DBPS_PORT": "8420",
        "DBPS_AUTH_TOKEN": "${DBPS_AUTH_TOKEN}"
      }
    }
  }
}
```

### 18.3 MCP Tools

The DBPS MCP server MUST expose the following 6 tools:

#### Tool 1: `dmb_store`

Stores a new knowledge entry in DimensionalBase.

```json
{
  "name": "dmb_store",
  "description": "Store a knowledge entry in DimensionalBase with automatic embedding, confidence scoring, and conflict detection.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "content": { "type": "string", "description": "The knowledge content to store" },
      "source": { "type": "string", "description": "Source identifier (e.g., 'agent:claude')" },
      "tags": { "type": "array", "items": { "type": "string" }, "description": "Classification tags" },
      "event_time": { "type": "string", "format": "date-time", "description": "When the knowledge was true (optional)" },
      "metadata": { "type": "object", "description": "Additional metadata (optional)" }
    },
    "required": ["content", "source"]
  }
}
```

**Returns:** The created entry object with all computed fields (id, confidence, embedding dimensions, timestamps).

#### Tool 2: `dmb_query`

Performs a multi-dimensional semantic search.

```json
{
  "name": "dmb_query",
  "description": "Search DimensionalBase for relevant knowledge entries using semantic similarity, with confidence and trust-aware ranking.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": { "type": "string", "description": "Natural language search query" },
      "limit": { "type": "integer", "default": 5, "description": "Maximum results to return" },
      "min_confidence": { "type": "number", "default": 0.3, "description": "Minimum confidence threshold" },
      "tags": { "type": "array", "items": { "type": "string" }, "description": "Filter by tags" },
      "time_range": {
        "type": "object",
        "properties": {
          "start": { "type": "string", "format": "date-time" },
          "end": { "type": "string", "format": "date-time" }
        }
      }
    },
    "required": ["query"]
  }
}
```

**Returns:** Array of entries with scores and explanations.

#### Tool 3: `dmb_confirm`

Confirms an existing entry, boosting its confidence.

```json
{
  "name": "dmb_confirm",
  "description": "Confirm a knowledge entry as accurate, increasing its confidence score via Bayesian update.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "entry_id": { "type": "string", "description": "UUID of the entry to confirm" },
      "evidence": { "type": "string", "description": "Evidence supporting the confirmation" },
      "source": { "type": "string", "description": "Source of the confirmation" }
    },
    "required": ["entry_id"]
  }
}
```

**Returns:** Updated entry with new confidence distribution.

#### Tool 4: `dmb_contradict`

Registers a contradiction against an existing entry.

```json
{
  "name": "dmb_contradict",
  "description": "Register a contradiction against a knowledge entry, decreasing its confidence and triggering conflict resolution.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "entry_id": { "type": "string", "description": "UUID of the entry to contradict" },
      "evidence": { "type": "string", "description": "Evidence supporting the contradiction" },
      "source": { "type": "string", "description": "Source of the contradiction" },
      "alternative_content": { "type": "string", "description": "The corrected information (optional)" }
    },
    "required": ["entry_id"]
  }
}
```

**Returns:** Updated entry with new confidence distribution and conflict status.

#### Tool 5: `dmb_inspect`

Retrieves detailed information about an entry including its full dimensional profile.

```json
{
  "name": "dmb_inspect",
  "description": "Get detailed information about a knowledge entry including confidence distribution, trust lineage, and relational context.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "entry_id": { "type": "string", "description": "UUID of the entry to inspect" },
      "include_relations": { "type": "boolean", "default": true, "description": "Include related entries" },
      "include_history": { "type": "boolean", "default": false, "description": "Include version history" },
      "relation_depth": { "type": "integer", "default": 1, "description": "Depth of relation traversal" }
    },
    "required": ["entry_id"]
  }
}
```

**Returns:** Full entry object with expanded relations, confidence history, and trust lineage.

#### Tool 6: `dmb_status`

Returns the status and statistics of the DimensionalBase instance.

```json
{
  "name": "dmb_status",
  "description": "Get the current status, health, and statistics of the DimensionalBase knowledge store.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "include_store_details": { "type": "boolean", "default": false, "description": "Include per-store statistics" }
    },
    "required": []
  }
}
```

**Returns:** Health status, entry counts, store statistics, and version information.

### 18.4 MCP Resources

The DBPS MCP server MUST expose the following 4 resources:

#### Resource 1: `dmb://entries/{id}`

Provides access to individual entries as MCP resources.

```json
{
  "uri": "dmb://entries/{id}",
  "name": "Knowledge Entry",
  "description": "A single knowledge entry in DimensionalBase",
  "mimeType": "application/json"
}
```

#### Resource 2: `dmb://sources/{source_id}/profile`

Provides trust profile information for a source.

```json
{
  "uri": "dmb://sources/{source_id}/profile",
  "name": "Source Trust Profile",
  "description": "Trust scores, Elo rating, and history for a knowledge source",
  "mimeType": "application/json"
}
```

#### Resource 3: `dmb://conflicts/pending`

Lists all pending conflicts requiring resolution.

```json
{
  "uri": "dmb://conflicts/pending",
  "name": "Pending Conflicts",
  "description": "Knowledge entries with unresolved contradictions",
  "mimeType": "application/json"
}
```

#### Resource 4: `dmb://stats/overview`

Provides a summary of the knowledge store.

```json
{
  "uri": "dmb://stats/overview",
  "name": "Store Overview",
  "description": "Summary statistics of the DimensionalBase knowledge store including entry counts, confidence distributions, and store health",
  "mimeType": "application/json"
}
```

### 18.5 MCP Tool Error Handling

MCP tool errors MUST be returned as structured error objects:

```json
{
  "content": [
    {
      "type": "text",
      "text": "Error: Entry not found (ENTRY_NOT_FOUND). The entry with ID '...' does not exist."
    }
  ],
  "isError": true
}
```

### 18.6 MCP Resource Templates

Implementations SHOULD support MCP resource templates for dynamic resource discovery:

```json
{
  "uriTemplate": "dmb://entries?tags={tags}&min_confidence={min_confidence}",
  "name": "Filtered Entries",
  "description": "Browse entries filtered by tags and confidence"
}
```

### 18.7 MCP Prompts (OPTIONAL)

Implementations MAY expose MCP prompts for guided knowledge interactions:

- `dmb_research` — Guides the AI through a multi-step knowledge research workflow
- `dmb_audit` — Guides the AI through auditing low-confidence or conflicting entries

---

## 19. Security

### 19.1 Overview

Security in DBPS covers authentication, authorization, transport security, and data
protection. All implementations MUST enforce the security requirements in this section.

### 19.2 Transport Security

All DBPS communications MUST use TLS 1.2 or higher. TLS 1.3 is RECOMMENDED.
Implementations MUST NOT support SSL or TLS versions below 1.2. Certificate validation
MUST be enforced; self-signed certificates MAY be accepted only in development
environments with explicit configuration.

### 19.3 Authentication

DBPS supports the following authentication mechanisms:

1. **Bearer Token (JWT)** — REQUIRED. All implementations MUST support JWT authentication.
   Tokens MUST use RS256 or ES256 signing algorithms. Tokens MUST include `exp` (expiration),
   `iat` (issued at), `sub` (subject/client ID), and `scope` (permissions) claims.

2. **API Key** — OPTIONAL. Implementations MAY support static API keys for simple
   deployments. API keys MUST be transmitted via the `X-API-Key` header, never in query
   parameters.

3. **mTLS** — OPTIONAL. Implementations MAY support mutual TLS for service-to-service
   authentication.

### 19.4 Authorization

DBPS defines the following permission scopes:

| Scope              | Description                                    |
|--------------------|------------------------------------------------|
| `entries:read`     | Read entries and perform searches               |
| `entries:write`    | Create and update entries                        |
| `entries:delete`   | Soft-delete entries                              |
| `entries:confirm`  | Confirm entries                                  |
| `entries:contradict` | Contradict entries                             |
| `sources:read`     | Read source trust profiles                       |
| `conflicts:read`   | Read conflict information                        |
| `conflicts:resolve`| Resolve conflicts manually                      |
| `admin:*`          | Full administrative access                       |

Authorization MUST be checked on every request. Unauthorized requests MUST receive HTTP
403 Forbidden with error code `INSUFFICIENT_PERMISSIONS`.

### 19.5 Input Validation

All inputs MUST be validated against the schemas defined in this specification. SQL
injection, NoSQL injection, and script injection MUST be mitigated through parameterized
queries and content sanitization. Implementations MUST NOT evaluate user-supplied content
as code.

### 19.6 Rate Limiting and Abuse Prevention

Rate limiting is defined in Section 16.15. Additionally:
- Implementations MUST detect and block brute-force authentication attempts (max 10
  failed attempts per minute per IP)
- Implementations MUST limit request body size to 1 MiB for single-entry operations
  and 10 MiB for batch operations
- Implementations SHOULD implement request timeout of 30 seconds for all operations

### 19.7 Data Protection

- Embeddings MUST NOT be reversible to original content (note: this is a property of
  the embedding model, not the implementation, but implementations MUST NOT store
  additional data that would enable reversal)
- Implementations MUST support encryption at rest for all stored data
- Audit logs MUST record all write operations with timestamp, client ID, and operation type
- Soft-deleted entries MUST be purged after a configurable retention period (default: 90 days)

### 19.8 Secrets Management

- Authentication tokens MUST NOT be logged
- Database credentials MUST NOT be stored in plaintext configuration files
- Implementations MUST support environment variable or secrets manager integration for
  all sensitive configuration

---

## 20. Error Handling

### 20.1 Overview

DBPS defines a structured error model that provides consistent, machine-readable error
information across all transports (REST, WebSocket, MCP).

### 20.2 Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "numeric_code": 4001,
    "message": "Human-readable error description.",
    "details": {
      "field": "content",
      "constraint": "max_length",
      "value": 70000,
      "limit": 65536
    },
    "request_id": "req-uuid",
    "timestamp": "2025-06-15T14:30:00.000Z"
  }
}
```

### 20.3 Error Code Registry

#### Client Errors (4xxx)

| Numeric Code | String Code                  | HTTP Status | Description                               |
|-------------|------------------------------|-------------|-------------------------------------------|
| 4001        | SCHEMA_VALIDATION_ERROR       | 400         | Entry schema validation failed            |
| 4002        | INVALID_QUERY                 | 400         | Search query is malformed                 |
| 4003        | INVALID_WEIGHTS               | 400         | Scoring weights do not sum to 1.0         |
| 4004        | BATCH_SIZE_EXCEEDED           | 400         | Batch exceeds max_batch_size              |
| 4005        | PAYLOAD_TOO_LARGE             | 413         | Request body exceeds size limit           |
| 4010        | AUTH_FAILED                   | 401         | Authentication failed                     |
| 4011        | TOKEN_EXPIRED                 | 401         | JWT token has expired                     |
| 4012        | INSUFFICIENT_PERMISSIONS      | 403         | Missing required scope                    |
| 4020        | UNSUPPORTED_VERSION           | 400         | Protocol version not supported            |
| 4030        | ENTRY_NOT_FOUND               | 404         | Entry with given ID does not exist        |
| 4031        | SOURCE_NOT_FOUND              | 404         | Source with given ID does not exist       |
| 4040        | DUPLICATE_ENTRY               | 409         | Duplicate entry from same source          |
| 4041        | CONFLICT_VERSION_MISMATCH     | 409         | Optimistic concurrency conflict           |
| 4050        | RATE_LIMITED                  | 429         | Rate limit exceeded                       |

#### Server Errors (5xxx)

| Numeric Code | String Code                  | HTTP Status | Description                               |
|-------------|------------------------------|-------------|-------------------------------------------|
| 5001        | EMBEDDING_GENERATION_FAILED   | 502         | Embedding model API failure               |
| 5002        | STORAGE_ERROR                 | 500         | Storage backend failure                   |
| 5003        | INDEX_ERROR                   | 500         | Vector index failure                      |
| 5010        | INTERNAL_ERROR                | 500         | Unclassified internal error               |
| 5020        | SERVICE_UNAVAILABLE           | 503         | Server is shutting down or overloaded     |
| 5030        | SERVER_OVERLOADED             | 503         | Server cannot accept new connections      |

### 20.4 Error Handling Requirements

1. Implementations MUST return the appropriate error code for every failure.
2. Implementations MUST NOT expose internal stack traces or implementation details in
   production error responses.
3. Implementations MUST log full error details (including stack traces) server-side.
4. Implementations MUST include `request_id` in all error responses for correlation.
5. For batch operations, per-entry errors MUST be returned in an `errors` array with
   the index of the failed entry.

### 20.5 Retry Guidance

| Error Code              | Retryable | Backoff Strategy          |
|------------------------|-----------|---------------------------|
| SCHEMA_VALIDATION_ERROR | No        | Fix request and retry     |
| AUTH_FAILED             | No        | Re-authenticate           |
| TOKEN_EXPIRED           | Yes       | Refresh token and retry   |
| RATE_LIMITED            | Yes       | Wait per Retry-After      |
| EMBEDDING_GENERATION_FAILED | Yes  | Exponential backoff, max 3|
| STORAGE_ERROR           | Yes       | Exponential backoff, max 3|
| SERVICE_UNAVAILABLE     | Yes       | Exponential backoff, max 5|
| SERVER_OVERLOADED       | Yes       | Wait per Retry-After      |

---

## 21. Versioning and Migration

### 21.1 Protocol Versioning

DBPS uses semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR** version increments indicate breaking changes. Clients MUST NOT assume
  compatibility across major versions.
- **MINOR** version increments indicate backward-compatible additions. Clients SHOULD
  handle unknown fields gracefully.
- **PATCH** version increments indicate backward-compatible bug fixes.

### 21.2 Version Negotiation

Version negotiation occurs during channel handshake (Section 7). The server MUST support
the exact major.minor version requested. If the server supports a higher patch version,
it MUST use the higher patch version and indicate this in the response.

### 21.3 Deprecation Policy

Features marked as deprecated MUST remain functional for at least 2 minor versions after
the deprecation notice. Deprecated features MUST be documented in the API response via
a `Deprecation` header (RFC 8594) and a `Sunset` header indicating the removal date.

### 21.4 Schema Migration

When the entry schema changes between versions:

1. New OPTIONAL fields MAY be added in minor versions. Existing entries MUST be readable
   without the new field (the field defaults to null or its specified default).
2. Existing fields MUST NOT be removed or have their type changed in minor versions.
3. Major version upgrades MAY require data migration. Implementations MUST provide
   migration tools that:
   - Back up all data before migration
   - Migrate data atomically (all or nothing)
   - Validate migrated data against the new schema
   - Support rollback to the previous version

### 21.5 Wire Format Stability

The JSON wire format for entries (Section 4) is considered stable within a major version.
Implementations MUST accept entries serialized by any implementation of the same major
version.

### 21.6 Embedding Model Migration

When the configured embedding model changes:
1. The implementation MUST re-embed all existing entries (Section 9.6)
2. During migration, queries MUST continue to work against old embeddings
3. Once migration is complete, old embeddings MUST be replaced atomically
4. The implementation MUST record the embedding model version in entry metadata

### 21.7 Backward Compatibility Testing

Implementations MUST pass the conformance test suite (Section 22) for all supported
versions. Backward compatibility tests MUST verify that:
- Entries created under version N can be read under version N+1
- Queries valid under version N produce equivalent results under version N+1
- Channel negotiation with version N clients succeeds on version N+1 servers

---

## 22. Conformance and Test Vectors

### 22.1 Overview

This section defines the requirements for conformance testing of DBPS implementations.
A conforming implementation MUST pass all test vectors defined herein and MUST satisfy
all normative requirements (MUST, MUST NOT, SHALL, SHALL NOT, REQUIRED) in this
specification.

### 22.2 Conformance Levels

DBPS defines three conformance levels:

| Level    | Requirements                                                        |
|----------|---------------------------------------------------------------------|
| Core     | Sections 4-6, 8-11, 15-16, 20 (entry CRUD, search, confidence, REST)|
| Extended | Core + Sections 7, 12-14, 17 (channels, trust, conflicts, temporal, WS)|
| Full     | Extended + Sections 18-19 (MCP integration, full security)           |

### 22.3 Test Vector Format

Test vectors are provided as JSON documents with the following structure:

```json
{
  "test_id": "DBPS-TV-001",
  "section": "11.3",
  "description": "Confidence initialization from source trust",
  "input": { ... },
  "expected_output": { ... },
  "tolerance": 0.001
}
```

### 22.4 Confidence Test Vectors

#### TV-001: Confidence Initialization

```json
{
  "test_id": "DBPS-TV-001",
  "section": "11.3",
  "description": "Beta distribution initialization from confidence value",
  "inputs": [
    { "confidence": 0.5, "expected_alpha": 1.5, "expected_beta": 1.5 },
    { "confidence": 0.8, "expected_alpha": 1.8, "expected_beta": 1.2 },
    { "confidence": 0.0, "expected_alpha": 1.0, "expected_beta": 2.0 },
    { "confidence": 1.0, "expected_alpha": 2.0, "expected_beta": 1.0 },
    { "confidence": 0.33, "expected_alpha": 1.33, "expected_beta": 1.67 }
  ],
  "tolerance": 0.001
}
```

#### TV-002: Confidence Confirmation Update

```json
{
  "test_id": "DBPS-TV-002",
  "section": "11.4.1",
  "description": "Confidence update after single confirmation",
  "input": { "alpha": 1.8, "beta": 1.2 },
  "operation": "confirm",
  "expected_output": { "alpha": 2.8, "beta": 1.2 },
  "expected_mean": 0.7,
  "tolerance": 0.001
}
```

#### TV-003: Confidence Contradiction Update

```json
{
  "test_id": "DBPS-TV-003",
  "section": "11.4.2",
  "description": "Confidence update after single contradiction",
  "input": { "alpha": 1.8, "beta": 1.2 },
  "operation": "contradict",
  "expected_output": { "alpha": 1.8, "beta": 2.2 },
  "expected_mean": 0.45,
  "tolerance": 0.001
}
```

#### TV-004: Temporal Decay

```json
{
  "test_id": "DBPS-TV-004",
  "section": "11.5",
  "description": "Confidence decay over time with default half-life",
  "input": { "alpha": 1.8, "beta": 1.2, "base_confidence": 0.6 },
  "half_life": 7200,
  "test_cases": [
    { "elapsed_seconds": 0, "expected_confidence": 0.6 },
    { "elapsed_seconds": 7200, "expected_confidence": 0.3 },
    { "elapsed_seconds": 14400, "expected_confidence": 0.15 },
    { "elapsed_seconds": 3600, "expected_confidence": 0.4243 },
    { "elapsed_seconds": 36000, "expected_confidence": 0.01875 }
  ],
  "tolerance": 0.01
}
```

### 22.5 Trust Test Vectors

#### TV-005: Elo Rating Update

```json
{
  "test_id": "DBPS-TV-005",
  "section": "12.2",
  "description": "Elo rating update after pairwise comparison",
  "input": { "rating_a": 1500, "rating_b": 1500 },
  "outcome": "a_wins",
  "K": 32,
  "expected_output": { "rating_a": 1516, "rating_b": 1484 },
  "tolerance": 1.0
}
```

#### TV-006: Elo to Trust Conversion

```json
{
  "test_id": "DBPS-TV-006",
  "section": "12.2.3",
  "description": "Conversion of Elo rating to trust score",
  "inputs": [
    { "rating": 1500, "expected_trust": 0.5 },
    { "rating": 1900, "expected_trust": 0.909 },
    { "rating": 1100, "expected_trust": 0.091 },
    { "rating": 2000, "expected_trust": 0.9686 },
    { "rating": 1000, "expected_trust": 0.0314 }
  ],
  "tolerance": 0.01
}
```

#### TV-007: PageRank Convergence

```json
{
  "test_id": "DBPS-TV-007",
  "section": "12.3",
  "description": "PageRank computation on simple 3-node graph",
  "input": {
    "nodes": ["A", "B", "C"],
    "edges": [
      { "from": "A", "to": "B" },
      { "from": "B", "to": "C" },
      { "from": "C", "to": "A" }
    ],
    "damping": 0.85
  },
  "expected_output": {
    "A": 0.333,
    "B": 0.333,
    "C": 0.333
  },
  "tolerance": 0.01,
  "note": "Symmetric cycle should produce equal PageRank values"
}
```

### 22.6 Composite Scoring Test Vectors

#### TV-008: Default Weight Scoring

```json
{
  "test_id": "DBPS-TV-008",
  "section": "15.2",
  "description": "Composite score with default weights",
  "input": {
    "relevance": 0.92,
    "confidence": 0.87,
    "trust": 0.91,
    "recency": 0.85
  },
  "weights": { "w1": 0.45, "w2": 0.25, "w3": 0.20, "w4": 0.10 },
  "expected_score": 0.8985,
  "computation": "0.45*0.92 + 0.25*0.87 + 0.20*0.91 + 0.10*0.85 = 0.414+0.2175+0.182+0.085 = 0.8985",
  "tolerance": 0.001
}
```

#### TV-009: Custom Weight Scoring

```json
{
  "test_id": "DBPS-TV-009",
  "section": "15.4",
  "description": "Composite score with custom weights emphasizing relevance",
  "input": {
    "relevance": 0.70,
    "confidence": 0.95,
    "trust": 0.60,
    "recency": 0.40
  },
  "weights": { "w1": 0.70, "w2": 0.10, "w3": 0.10, "w4": 0.10 },
  "expected_score": 0.685,
  "computation": "0.70*0.70 + 0.10*0.95 + 0.10*0.60 + 0.10*0.40 = 0.49+0.095+0.06+0.04 = 0.685",
  "tolerance": 0.001
}
```

### 22.7 Schema Validation Test Vectors

#### TV-010: Valid Entry

```json
{
  "test_id": "DBPS-TV-010",
  "section": "4",
  "description": "A fully valid entry with all fields populated",
  "input": {
    "content": "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "source": "doc:https://chemistry.reference.org/water",
    "tags": ["chemistry", "phase-transitions", "water"],
    "event_time": null,
    "relations": [],
    "metadata": { "language": "en", "domain": "chemistry" }
  },
  "expected_result": "accepted",
  "validation_errors": []
}
```

#### TV-011: Invalid Entry (Content Too Long)

```json
{
  "test_id": "DBPS-TV-011",
  "section": "4.2",
  "description": "Entry rejected due to content exceeding 65536 bytes",
  "input": {
    "content": "<65537 bytes of UTF-8 text>",
    "source": "user:tester"
  },
  "expected_result": "rejected",
  "expected_error_code": "SCHEMA_VALIDATION_ERROR",
  "expected_error_field": "content"
}
```

### 22.8 Channel Negotiation Test Vectors

#### TV-012: Successful Handshake

```json
{
  "test_id": "DBPS-TV-012",
  "section": "7.2",
  "description": "Successful channel negotiation with capability matching",
  "input": {
    "protocol_version": "1.0.0",
    "capabilities": {
      "streaming": true,
      "batch_operations": true,
      "max_embedding_dimensions": 3072,
      "supported_similarity_functions": ["cosine", "dot_product"],
      "compression": ["gzip", "none"]
    }
  },
  "expected_result": "channel_opened",
  "expected_negotiated": {
    "streaming": true,
    "batch_operations": true,
    "similarity_function": "cosine",
    "compression": "gzip"
  }
}
```

### 22.9 Similarity Search Test Vectors

#### TV-013: Cosine Similarity

```json
{
  "test_id": "DBPS-TV-013",
  "section": "5.2",
  "description": "Cosine similarity between L2-normalized vectors",
  "input": {
    "vector_a": [0.6, 0.8, 0.0],
    "vector_b": [0.8, 0.6, 0.0]
  },
  "expected_similarity": 0.96,
  "tolerance": 0.001,
  "note": "Both vectors are already L2-normalized (‖a‖=1.0, ‖b‖=1.0)"
}
```

#### TV-014: Temporal Proximity

```json
{
  "test_id": "DBPS-TV-014",
  "section": "5.3",
  "description": "Temporal proximity with default decay constant",
  "input": {
    "time_a": 1718458200,
    "time_b": 1718544600,
    "tau": 86400
  },
  "expected_similarity": 0.3679,
  "tolerance": 0.01,
  "computation": "exp(-86400/86400) = exp(-1) ≈ 0.3679"
}
```

### 22.10 Conformance Test Execution

#### 22.10.1 Test Environment

Conformance tests MUST be executed against a clean DBPS instance with:
- No pre-existing entries
- Default configuration values
- A test source `test:conformance` with trust score 0.5

#### 22.10.2 Test Ordering

Tests MUST be executable in any order. Each test MUST be independent and MUST clean up
any entries it creates. Tests MUST NOT depend on side effects of other tests.

#### 22.10.3 Tolerance

Floating-point comparisons MUST use the tolerance specified in each test vector. If no
tolerance is specified, the default tolerance is ±0.001.

#### 22.10.4 Reporting

Conformance test results MUST be reported in the following format:

```json
{
  "implementation": "DimensionalBase Reference",
  "version": "1.0.0",
  "conformance_level": "Full",
  "test_date": "2025-06-15T14:30:00.000Z",
  "results": {
    "total": 14,
    "passed": 14,
    "failed": 0,
    "skipped": 0
  },
  "details": [
    { "test_id": "DBPS-TV-001", "status": "passed", "actual": { ... }, "expected": { ... } }
  ]
}
```

### 22.11 Self-Certification

Implementations that pass all test vectors for a given conformance level MAY self-certify
compliance by including the following in their documentation:

> This implementation conforms to DBPS v1.0.0 at the [Core|Extended|Full] level, as
> verified by the DBPS Conformance Test Suite on [date].

### 22.12 Test Vector Updates

Test vectors are versioned alongside the specification. New test vectors MAY be added in
minor specification updates. Existing test vectors MUST NOT be modified or removed within
a major version, except to correct errors in the test vector itself (with clear errata
documentation).

---

## Appendix A: JSON Schema

The complete JSON Schema for the DBPS Entry type is available as a separate artifact at:

```
https://schema.dimensionalbase.dev/v1/entry.schema.json
```

## Appendix B: Reference Implementation

The reference implementation of DBPS v1.0.0 is available at:

```
https://github.com/dimensionalbase/dbps-reference
```

## Appendix C: Change Log

| Version | Date       | Changes                    |
|---------|------------|----------------------------|
| 1.0.0   | 2026-04-10 | Initial specification      |

---

*End of DimensionalBase Protocol Specification (DBPS) Version 1.0.0*
