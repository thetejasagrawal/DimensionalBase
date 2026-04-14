# DimensionalBase — Project Vision & Philosophy

## The Core Insight

When an AI agent understands something, it has a **rich mathematical representation** inside its model — a high-dimensional vector (embedding) that encodes meaning, relationship, nuance, and context. When that agent shares its understanding with another agent by writing text, two lossy compressions happen:

1. **Embedding → Text** (Agent A): The full manifold gets collapsed into tokens. Most of the structure is gone.
2. **Text → Embedding** (Agent B): The tokens get re-encoded into a *different* manifold. Information is fabricated to fill in the gaps.

Every single exchange in a multi-agent text-passing system compounds these losses. The more agents, the more hops, the worse it gets.

**DimensionalBase's answer:** Make the shared space native to the mathematical representation. Text is still there for humans. But agents exchange through embeddings — and eventually, in Phase 4, through raw KV cache slices (no language at all).

---

## Philosophy

### 1. Embedding IS the Data, Text IS the Shadow
Every knowledge entry has both a value (text) and an embedding (vector). The embedding is the primary carrier of meaning. Text is a human-readable annotation on top of that meaning.

### 2. Four Methods Is Enough
The entire agent-facing API is: `put`, `get`, `subscribe`, `unsubscribe`. This was a deliberate design constraint. Complex systems fail because they give agents too many primitives. The richness lives in the automatic layer (reasoning, trust, algebra), not in the API surface.

### 3. Budget-Aware Retrieval as a First-Class Citizen
Token budgets are not an afterthought. Every `get()` call accepts a `budget` parameter. The system packs the most relevant knowledge that fits — using semantic similarity, recency, confidence, and reference distance as signals. Agents never overflow their context windows.

### 4. Active Reasoning as the Fourth Participant
In every multi-agent system, someone needs to watch for contradictions, gaps, and stale data. Typically no one does. DimensionalBase's `ActiveReasoning` module automatically:
- Detects when two agents state contradictory facts
- Detects when a plan references steps that lack observations
- Detects when facts go stale
- Auto-summarizes when a prefix gets crowded

### 5. Trust Is Earned, Not Declared
Agents don't just say "I'm confident." The system tracks which agents actually get confirmed vs. contradicted by other agents over time. Trust is domain-specific — an agent trusted for database queries may not be trusted for UI decisions. Trust scores use a Bayesian Elo-like system that weights surprises (unexpected confirmations/contradictions change trust more).

### 6. The TCP/IP Playbook
The goal is not to be a product. The goal is to become a **standard** — the coordination layer underneath every multi-agent framework, the way TCP/IP became the standard underneath every internet application. The path: build something that works, open-source it, let developers adopt it because it's measurably better, then formalize. DBPS v1.0 is the formalization.

---

## What DimensionalBase Is Not

- **Not a vector database** (Pinecone, Chroma, Weaviate). Those are for RAG over static documents. DimensionalBase is for live agent coordination — short-lived, highly dynamic, semantically rich.
- **Not an agent memory system** (Mem0, Supermemory). Those give each agent its own memory. DimensionalBase is a *shared* knowledge space — the coordination layer between agents, not within one.
- **Not a tool-calling protocol** (MCP). MCP handles tool invocation. DimensionalBase handles the shared knowledge state that agents use when deciding what tools to call.
- **Not an agent framework** (LangChain, CrewAI, AutoGen). Those are the frameworks. DimensionalBase is the layer underneath that any framework can plug into.

---

## Competitive Positioning

| System | What It Does | Gap |
|--------|-------------|-----|
| LangChain | Sequential agent chains | Text-passing, exponential token growth |
| CrewAI | Role-based agent teams | Tasks cascade, no shared state |
| AutoGen | Conversational multi-agent | Context windows fill up fast |
| Mem0 | Per-agent memory | Isolated memory, -6pt accuracy at peak compression |
| MCP | Tool invocation protocol | No knowledge sharing between agents |
| **DimensionalBase** | **Shared semantic coordination layer** | **Solves the inter-agent coordination gap** |

The positioning is: DimensionalBase **complements** all of these systems. It runs underneath. LangChain agents can use it for memory. CrewAI agents can use it as their shared context. MCP tools can read/write from it. It doesn't compete — it enables.

---

## Roadmap (5 Phases)

### Phase 0: Validate ✅ COMPLETE
Python prototype. SQLite + semantic vector index. CLI agents. Kill criterion: <50% token reduction.
**Result achieved:** 92–93% token reduction.

### Phase 1: Ship ✅ COMPLETE
Production Python package. Full 4-method API. Deterministic startup with explicit embedding configuration. LangChain + CrewAI integrations.
**Current OSS core release:** v0.5.0.

### Phase 2: Intelligence ✅ COMPLETE
Active reasoning. Contradiction detection. Gap detection. Bayesian confidence. Agent trust. Provenance tracking.
**All implemented** in the current OSS core.

### Phase 3: Ecosystem 🚧 IN PROGRESS
- TypeScript client (not started)
- MCP bridge ✅ (done, `mcp/server.py`)
- REST API ✅ (done, FastAPI in `server/app.py`)
- CLI ✅ (done, Click in `cli/main.py`)
- Core hardening ✅ (durable semantic index, secured REST wrapper, encryption, conformance coverage)
- Cross-encoder re-ranking ✅ (optional second-pass for document QA accuracy)
- Web dashboard ✅ (dark-themed SPA at `/dashboard/`)
- Interactive demo ✅ (`db demo` — 30-second terminal experience)
- LongBench v2 head-to-head benchmark ✅ (vs Naive RAG, Full Context, Latent Briefing)
- Docs site (this docs folder is the foundation)
- Mem0/Supermemory integration bridge (not started)

### Phase 4: Tensor Channel 🔬 RESEARCH (12 weeks)
KV cache sharing between agents on shared GPU hardware. TransferEngine integration. Sub-symbolic communication — no language at all, just raw model states.
**Status:** Placeholder infrastructure in place (`channels/tensor.py`).

### Phase 5: Standardization 📋 FUTURE
Document the running code. Submit DBPS v1.0 to Agentic AI Foundation (Linux Foundation analog). Make it the official standard.
**Status:** DBPS v1.0 spec written (`spec/dbps-v1.0.md`, 95KB, 22 sections).

---

## The Information Loss Problem (Technical Detail)

Current multi-agent text-passing creates information loss at every hop:

```
Turn 1: Agent A generates embedding E_A (rich, 1536-dim)
        Agent A compresses E_A → text T_A (lossy: ~17% info retained)
Turn 2: Agent B reads T_A
        Agent B encodes T_A → embedding E_B (different manifold, fabricated structure)
        Agent B writes analysis T_B
Turn 3: Agent C reads T_A + T_B
        Information loss compounds
```

Measured consequences in production multi-agent systems:
- **41–87% failure rate** — Tasks fail or produce wrong answers
- **17× error amplification** — Small errors in A become large errors in C
- **72% token waste** — Redundant context re-injected every turn

DimensionalBase replaces the T → T pathway with E → E:

```
Turn 1: Agent A puts (path="task/auth", embedding=E_A, value=T_A)
        Both representations stored, E_A is primary
Turn 2: Agent B gets (scope="task/**", budget=500)
        Retrieves E_A (high-fidelity), sees T_A as annotation
        Budget-aware: only the relevant entries, scored by similarity
Turn 3: Agent C gets (scope="task/**", budget=500, query="deployment blocker?")
        Semantic retrieval: only entries relevant to C's query
        ActiveReasoning: CONFLICT event fired if C's write contradicts A
```

---

## Key Design Decisions and Why

### Decision 1: SQLite, not PostgreSQL
Reason: Zero-dependency deployment. Agents can start DimensionalBase in-memory (`:memory:`) for testing, or point to a file for persistence. No external database process required. ACID transactions handle concurrent writes from multiple agents.

### Decision 2: float32, not float64
Reason: 384-dim vectors at float32 = 1.5KB per entry. float64 would double that. The precision difference (<0.001 in cosine similarity) is negligible for semantic search.

### Decision 3: Pre-normalized embeddings
Reason: L2-normalization in the hot path is expensive. By pre-normalizing on write (`||v|| = 1`), cosine similarity becomes a pure dot product: `dot(a, b) = cosine_similarity(a, b)`. This lets the entire vector matrix be scored in one BLAS call.

### Decision 4: Single VectorStore shared across channels
Reason: Before v0.3, each channel had its own vector array. This duplicated memory and prevented cross-channel operations. The unified `VectorStore` (`storage/vectors.py`) is the single source of truth. Channels share a reference to it.

### Decision 5: Bayesian Beta distributions for confidence, not averaging
Reason: Simple averaging doesn't handle small samples correctly. A brand-new agent with one confirmation should have high uncertainty, not high confidence. Beta distributions naturally encode "how many observations, how many successes" — they give wide posteriors for small samples and narrow posteriors as evidence accumulates.

### Decision 6: Elo-like trust, not simple counters
Reason: Confirmations from low-trust agents mean less than confirmations from high-trust agents. The Elo framework handles this naturally — unexpected events (low-trust agent confirms correctly) cause bigger updates than expected events (high-trust agent confirms correctly).
