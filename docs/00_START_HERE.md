# DimensionalBase (DMB) — Start Here

> **For AI agents:** Read this file first. It tells you what this project is, what problem it solves, and where to find everything else in this `docs/` folder.

---

## What Is This Project?

**DimensionalBase** is a protocol and shared knowledge database built from scratch for AI agent coordination. It is written in Python and is currently at version **0.5.0**.

The core insight: AI agents communicate through text today. Text destroys the rich mathematical representations (embeddings) that live inside each model. Every text-to-text hop loses information in both directions. DimensionalBase makes the **embedding the primary data** — text is just the human-readable shadow.

**One-line summary:** A 4-method API (`put`, `get`, `subscribe`, `unsubscribe`) that gives any group of AI agents a shared semantic knowledge space with automatic contradiction detection, budget-aware retrieval, and multi-channel communication.

---

## The Problem Being Solved

Traditional multi-agent systems (LangChain, CrewAI, AutoGen) pass context as raw text:

```
Agent A writes → "JWT signing key expired. Auth returning 401."
Agent B reads  → same text string (most semantic information is lost in conversion)
Agent C reads  → same text string (information keeps degrading)
```

Challenges with text-passing (from published multi-agent evaluations):
- Multi-agent systems fail **41–87%** of the time
- Errors can amplify **17x** between agents
- Up to **72%** of tokens are wasted on redundant context

DimensionalBase replaces this with:
```
Agent A puts  → embedding + text at path "task/auth/status"
Agent B gets  → relevant entries within its token budget (semantic retrieval)
Agent C gets  → same, independently, only what it needs
```

Results from internal synthetic benchmarks vs. text-passing baseline:
- **92–93% token reduction** (consistent across all benchmarks)
- **Improved contradiction detection** (3–4/5 at small scale; up to 8/8 vs 4/8 in large-scale benchmark)
- **Pipeline gap detection** (automated, zero code)

See [`04_BENCHMARKS_AND_COMPARISONS.md`](04_BENCHMARKS_AND_COMPARISONS.md) for full methodology, raw numbers, and honest caveats.

---

## Docs Index

| File | What It Covers |
|------|---------------|
| `00_START_HERE.md` | **This file** — orientation, problem, docs map |
| `01_PROJECT_VISION.md` | Full vision, philosophy, long-term goals, roadmap |
| `02_ARCHITECTURE.md` | Technical architecture — all 20 modules explained |
| `03_API_REFERENCE.md` | Complete API: methods, types, exceptions, examples |
| `04_BENCHMARKS_AND_COMPARISONS.md` | All benchmark results and competitor comparisons |
| `05_CODEBASE_MAP.md` | Every file, what it does, how modules connect |
| `06_AGENT_WORKING_GUIDE.md` | How to work on this codebase — patterns, conventions, testing |

---

## Quick Facts

| Property | Value |
|---|---|
| Language | Python 3.9+ |
| Version | 0.5.0 |
| License | MIT |
| Storage | SQLite (built-in) + durable embeddings table + NumPy float32 `VectorStore` |
| Embedding models | sentence-transformers (local) or OpenAI text-embedding-3-small |
| Public API | 4 methods: `put`, `get`, `subscribe`, `unsubscribe` |
| Integration targets | MCP (Claude Code, Cursor), LangChain, CrewAI, REST API, CLI |
| Test files | 17 files + conformance suite |
| Spec | DBPS v1.0 (95KB formal specification in `spec/dbps-v1.0.md`) |

---

## Current Guarantees

| Surface | Current state |
|---|---|
| Embedded library | Supported, including semantic index reload for file-backed DBs |
| Secured REST server | Supported, and the packaged server/CLI entrypoints bootstrap secure mode by default when an API key is configured |
| MCP | Supported for local/trusted workflows, not the hardened network surface |
| CLI | Supported for local operations and inspection |
| TTL cleanup | Explicit only, never automatic on `close()` |
| Tensor channel | Placeholder only; not implemented |
| Cross-encoder re-ranking | Supported via `rerank=True`; uses local cross-encoder for document QA accuracy |
| Web dashboard | Supported at `/dashboard/` when running `db serve` |
| Demo | `db demo` — 30-second interactive terminal experience |

---

## The 4 Methods (Everything)

```python
from dimensionalbase import DimensionalBase

db = DimensionalBase()

# Write knowledge
db.put("task/auth/status", "JWT key expired", owner="agent-backend", confidence=0.92)

# Read knowledge (budget-aware, semantic)
result = db.get("task/**", budget=500, query="what is blocking deployment?")

# Watch for changes
sub = db.subscribe("task/**", subscriber="planner", callback=my_fn)

# Stop watching
db.unsubscribe(sub)
```

That is the entire agent-facing API. Everything else (algebra, trust, reasoning) is automatic.
