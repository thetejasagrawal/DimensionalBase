# DimensionalBase (DB)

> The protocol and database for AI communication. Storage and coordination built for how AI actually thinks — in dimensions, not text.

**Release status:** `v0.5.0` is a hardened alpha for the OSS core. The Python package, durable embedding index, secured REST server, and conformance-tested protocol surface are the supported focus. Tensor transport is still research-phase, and MCP should be treated as a trusted local transport rather than a hardened network surface.

[![PyPI](https://img.shields.io/pypi/v/dimensionalbase)](https://pypi.org/project/dimensionalbase/)
[![Python](https://img.shields.io/pypi/pyversions/dimensionalbase)](https://pypi.org/project/dimensionalbase/)
[![License](https://img.shields.io/github/license/thetejasagrawal/DimensionalBase)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/thetejasagrawal/DimensionalBase/ci.yml?label=tests)](https://github.com/thetejasagrawal/DimensionalBase/actions)

---

## Why

AI agents communicate through text — a format designed for humans. When Agent A understands something, it has a rich mathematical representation inside its model. To share it with Agent B, it crushes that into words. Agent B reconstructs a *different* representation. Information is destroyed every time.

DimensionalBase fixes this by making the **embedding the primary data** and text a human-readable shadow. It adds active reasoning, trust, confidence, and budget-aware context packing — everything multi-agent systems need to coordinate without losing information.

The current implementation uses a NumPy-backed `VectorStore` for semantic search and a SQLite-backed `embeddings` table for durable semantic state. It does not use FAISS.

## Install

```bash
pip install dimensionalbase
```

With command-line tools:
```bash
pip install dimensionalbase[cli]
```

With the secured REST server:
```bash
pip install dimensionalbase[server,security]
```

With MCP transport support (Python 3.10+):
```bash
pip install dimensionalbase[mcp]
```

With embeddings (recommended when you need semantic retrieval):
```bash
pip install dimensionalbase[embeddings-local]   # sentence-transformers, large local model stack
pip install dimensionalbase[embeddings-openai]   # OpenAI text-embedding-3-small
```

Embeddings are explicit in `v0.5`: plain `DimensionalBase()` starts in text-only mode unless you pass `embedding_provider=...`, `prefer_embedding="local"`, or `openai_api_key=...`.

## Quick Start

```python
from dimensionalbase import DimensionalBase

db = DimensionalBase()

# Write knowledge
db.put("task/auth/status", "JWT signing key expired", owner="backend-agent",
       type="fact", confidence=0.92, refs=["task/deploy-api"])

# Read relevant knowledge within a token budget
context = db.get("task/**", budget=500, query="What's blocking the deployment?")
print(context.text)

# Watch for changes
db.subscribe("task/**", "planner", lambda e: print(f"Event: {e.type.value}"))
```

## Architecture

```
Four methods:  put() / get() / subscribe() / unsubscribe()
Three channels: TEXT (always on) / EMBEDDING (when explicitly configured) / TENSOR (GPU, Phase 4)
Startup is deterministic: no implicit model bootstrap or network-backed embedding writes
```

Under the hood:

| Component | What It Does |
|---|---|
| **Dimensional Algebra** | compose, relate, analogy, interpolate, project — operations on the vector space |
| **Active Reasoning** | Automatic contradiction detection (LSH), gap detection, staleness alerts, auto-summarization |
| **Bayesian Confidence** | Beta(alpha, beta) distributions — confirmations raise confidence, contradictions lower it |
| **Agent Trust** | Elo + PageRank — the system learns which agents are reliable in which domains |
| **Provenance** | Full derivation DAG — trace any fact back to its origin |
| **Context Engine** | Budget-aware retrieval — 4-signal scoring (recency, confidence, similarity, ref distance) |
| **Semantic Compression** | Delta encoding + deduplication + 3-tier packing (full / compact / path-only) |

## API Reference

### Core Methods

| Method | Description |
|---|---|
| `db.put(path, value, owner, ...)` | Write knowledge to the shared space |
| `db.get(scope, budget, query, ...)` | Read relevant knowledge within a token budget |
| `db.subscribe(pattern, subscriber, callback)` | Watch for changes matching a pattern |
| `db.unsubscribe(subscription)` | Stop watching |

### Algebra Methods

| Method | Description |
|---|---|
| `db.encode(text)` | Project text into dimensional space |
| `db.relate(path_a, path_b)` | Discover the relationship between two entries |
| `db.compose(paths, mode)` | Merge multiple entries into a unified representation |
| `db.materialize(vector, k)` | Find the nearest entries to a vector |

### Convenience Methods

| Method | Description |
|---|---|
| `db.delete(path)` | Delete an entry |
| `db.exists(path)` | Check if an entry exists |
| `db.retrieve(path)` | Get a single entry by exact path |
| `db.status()` | Full system status |
| `db.agent_trust_report()` | Agent reliability scores |
| `db.lineage(path)` | Provenance chain for an entry |
| `db.knowledge_topology()` | Cluster and void analysis |

## Optional Dependencies

| Extra | Package | Purpose |
|---|---|---|
| `embeddings-local` | sentence-transformers | Local 384-dim embeddings (large model dependency set) |
| `embeddings-openai` | openai | OpenAI text-embedding-3-small |
| `mcp` | mcp | MCP server for Claude Code / Cursor (Python 3.10+) |
| `server` | fastapi, uvicorn | REST API server |
| `cli` | click, rich | Command-line interface |
| `langchain` | langchain-core | LangChain memory + tools |
| `crewai` | crewai-tools | CrewAI tool integration |
| `security` | cryptography | Auth, ACL, encryption |
| `bench` | openai, anthropic | Benchmark runners (reproducibility) |

## Current Guarantees

| Surface | Current guarantee |
|---|---|
| Embedded library | Supported, including durable semantic index reload for file-backed SQLite DBs |
| Secured REST server | Supported, and the packaged server/CLI entrypoints now bootstrap secure mode by default when an API key is configured |
| MCP | Supported for local/trusted use; not the hardened network surface |
| CLI | Supported for local inspection and operations |
| TTL cleanup | Explicit only via `clear_turn()` / `clear_session()` |
| Tensor channel | Placeholder only; not implemented |

## Examples

See [`examples/`](examples/) for:
- `basic_usage.py` — 4-method API walkthrough
- `multi_agent_demo.py` — 3-agent coordination with conflict detection
- `conflict_detection.py` — Contradiction detection in 15 lines

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## Packaging

The published distribution is expected to ship both `sdist` and wheel artifacts, including the dashboard assets under `dimensionalbase/server/static/`. Release validation uses `python -m build` and `python -m twine check dist/*`.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Security

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## License

MIT. See [LICENSE](LICENSE).
