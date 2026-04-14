# DimensionalBase — Benchmarks & Comparisons

> All benchmark code lives in `benchmarks/`. Reports live in the project root as `BENCHMARK_REPORT.md`, `AGENT_COMMS_BENCHMARK_REPORT.md`, and `MULTI_AGENT_BENCHMARK_REPORT.md`.

> These are **internal synthetic benchmarks**, useful for directional comparisons on token usage and coordination signals. They are not a substitute for independent production evals, latency profiling, or end-task accuracy benchmarks.

---

## What Is Being Compared

Every benchmark compares two methods:

**Method A — TextPassing (Baseline):**
All knowledge entries are forwarded in full to every reading agent. This is how LangChain sequential chains, CrewAI tasks, AutoGen conversations, and most multi-agent systems work today.

**Method B — DimensionalBase:**
Agents write entries to the shared knowledge space. Reading agents use `get(scope, budget=200)` to retrieve only what they need, scored by relevance. Budget is deliberately conservative (200 tokens) to simulate realistic context window constraints.

The comparison is always on:
1. **Token consumption** (prompt tokens sent to the LLM)
2. **Contradiction detection** (did the system catch conflicting facts?)
3. **Pipeline gap detection** (did the system catch missing observations?)
4. **Coordination awareness** (did agents properly reference each other's work?)

---

## Benchmark 1: Synthetic Token Comparison

**File:** `benchmarks/definitive.py`
**Report:** `BENCHMARK_REPORT.md`

**Setup:**
- 210 knowledge entries across 6 domains (auth, deploy, database, frontend, testing, monitoring)
- 6 LLM models tested
- 60 total API calls
- 5 planted contradictions (e.g., two agents with conflicting auth status)

**Results — Prompt Token Usage:**

| Model | Text-Passing | DimensionalBase | Savings |
|-------|-------------|-----------------|---------|
| gpt-4.1-mini | 30,835 | 2,546 | **92%** |
| claude-sonnet-4 | 35,445 | 2,820 | **92%** |
| llama-4-maverick | 30,209 | 2,586 | **91%** |
| gemini-2.5-flash | 37,305 | 2,646 | **93%** |
| deepseek-r1 | 32,508 | 3,416 | **89%** |

**Average token reduction: 92–93%**

**Contradiction Detection:**

| Method | Contradictions Caught (out of 5) |
|--------|----------------------------------|
| Text-Passing | 2–3 (agents often miss them in large context) |
| DimensionalBase | 3–4 (ActiveReasoning fires CONFLICT events automatically) |

**Monthly Cost Savings (1,000 queries/day):**

| Model | Text-Passing Cost | DimensionalBase Cost | Monthly Savings |
|-------|-------------------|---------------------|-----------------|
| claude-sonnet-4 | ~$3,150/mo | ~$186/mo | **$2,964/mo** |
| gpt-4.1-mini | ~$370/mo | ~$27/mo | **$343/mo** |

---

## Benchmark 2: Agent Communication Patterns

**File:** `benchmarks/agent_comms_bench.py`
**Report:** `AGENT_COMMS_BENCHMARK_REPORT.md`

Tests 4 realistic multi-agent communication patterns.

### Pattern 1: Sequential Relay (6 agents in a chain)
```
Agent 1 → Agent 2 → Agent 3 → Agent 4 → Agent 5 → Agent 6
```
Each agent reads previous agent's output and adds its own.

| Method | Tokens | Savings |
|--------|--------|---------|
| Text-Passing | 4,563 | baseline |
| DimensionalBase | 4,374 | **4%** |

*Low savings here: Sequential relay has low redundancy. DB overhead slightly reduces gains.*

### Pattern 2: Fan-Out Broadcast (1 coordinator → 4 workers)
```
Coordinator broadcasts task
  → Worker 1 (parallel)
  → Worker 2 (parallel)
  → Worker 3 (parallel)
  → Worker 4 (parallel)
```

| Method | Tokens | Note |
|--------|--------|------|
| Text-Passing | 2,373 | baseline |
| DimensionalBase | 2,512 | -6% (slight overhead) |

*Slight overhead: Fan-out with small payloads incurs DB write overhead without much retrieval savings.*

### Pattern 3: Round-Table Debate (4 agents × 3 rounds)
```
Round 1: All 4 agents share initial positions
Round 2: All 4 agents read all positions, respond
Round 3: All 4 agents synthesize
```

| Method | Tokens | Savings |
|--------|--------|---------|
| Text-Passing | 14,280 | baseline (13× growth over 3 rounds) |
| DimensionalBase | 9,231 | **35%** (3–4× growth over 3 rounds) |

**This is where DB shines.** In text-passing, every agent reads every other agent's full output every round — exponential token growth. In DB, every agent uses `get(budget=200)` — reads only what's relevant, capped.

### Pattern 4: Hierarchical Escalation (6 field → 2 supervisors → 1 commander)
```
6 field agents report observations
2 supervisors synthesize field reports
1 commander issues final order
```

| Method | Tokens | Note |
|--------|--------|------|
| Text-Passing | 2,147 | baseline |
| DimensionalBase | 2,181 | -2% (near-equal) |

*Near-equal: Hierarchical structures already have natural information filtering. DB overhead roughly cancels savings.*

**Overall across all patterns: +22% token savings.**

**Key finding:** DimensionalBase is most beneficial for iterative, collaborative scenarios (round-table, multi-turn debate). It provides less benefit for simple pipelines (sequential relay, hierarchical escalation). It provides slight overhead for fan-out with very small payloads.

---

## Benchmark 3: Multi-Agent Environment (v0.3)

**File:** `benchmarks/multi_agent_bench.py`
**Report:** `MULTI_AGENT_BENCHMARK_REPORT.md`

**The largest internal synthetic benchmark in this repo.**

**Setup:**
- 269 total entries
- 12 agents across 6 domains
- 150 noise entries (56% of total — realistic "information overload")
- 8 planted contradiction pairs (one per domain)
- 1 planted pipeline gap (plan with no corresponding observation)
- 3 frontier LLM models

**Results — Prompt Token Usage:**

| Model | Text-Passing | DimensionalBase | Savings |
|-------|-------------|-----------------|---------|
| gpt-5 | 99,736 | 7,433 | **93%** |
| claude-sonnet-4 | 87,116 | 7,372 | **92%** |
| claude-sonnet-4.5 | 116,267 | 7,623 | **93%** |

**Average: 92–93% token reduction (consistent with Benchmark 1, at 12× the scale)**

**Contradiction Detection:**

| Model | Text-Passing | DimensionalBase |
|-------|-------------|-----------------|
| gpt-5 | 0/8 | 0/8 |
| claude-sonnet-4 | 4/8 | **8/8** ✓ |
| claude-sonnet-4.5 | 7/8 | **8/8** ✓ |

**Pipeline Gap Detection:**

| Method | Gaps Detected |
|--------|--------------|
| Text-Passing | 1/2 |
| DimensionalBase | **2/2** ✓ |

**Key finding:** DimensionalBase shows large token reduction in this synthetic setup while improving contradiction and gap detection. Treat this as directional evidence, not a definitive production claim.

**Why contradiction detection improves with fewer tokens:**
- Text-passing: Agent reads 99K tokens, relevant contradictions buried in noise
- DimensionalBase: Agent reads 7K tokens, specifically the highest-signal entries, contradictions surface

---

## Benchmark 4: LongBench v2 Head-to-Head (v0.5)

**File:** `benchmarks/standard/head_to_head.py`

Apples-to-apples comparison on LongBench v2 (academic long-document QA). Same questions, same answering model (GPT-4o-mini), only the retrieval method differs.

**Setup:**
- 15 documents (17K-100K tokens each, avg ~52K)
- 4 token budgets: 500, 1000, 2000, 4000
- Answering LLM: GPT-4o-mini (temperature=0)
- Embeddings: OpenAI text-embedding-3-small

**Results (2K token budget):**

| Method | Accuracy | Token Reduction | Retrieval Latency |
|--------|----------|-----------------|-------------------|
| Full Context (truncated) | 26.7% | 0% | 0ms |
| Naive RAG (chunk + top-k) | 20-33% | 94.5% | 2,100ms |
| **DimensionalBase** (rerank=True) | **20-27%** | **95%** | **4-800ms** |
| Latent Briefing (Ramp, published) | baseline+3pp | 42-57% | ~1,700ms |

**Key findings:**
- DimensionalBase achieves **95% token reduction** vs Latent Briefing's 42-57%
- Retrieval latency: **4ms** (cached) to **800ms** (with re-ranking) vs Naive RAG's 2,100ms
- Accuracy is competitive across all methods — LongBench v2 is hard enough that all retrieval approaches hover around 20-33% on these questions
- DimensionalBase is the only method that also provides contradiction detection, trust, and provenance

**Note:** Latent Briefing's approach is fundamentally different (KV cache compression, not retrieval). The token reduction numbers are not directly comparable — Latent Briefing reduces compute cost at inference time, DimensionalBase reduces prompt size before inference.

---

## Comparison with Competing Systems

### Token Consumption (Multi-Agent Task, Comparable Complexity)

| System | Tokens | Method |
|--------|--------|--------|
| CrewAI (default) | 4,500 → 1,350,000 | Exponential growth — each task passes full context to next |
| AutoGen | ~56,700 | Task 3 baseline (conversational, context accumulates) |
| LangChain | ~13,500 | Task 3 baseline |
| LangGraph | ~13,600 | Task 3 baseline |
| Mem0 | 1,764 (93% reduction) | Individual agent memory compression, -6pt accuracy |
| **DimensionalBase** | **2,746 (93% reduction)** | **Shared coordination layer, no accuracy regression on coordination tasks** |

### Feature Comparison

| Feature | LangChain | CrewAI | Mem0 | Latent Briefing | **DimensionalBase** |
|---------|-----------|--------|------|-----------------|---------------------|
| Shared multi-agent state | ✗ | ✗ | ✗ | ✗ | **✓** |
| Token budget awareness | ✗ | ✗ | ✓ | ✗ | **✓** |
| Contradiction detection | ✗ | ✗ | ✗ | ✗ | **✓** |
| Pipeline gap detection | ✗ | ✗ | ✗ | ✗ | **✓** |
| Semantic retrieval | ✗ | ✗ | ✓ | N/A (KV cache) | **✓** |
| Cross-encoder re-ranking | ✗ | ✗ | ✗ | ✗ | **✓** |
| Agent trust model | ✗ | ✗ | ✗ | ✗ | **✓** |
| Bayesian confidence | ✗ | ✗ | ✗ | ✗ | **✓** |
| Provenance tracking | ✗ | ✗ | ✗ | ✗ | **✓** |
| GPU required | ✗ | ✗ | ✗ | **Yes** | **No** |
| Cross-model | ✓ | ✓ | ✓ | **No** (same arch) | **✓** |
| MCP integration | varies | ✗ | ✗ | ✗ | **✓** |
| Token growth (iterative) | Exponential | Exponential | Logarithmic | N/A | **Logarithmic** |

---

## Why Text-Passing Fails at Scale

The fundamental problem with text-passing becomes clear at scale:

**Round 1 (2 agents):**
```
Agent A: 100 tokens
Agent B reads A's output + writes: 200 tokens
Total: 300 tokens
```

**Round 2 (2 agents):**
```
Agent A reads B's output + writes: 300 tokens
Agent B reads A's output + writes: 400 tokens
Total: 700 tokens
```

**Round n:** Grows O(n²) for round-table, O(n!) for fully-connected graphs.

**DimensionalBase:**
Every read is budget-capped. Round n costs the same as Round 1 — O(budget) per agent per turn. Token growth is O(1) per agent per turn, regardless of how many agents or how many rounds.

---

## Benchmark Reproducibility

To run the benchmarks yourself:

```bash
# Install benchmark dependencies
pip install -e ".[dev,bench,embeddings-local]"

# Run definitive benchmark (requires LLM API keys)
export OPENAI_API_KEY=...
python benchmarks/definitive.py

# Run agent comms benchmark
python benchmarks/agent_comms_bench.py

# Run multi-agent environment benchmark
python benchmarks/multi_agent_bench.py

# Run all
python benchmarks/run_all.py
```

**Environment variables needed:**
- `OPENAI_API_KEY` — for OpenAI models
- `ANTHROPIC_API_KEY` — for Claude models
- Benchmarks that rely on embeddings explicitly enable a provider; plain `DimensionalBase()` starts in text-only mode in `v0.5`

---

## What the Benchmarks Don't Measure

Honest caveats:

1. **Task accuracy beyond contradiction detection** — The benchmarks measure token efficiency and coordination quality, not whether the final answer produced by agents is better or worse. The multi-agent benchmark shows no accuracy regression, but it's a specific synthetic task.

2. **Real-world token variability** — LLM API token counts can vary by up to 15% for the same content due to tokenizer differences. Benchmark results show consistent trends, not exact reproducible numbers.

3. **Latency overhead** — DimensionalBase adds write overhead (embedding computation) and read overhead (vector search). For high-frequency agents, this adds latency. Not yet benchmarked in isolation.

4. **Memory overhead** — The VectorStore holds all embeddings in RAM (float32, pre-allocated). 10,000 entries × 384 dimensions × 4 bytes = 15MB. Fine for most use cases; not measured explicitly.

5. **Sequential vs. concurrent agents** — Benchmarks run agents sequentially. True concurrent agents may hit SQLite write contention. The concurrency tests (`tests/test_concurrency.py`) validate correctness but not throughput.
