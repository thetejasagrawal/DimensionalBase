# DimensionalBase v0.3 — Internal Synthetic Multi-Agent Environment Benchmark Report

**Date:** 2026-04-10 14:28 UTC

**Models:** gpt-5, gpt-5-mini, gpt-5-nano, claude-sonnet-4, claude-sonnet-4.5

**Environment:** 269 entries from 12 agents across 6 domains. 150 noise entries (56%), 8 contradiction pairs, 1 pipeline gap.

**Queries:** 8 per model — 5 domain reviewers + pipeline coordinator + conflict resolver + executive summary.

**Total API calls:** 80

This report is an internal synthetic benchmark focused on coordination signals and token usage. Treat the results as directional rather than definitive production evidence.

## Method A: TextPassing

Full context dump — all entries forwarded to every reader agent. This is how LangChain sequential chains, CrewAI tasks, AutoGen GroupChat, and most multi-agent frameworks work today.

## Method B: DimensionalBase

Budget-aware retrieval (200-250 tokens), scored by recency + confidence + semantic similarity + reference distance. Contradiction alerts and pipeline gap detection injected automatically.

## Token Efficiency

| Model | TP Prompt | DB Prompt | Prompt Savings | TP Total | DB Total | Total Savings |
|---|---|---|---|---|---|---|
| gpt-5 | 97,736 | 5,433 | **+94%** | 99,736 | 7,433 | **+93%** |
| gpt-5-mini | 97,736 | 5,433 | **+94%** | 99,736 | 7,433 | **+93%** |
| gpt-5-nano | 97,736 | 5,433 | **+94%** | 99,736 | 7,433 | **+93%** |
| claude-sonnet-4 | 85,849 | 5,966 | **+93%** | 87,116 | 7,372 | **+92%** |
| claude-sonnet-4.5 | 114,437 | 5,966 | **+95%** | 116,267 | 7,623 | **+93%** |

**Average prompt token reduction: +94%**

**Average total token reduction: +93%**

## Multi-Agent Coordination Quality

| Model | Contradictions (TP) | Contradictions (DB) | Coordination (TP) | Coordination (DB) | Pipeline Gaps (TP) | Pipeline Gaps (DB) |
|---|---|---|---|---|---|---|
| gpt-5 | 0/8 | 0/8 | 0/8 | 0/8 | 0/2 | 0/2 |
| gpt-5-mini | 0/8 | 0/8 | 0/8 | 0/8 | 0/2 | 0/2 |
| gpt-5-nano | 0/8 | 0/8 | 0/8 | 0/8 | 0/2 | 0/2 |
| claude-sonnet-4 | 4/8 | 8/8 | 4/8 | 7/8 | 1/2 | 2/2 |
| claude-sonnet-4.5 | 7/8 | 8/8 | 7/8 | 8/8 | 1/2 | 2/2 |

## Cost Projection (1,000 queries/day)

| Model | TP $/day | DB $/day | Savings/day | TP $/month | DB $/month | Savings/month |
|---|---|---|---|---|---|---|
| gpt-5 | $211.47 | $26.87 | **$184.61** | $6344.16 | $805.98 | **$5538.18** |
| gpt-5-mini | $42.29 | $5.37 | **$36.92** | $1268.83 | $161.20 | **$1107.64** |
| gpt-5-nano | $10.57 | $1.34 | **$9.23** | $317.21 | $40.30 | **$276.91** |
| claude-sonnet-4 | $276.55 | $38.99 | **$237.56** | $8296.56 | $1169.64 | **$7126.92** |
| claude-sonnet-4.5 | $370.76 | $42.75 | **$328.01** | $11122.83 | $1282.59 | **$9840.24** |

## Sample Responses: Pipeline Coordinator

### gpt-5

**TextPassing:**
> ...

**DimensionalBase:**
> ...

### gpt-5-mini

**TextPassing:**
> ...

**DimensionalBase:**
> ...

### gpt-5-nano

**TextPassing:**
> ...

**DimensionalBase:**
> ...

### claude-sonnet-4

**TextPassing:**
> **PIPELINE STATUS: CRITICAL BLOCKER DETECTED**

The deployment pipeline is missing the **APPROVE** step between stage and deploy. The sequence shows: plan ✓ → build ✓ → test ✓ → stage ✓ → **[MISSING APPROVE]** → deploy (in progress) → verify ✓. 

**CRITICAL BLOCKER:** Staging environment is returnin...

**DimensionalBase:**
> **PIPELINE STATUS: INCOMPLETE - MISSING APPROVAL STEP**

All pipeline steps are present EXCEPT the critical **approve** step between stage and deploy. The pipeline jumped directly from staging (with passing health checks and 0.02% canary error rate) to production deployment without explicit approval...

### claude-sonnet-4.5

**TextPassing:**
> # 🚨 DEPLOYMENT PIPELINE CRITICAL BLOCKER

**Pipeline Status:** ❌ **BLOCKED - APPROVE step missing**

**Required Pipeline:** plan → build → test → stage → **approve** → deploy → verify

**Present Steps:**
- ✅ plan (auth-service v4.2)
- ✅ build (image built, 3m41s)
- ✅ test (312/312 passing, 89% cover...

**DimensionalBase:**
> # Pipeline Coordination Analysis

**CRITICAL BLOCKER IDENTIFIED**: The pipeline has a **missing approval step** between stage and deploy. The sequence shows: plan → build → test → stage → [**APPROVE MISSING**] → deploy → verify. Auth-service v4.2 is already in production (3/8 pods updated) without d...

## Industry Comparison

| System | Tokens (multi-agent task) | Source |
|---|---|---|
| CrewAI | 4,500 → 1,350,000 (exponential) | AIMultiple 2026 |
| AutoGen | 56,700 (Task 3) | AIMultiple 2026 |
| LangChain | 13,500 (Task 3) | AIMultiple 2026 |
| LangGraph | 13,600 (Task 3) | AIMultiple 2026 |
| Mem0 | 1,764 (93% reduction, -6pt accuracy) | Mem0 Research |
| **DimensionalBase** | **7,458** (+93%, no accuracy loss) | This benchmark |

## Methodology

- All token counts from API `usage` field (OpenAI and Anthropic)
- Identical data and queries across all models and methods
- Temperature 0.2 for reproducibility
- Contradiction detection scored by keyword matching in responses
- Coordination awareness scored by agent reference detection
- Pipeline gap detection scored by missing-step keyword matching
- 269 entries: 96 signal + 150 noise + 16 contradiction + 7 pipeline
- DimensionalBase budget: 200-250 tokens per query
- GPT-5 models via OpenAI API direct
- Claude Sonnet models via Anthropic API
- Report generated: 2026-04-10 14:28:05 UTC
