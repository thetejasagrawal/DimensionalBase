# DimensionalBase v0.3 — Internal Synthetic Agent-to-Agent Communication Benchmark Report

**Date:** 2026-04-10 14:50 UTC

**Models:** gpt-5, gpt-5-mini, gpt-5-nano, claude-sonnet-4, claude-sonnet-4.5

**Total API calls:** 104

This report is an internal synthetic benchmark. It is useful for comparing communication patterns under controlled conditions, not for making definitive production claims.

## Communication Patterns Tested

- **Sequential Relay:** 6 agents in chain — each reads, analyzes, writes. Tests info fidelity & token growth.
- **Fan-Out Broadcast:** 1 coordinator → 4 workers → synthesis. Tests redundancy elimination.
- **Round-Table Debate:** 4 agents × 3 rounds. Tests token explosion vs flat budget.
- **Hierarchical Escalation:** 6 field → 2 supervisors → 1 commander. Tests prioritization & context quality.

## Results Summary

| Pattern | Avg TP Tokens | Avg DB Tokens | Savings |
|---|---|---|---|
| Sequential Relay | 4,563 | 4,374 | **+4%** |
| Fan-Out Broadcast | 2,373 | 2,512 | **-6%** |
| Round-Table Debate | 14,280 | 9,231 | **+35%** |
| Hierarchical Escalation | 2,147 | 2,181 | **-2%** |

**Overall token savings: +22%**

## Sequential Relay

6 agents in chain — each reads, analyzes, writes. Tests info fidelity & token growth.

| Model | TP Tokens | DB Tokens | Savings | TP Fidelity | DB Fidelity |
|---|---|---|---|---|---|
| claude-sonnet-4 | 4,477 | 4,851 | **-8%** | 67% | 56% |
| claude-sonnet-4.5 | 4,649 | 3,898 | **+16%** | 78% | 56% |

## Fan-Out Broadcast

1 coordinator → 4 workers → synthesis. Tests redundancy elimination.

| Model | TP Tokens | DB Tokens | Savings | TP Issues | DB Issues |
|---|---|---|---|---|---|
| claude-sonnet-4 | 2,268 | 2,413 | **-6%** | 6 | 4 |
| claude-sonnet-4.5 | 2,479 | 2,612 | **-5%** | 5 | 5 |

## Round-Table Debate

4 agents × 3 rounds. Tests token explosion vs flat budget.

| Model | TP Tokens | DB Tokens | Savings | TP Growth | DB Growth | TP Rebuttals | DB Rebuttals |
|---|---|---|---|---|---|---|
| claude-sonnet-4 | 12,222 | 8,175 | **+33%** | 10.1x | 3.9x | 10 | 10 |
| claude-sonnet-4.5 | 16,339 | 10,287 | **+37%** | 13.6x | 3.6x | 8 | 7 |

## Hierarchical Escalation

6 field → 2 supervisors → 1 commander. Tests prioritization & context quality.

| Model | TP Tokens | DB Tokens | Savings | TP Issues | DB Issues |
|---|---|---|---|---|---|
| claude-sonnet-4 | 2,160 | 2,000 | **+7%** | 2 | 1 |
| claude-sonnet-4.5 | 2,135 | 2,362 | **-11%** | 2 | 2 |

## Cost Projection (1,000 conversations/day)

| Model | TP $/day | DB $/day | Savings/day | TP $/month | DB $/month | Savings/month |
|---|---|---|---|---|---|---|
| gpt-5 | $66.49 | $59.83 | **$6.67** | $1994.82 | $1794.78 | **$200.04** |
| gpt-5-mini | $16.09 | $13.24 | **$2.85** | $482.80 | $397.24 | **$85.56** |
| gpt-5-nano | $0.00 | $0.00 | **$0.00** | $0.00 | $0.00 | **$0.00** |
| claude-sonnet-4 | $0.00 | $0.00 | **$0.00** | $0.00 | $0.00 | **$0.00** |
| claude-sonnet-4.5 | $0.00 | $0.00 | **$0.00** | $0.00 | $0.00 | **$0.00** |

## Methodology

- All token counts from API `usage` field (OpenAI and Anthropic)
- Identical scenarios and queries across all models and methods
- Temperature 0.2 for reproducibility
- Information fidelity scored by ground truth keyword matching
- Token growth measured as ratio of last-hop to first-hop prompt tokens
- GPT-5 models via OpenAI API direct
- Claude Sonnet models via Anthropic API
- Report generated: 2026-04-10 14:50:27 UTC
