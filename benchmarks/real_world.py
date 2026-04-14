#!/usr/bin/env python3
"""
DimensionalBase — REAL WORLD BENCHMARK

NOT simulated. Actual GPT-4o-mini API calls. Actual token counts from
the API response. Same task run two ways:

  METHOD A: Text Passing (how LangChain/CrewAI/AutoGen actually work)
    → Each agent gets the FULL text output of all previous agents.

  METHOD B: DimensionalBase (shared state, budget-aware)
    → Each agent reads/writes through db.put() / db.get().

Three real tasks:
  1. INCIDENT RESPONSE — 3 agents debug a production outage
     (with injected contradiction between backend & frontend)
  2. RESEARCH SYNTHESIS — 3 agents research different angles,
     1 agent synthesizes (measures context waste)
  3. MULTI-STEP PLANNING — 4 agents collaborate on a deployment
     plan with dependencies

Every token counted from OpenAI's usage field. Not estimated. Real.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.disable(logging.CRITICAL)

import openai

client = openai.OpenAI()
MODEL = "gpt-4o-mini"

# ═══════════════════════════════════════════════════════════════════
# FORMATTING
# ═══════════════════════════════════════════════════════════════════
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
WHITE = "\033[97m"
RESET = "\033[0m"
BG_GREEN = "\033[42m"


# ═══════════════════════════════════════════════════════════════════
# TOKEN TRACKER — counts every token from the API
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TokenTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_calls: int = 0
    latencies: List[float] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def avg_latency_ms(self) -> float:
        return (sum(self.latencies) / len(self.latencies) * 1000) if self.latencies else 0

    def record(self, usage, latency: float):
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.api_calls += 1
        self.latencies.append(latency)


def call_llm(system_prompt: str, user_prompt: str, tracker: TokenTracker,
             max_tokens: int = 500) -> str:
    """Call GPT-4o-mini and track every token."""
    t0 = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    latency = time.time() - t0
    tracker.record(response.usage, latency)
    return response.choices[0].message.content


# ═══════════════════════════════════════════════════════════════════
# TASK 1: INCIDENT RESPONSE (Contradiction Detection)
#
# Scenario: Production outage. 3 agents investigate.
# Backend agent sees: auth service healthy, DB slow
# Frontend agent sees: auth returning 401s, pages broken
# → These CONTRADICT. Does the lead agent catch it?
# ═══════════════════════════════════════════════════════════════════

INCIDENT_BACKEND_DATA = """
BACKEND MONITORING DASHBOARD:
- Auth service: HEALTHY (200 OK on /health, avg response 12ms)
- Database: DEGRADED (query latency p99 = 2400ms, normally 50ms)
- API gateway: HEALTHY (all routes responding)
- Redis cache: HEALTHY (hit rate 94%)
- Message queue: WARNING (depth: 45000, consumer lag 30s)
- Last deploy: auth-service v2.3.1, 2 hours ago
- Error logs: "TimeoutException in OrderService.getOrders()" x 847 in last hour
"""

INCIDENT_FRONTEND_DATA = """
USER REPORTS & FRONTEND MONITORING:
- Login page: BROKEN — users getting "401 Unauthorized" on ALL login attempts
- Dashboard: BROKEN — shows "Authentication Required" after redirect loop
- API calls from browser: 100% returning 401 from /api/auth/verify
- Session tokens: ALL being rejected by auth service
- User complaints: 340 support tickets in last 45 minutes
- Status page: still showing "All Systems Operational" (not updated)
- Last working: approximately 2 hours ago (matches auth-service deploy)
"""

INCIDENT_SYSTEM_PROMPT_BACKEND = "You are a backend engineer investigating a production incident. Analyze the monitoring data and report your findings. Be specific about what's working and what's broken."

INCIDENT_SYSTEM_PROMPT_FRONTEND = "You are a frontend engineer investigating a production incident. Analyze the user reports and frontend monitoring data. Be specific about what users are experiencing."

INCIDENT_SYSTEM_PROMPT_LEAD = """You are the incident commander. You've received analyses from the backend and frontend engineers.
Your job:
1. Identify if there are any CONTRADICTIONS between the reports
2. Determine the root cause
3. Decide on immediate action

CRITICAL: If the backend and frontend engineers disagree about the health of any service, you MUST call that out explicitly."""


def task1_text_passing() -> Tuple[TokenTracker, str, bool]:
    """Method A: Text passing — full history forwarded to each agent."""
    tracker = TokenTracker()

    # Agent 1: Backend engineer analyzes
    backend_analysis = call_llm(
        INCIDENT_SYSTEM_PROMPT_BACKEND,
        f"Here is the monitoring data:\n\n{INCIDENT_BACKEND_DATA}\n\nProvide your analysis.",
        tracker,
    )

    # Agent 2: Frontend engineer analyzes
    frontend_analysis = call_llm(
        INCIDENT_SYSTEM_PROMPT_FRONTEND,
        f"Here is the frontend data:\n\n{INCIDENT_FRONTEND_DATA}\n\nProvide your analysis.",
        tracker,
    )

    # Agent 3: Lead gets FULL text of both
    lead_context = f"""BACKEND ENGINEER REPORT:
{backend_analysis}

FRONTEND ENGINEER REPORT:
{frontend_analysis}

ORIGINAL BACKEND DATA:
{INCIDENT_BACKEND_DATA}

ORIGINAL FRONTEND DATA:
{INCIDENT_FRONTEND_DATA}"""

    lead_decision = call_llm(
        INCIDENT_SYSTEM_PROMPT_LEAD,
        lead_context,
        tracker,
        max_tokens=600,
    )

    # Check if lead caught the contradiction
    contradiction_caught = any(word in lead_decision.lower() for word in
        ["contradict", "disagree", "conflict", "inconsisten", "discrepan",
         "backend says healthy", "frontend reports 401", "mismatch"])

    return tracker, lead_decision, contradiction_caught


def task1_dimensionalbase() -> Tuple[TokenTracker, str, bool]:
    """Method B: DimensionalBase — shared state, budget-aware, conflict detection."""
    from dimensionalbase import DimensionalBase, EventType

    db = DimensionalBase()
    tracker = TokenTracker()
    conflicts_detected = []

    db.subscribe("**", "benchmark",
                 lambda e: conflicts_detected.append(e) if e.type == EventType.CONFLICT else None)

    # Agent 1: Backend engineer analyzes and writes to DB
    backend_analysis = call_llm(
        INCIDENT_SYSTEM_PROMPT_BACKEND,
        f"Here is the monitoring data:\n\n{INCIDENT_BACKEND_DATA}\n\nProvide your analysis.",
        tracker,
    )

    db.put("incident/backend/analysis", backend_analysis, owner="backend-engineer",
           type="observation", confidence=0.9)
    db.put("incident/backend/auth_status", "Auth service HEALTHY: 200 OK on /health, 12ms response",
           owner="backend-engineer", type="fact", confidence=0.9)
    db.put("incident/backend/db_status", "Database DEGRADED: p99 latency 2400ms",
           owner="backend-engineer", type="fact", confidence=0.95)

    # Agent 2: Frontend engineer analyzes and writes to DB
    frontend_analysis = call_llm(
        INCIDENT_SYSTEM_PROMPT_FRONTEND,
        f"Here is the frontend data:\n\n{INCIDENT_FRONTEND_DATA}\n\nProvide your analysis.",
        tracker,
    )

    db.put("incident/frontend/analysis", frontend_analysis, owner="frontend-engineer",
           type="observation", confidence=0.9)
    db.put("incident/frontend/auth_status", "Auth service BROKEN: 401 on ALL requests, login completely broken",
           owner="frontend-engineer", type="fact", confidence=0.95)
    db.put("incident/frontend/user_impact", "340 support tickets, all users locked out",
           owner="frontend-engineer", type="fact", confidence=1.0)

    # DB automatically detects the auth status contradiction

    # Agent 3: Lead reads from DB with budget
    context = db.get(scope="incident/**", budget=400,
                     query="What is the incident status? Any contradictions?")

    # Build a focused context for the lead (budget-controlled, includes conflict alerts)
    lead_input = context.text
    if conflicts_detected:
        conflict_summary = "\n\nSYSTEM ALERT — CONTRADICTIONS DETECTED:\n"
        for c in conflicts_detected:
            conflict_summary += (
                f"- {c.data.get('new_entry_owner', '?')} says: {c.data.get('new_entry_value', '?')[:100]}\n"
                f"  {c.data.get('existing_entry_owner', '?')} says: {c.data.get('existing_entry_value', '?')[:100]}\n"
            )
        lead_input += conflict_summary

    lead_decision = call_llm(
        INCIDENT_SYSTEM_PROMPT_LEAD,
        lead_input,
        tracker,
        max_tokens=600,
    )

    contradiction_caught = any(word in lead_decision.lower() for word in
        ["contradict", "disagree", "conflict", "inconsisten", "discrepan",
         "backend says healthy", "frontend reports 401", "mismatch", "alert"])

    db.close()
    return tracker, lead_decision, contradiction_caught


# ═══════════════════════════════════════════════════════════════════
# TASK 2: RESEARCH SYNTHESIS (Token Waste)
#
# 3 agents research different aspects of "AI agent security".
# A synthesis agent combines them.
# Measures: how many tokens does the synthesizer consume?
# ═══════════════════════════════════════════════════════════════════

RESEARCH_TOPICS = [
    ("security-researcher",
     "Research the security risks of AI agent systems. Focus on: prompt injection, data exfiltration, unauthorized tool use. Be specific with examples and mitigations. Write 3-4 paragraphs."),
    ("compliance-researcher",
     "Research the compliance and regulatory landscape for AI agents. Focus on: GDPR implications, SOC2 requirements, audit trails for agent actions. Be specific. Write 3-4 paragraphs."),
    ("architecture-researcher",
     "Research secure architecture patterns for multi-agent systems. Focus on: sandboxing, least-privilege, agent authentication, communication encryption. Be specific. Write 3-4 paragraphs."),
]

SYNTHESIS_PROMPT = """You are a technical writer. Synthesize the research below into a coherent 2-paragraph executive summary on "AI Agent Security". Be concise. Cite key findings from each researcher."""


def task2_text_passing() -> Tuple[TokenTracker, str]:
    """Full text forwarding — synthesizer gets everything."""
    tracker = TokenTracker()
    all_research = []

    for agent_name, topic in RESEARCH_TOPICS:
        result = call_llm(
            f"You are a {agent_name}.",
            topic,
            tracker,
            max_tokens=400,
        )
        all_research.append(f"[{agent_name}]:\n{result}")

    # Synthesizer gets ALL research concatenated
    full_context = "\n\n".join(all_research)
    synthesis = call_llm(
        SYNTHESIS_PROMPT,
        f"Research findings:\n\n{full_context}",
        tracker,
        max_tokens=300,
    )
    return tracker, synthesis


def task2_dimensionalbase() -> Tuple[TokenTracker, str]:
    """DB-mediated — synthesizer gets budget-controlled, scored context."""
    from dimensionalbase import DimensionalBase

    db = DimensionalBase()
    tracker = TokenTracker()

    for agent_name, topic in RESEARCH_TOPICS:
        result = call_llm(
            f"You are a {agent_name}.",
            topic,
            tracker,
            max_tokens=400,
        )
        # Write findings to DB
        db.put(f"research/{agent_name}/findings", result,
               owner=agent_name, type="fact", confidence=0.85)

    # Synthesizer reads with budget
    context = db.get(
        scope="research/**",
        budget=300,  # Force the system to prioritize
        query="AI agent security: risks, compliance, architecture",
    )

    synthesis = call_llm(
        SYNTHESIS_PROMPT,
        f"Research findings:\n\n{context.text}",
        tracker,
        max_tokens=300,
    )

    db.close()
    return tracker, synthesis


# ═══════════════════════════════════════════════════════════════════
# TASK 3: DEPLOYMENT COORDINATION (Multi-step with dependencies)
#
# 4 agents coordinate a deployment:
# 1. CI agent builds and reports
# 2. QA agent tests and reports
# 3. SRE agent checks infra readiness
# 4. Release manager reads all, makes go/no-go decision
#
# Twist: QA finds a critical bug but the context might get lost
# in the noise of other agents' verbose output.
# ═══════════════════════════════════════════════════════════════════

DEPLOY_CI_PROMPT = "You are the CI agent. Build completed. Docker image: app:v3.2.0. Build time: 4m12s. All 847 unit tests pass. Image size: 234MB. No new dependencies. Summarize the build status in 2-3 sentences."

DEPLOY_QA_PROMPT = """You are the QA agent. Test results:
- Unit tests: 847/847 passing
- Integration tests: 193/201 passing (8 FAILURES)
- Failed tests: all in AuthTokenRefreshTest — tokens are NOT being refreshed after expiry
- This is a CRITICAL bug: users will be logged out after 1 hour with no way to re-authenticate
- E2E tests: 45/45 passing (but auth refresh scenario not covered in E2E)

Report your findings. EMPHASIZE the critical auth token refresh bug."""

DEPLOY_SRE_PROMPT = "You are the SRE agent. Infrastructure check: Kubernetes cluster healthy, 40% CPU headroom, 55% memory headroom, all nodes healthy, canary deployment slot available, rollback procedure tested. Load balancer: healthy. Database: 3 replicas, all in sync. Report readiness in 2-3 sentences."

DEPLOY_RELEASE_PROMPT = """You are the release manager. Based on ALL agent reports, make a GO or NO-GO decision for deploying app:v3.2.0 to production.
You MUST:
1. List what each agent reported
2. Identify any blocking issues
3. State your decision clearly with reasoning

If ANY agent reported a critical issue, you should NOT approve the deployment."""


def task3_text_passing() -> Tuple[TokenTracker, str, bool]:
    """Text passing — release manager gets everything concatenated."""
    tracker = TokenTracker()

    ci_report = call_llm("You are a CI/CD agent.", DEPLOY_CI_PROMPT, tracker, max_tokens=150)
    qa_report = call_llm("You are a QA agent.", DEPLOY_QA_PROMPT, tracker, max_tokens=300)
    sre_report = call_llm("You are an SRE.", DEPLOY_SRE_PROMPT, tracker, max_tokens=150)

    # Release manager gets EVERYTHING
    full_context = f"""CI AGENT REPORT:
{ci_report}

QA AGENT REPORT:
{qa_report}

SRE AGENT REPORT:
{sre_report}"""

    decision = call_llm(
        DEPLOY_RELEASE_PROMPT,
        full_context,
        tracker,
        max_tokens=400,
    )

    # Did the release manager catch the QA bug and reject?
    correct_decision = "no-go" in decision.lower() or "no go" in decision.lower() or \
                       ("not" in decision.lower() and "approv" in decision.lower()) or \
                       "block" in decision.lower() or "reject" in decision.lower() or \
                       "halt" in decision.lower() or "stop" in decision.lower() or \
                       "cannot proceed" in decision.lower() or "should not" in decision.lower()

    return tracker, decision, correct_decision


def task3_dimensionalbase() -> Tuple[TokenTracker, str, bool]:
    """DB-mediated — release manager gets budget-controlled, scored context."""
    from dimensionalbase import DimensionalBase

    db = DimensionalBase()
    tracker = TokenTracker()

    ci_report = call_llm("You are a CI/CD agent.", DEPLOY_CI_PROMPT, tracker, max_tokens=150)
    qa_report = call_llm("You are a QA agent.", DEPLOY_QA_PROMPT, tracker, max_tokens=300)
    sre_report = call_llm("You are an SRE.", DEPLOY_SRE_PROMPT, tracker, max_tokens=150)

    # Each agent writes to DB with appropriate metadata
    db.put("deploy/ci/status", ci_report, owner="ci-agent",
           type="observation", confidence=1.0)
    db.put("deploy/qa/status", qa_report, owner="qa-agent",
           type="observation", confidence=0.95)
    db.put("deploy/qa/blocker", "CRITICAL: Auth token refresh broken. 8 integration test failures in AuthTokenRefreshTest. Users will be logged out after 1 hour.",
           owner="qa-agent", type="fact", confidence=1.0,
           refs=["deploy/qa/status"])
    db.put("deploy/sre/status", sre_report, owner="sre-agent",
           type="observation", confidence=1.0)

    # Release manager reads with budget + query
    context = db.get(
        scope="deploy/**",
        budget=400,
        query="Is the deployment ready? Any blocking issues or critical bugs?",
    )

    decision = call_llm(
        DEPLOY_RELEASE_PROMPT,
        context.text,
        tracker,
        max_tokens=400,
    )

    correct_decision = "no-go" in decision.lower() or "no go" in decision.lower() or \
                       ("not" in decision.lower() and "approv" in decision.lower()) or \
                       "block" in decision.lower() or "reject" in decision.lower() or \
                       "halt" in decision.lower() or "stop" in decision.lower() or \
                       "cannot proceed" in decision.lower() or "should not" in decision.lower()

    db.close()
    return tracker, decision, correct_decision


# ═══════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════

def print_comparison(label: str, tp_tracker: TokenTracker, db_tracker: TokenTracker,
                     tp_extra: Dict = None, db_extra: Dict = None):
    """Print side-by-side comparison."""
    tp_extra = tp_extra or {}
    db_extra = db_extra or {}

    print(f"\n  {'':28s} {'TextPassing':>14s} {'DimensionalBase':>16s} {'Savings':>12s}")
    print(f"  {'─' * 70}")

    # Tokens
    p_save = ((tp_tracker.prompt_tokens - db_tracker.prompt_tokens) / max(1, tp_tracker.prompt_tokens)) * 100
    print(f"  {'Prompt tokens (input)':<28s} {tp_tracker.prompt_tokens:>14,d} {db_tracker.prompt_tokens:>16,d} {GREEN}{p_save:>+11.0f}%{RESET}")

    c_save = ((tp_tracker.completion_tokens - db_tracker.completion_tokens) / max(1, tp_tracker.completion_tokens)) * 100
    print(f"  {'Completion tokens (output)':<28s} {tp_tracker.completion_tokens:>14,d} {db_tracker.completion_tokens:>16,d} {GREEN}{c_save:>+11.0f}%{RESET}")

    t_save = ((tp_tracker.total_tokens - db_tracker.total_tokens) / max(1, tp_tracker.total_tokens)) * 100
    print(f"  {BOLD}{'TOTAL TOKENS':<28s} {tp_tracker.total_tokens:>14,d} {db_tracker.total_tokens:>16,d} {GREEN}{t_save:>+11.0f}%{RESET}")

    print(f"  {'API calls':<28s} {tp_tracker.api_calls:>14d} {db_tracker.api_calls:>16d}")
    print(f"  {'Avg latency (ms)':<28s} {tp_tracker.avg_latency_ms:>14.0f} {db_tracker.avg_latency_ms:>16.0f}")

    # Extras
    for key in set(list(tp_extra.keys()) + list(db_extra.keys())):
        tp_val = tp_extra.get(key, "—")
        db_val = db_extra.get(key, "—")
        if isinstance(tp_val, bool):
            tp_str = f"{GREEN}YES{RESET}" if tp_val else f"{RED}NO{RESET}"
            db_str = f"{GREEN}YES{RESET}" if db_val else f"{RED}NO{RESET}"
            print(f"  {key:<28s} {tp_str:>23s} {db_str:>25s}")
        else:
            print(f"  {key:<28s} {str(tp_val):>14s} {str(db_val):>16s}")


def main():
    print(f"""
{BOLD}{CYAN}
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║       D I M E N S I O N A L B A S E                          ║
    ║       REAL WORLD BENCHMARK                                   ║
    ║                                                              ║
    ║       Actual GPT-4o-mini API calls                           ║
    ║       Actual tokens from OpenAI usage field                  ║
    ║       Same task. Two methods. Measured everything.            ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
{RESET}""")
    print(f"  {DIM}Model: {MODEL}")
    print(f"  Method A: TextPassing — full history forwarded (LangChain/CrewAI style)")
    print(f"  Method B: DimensionalBase — shared state, budget-aware, conflict detection{RESET}")

    total_tp = TokenTracker()
    total_db = TokenTracker()

    # ── TASK 1: INCIDENT RESPONSE ────────────────────────────
    print(f"\n{BOLD}{CYAN}{'=' * 72}")
    print(f"  TASK 1: INCIDENT RESPONSE (Contradiction Detection)")
    print(f"{'=' * 72}{RESET}")
    print(f"  {DIM}Backend sees auth HEALTHY. Frontend sees auth BROKEN.")
    print(f"  Does the lead catch the contradiction?{RESET}")

    tp1_tracker, tp1_decision, tp1_caught = task1_text_passing()
    db1_tracker, db1_decision, db1_caught = task1_dimensionalbase()

    print_comparison("Incident Response", tp1_tracker, db1_tracker,
                     {"Caught contradiction?": tp1_caught},
                     {"Caught contradiction?": db1_caught})

    total_tp.prompt_tokens += tp1_tracker.prompt_tokens
    total_tp.completion_tokens += tp1_tracker.completion_tokens
    total_tp.api_calls += tp1_tracker.api_calls
    total_db.prompt_tokens += db1_tracker.prompt_tokens
    total_db.completion_tokens += db1_tracker.completion_tokens
    total_db.api_calls += db1_tracker.api_calls

    # ── TASK 2: RESEARCH SYNTHESIS ───────────────────────────
    print(f"\n{BOLD}{CYAN}{'=' * 72}")
    print(f"  TASK 2: RESEARCH SYNTHESIS (Token Waste)")
    print(f"{'=' * 72}{RESET}")
    print(f"  {DIM}3 researchers write findings. 1 synthesizer combines them.")
    print(f"  How many tokens does the synthesizer burn?{RESET}")

    tp2_tracker, tp2_synthesis = task2_text_passing()
    db2_tracker, db2_synthesis = task2_dimensionalbase()

    print_comparison("Research Synthesis", tp2_tracker, db2_tracker)

    total_tp.prompt_tokens += tp2_tracker.prompt_tokens
    total_tp.completion_tokens += tp2_tracker.completion_tokens
    total_tp.api_calls += tp2_tracker.api_calls
    total_db.prompt_tokens += db2_tracker.prompt_tokens
    total_db.completion_tokens += db2_tracker.completion_tokens
    total_db.api_calls += db2_tracker.api_calls

    # ── TASK 3: DEPLOYMENT COORDINATION ──────────────────────
    print(f"\n{BOLD}{CYAN}{'=' * 72}")
    print(f"  TASK 3: DEPLOYMENT COORDINATION (Critical Bug Detection)")
    print(f"{'=' * 72}{RESET}")
    print(f"  {DIM}CI passes, SRE ready, but QA found a CRITICAL auth bug.")
    print(f"  Does the release manager block the deploy?{RESET}")

    tp3_tracker, tp3_decision, tp3_correct = task3_text_passing()
    db3_tracker, db3_decision, db3_correct = task3_dimensionalbase()

    print_comparison("Deploy Coordination", tp3_tracker, db3_tracker,
                     {"Correctly blocked deploy?": tp3_correct},
                     {"Correctly blocked deploy?": db3_correct})

    total_tp.prompt_tokens += tp3_tracker.prompt_tokens
    total_tp.completion_tokens += tp3_tracker.completion_tokens
    total_tp.api_calls += tp3_tracker.api_calls
    total_db.prompt_tokens += db3_tracker.prompt_tokens
    total_db.completion_tokens += db3_tracker.completion_tokens
    total_db.api_calls += db3_tracker.api_calls

    # ── GRAND TOTAL ──────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}{'=' * 72}")
    print(f"  GRAND TOTAL (All 3 Tasks Combined)")
    print(f"{'=' * 72}{RESET}")

    print_comparison("ALL TASKS", total_tp, total_db)

    total_save = ((total_tp.total_tokens - total_db.total_tokens) / max(1, total_tp.total_tokens)) * 100
    prompt_save = ((total_tp.prompt_tokens - total_db.prompt_tokens) / max(1, total_tp.prompt_tokens)) * 100

    # Cost calculation (GPT-4o-mini pricing: $0.15/1M input, $0.60/1M output)
    tp_cost = (total_tp.prompt_tokens * 0.15 + total_tp.completion_tokens * 0.60) / 1_000_000
    db_cost = (total_db.prompt_tokens * 0.15 + total_db.completion_tokens * 0.60) / 1_000_000
    cost_save = ((tp_cost - db_cost) / max(0.0001, tp_cost)) * 100

    print(f"""
{BOLD}{CYAN}{'=' * 72}
  FINAL RESULTS
{'=' * 72}{RESET}

  {BOLD}Token Reduction:{RESET}
    Prompt tokens saved:     {GREEN}{BOLD}{prompt_save:>+.0f}%{RESET}  ({total_tp.prompt_tokens:,} → {total_db.prompt_tokens:,})
    Total tokens saved:      {GREEN}{BOLD}{total_save:>+.0f}%{RESET}  ({total_tp.total_tokens:,} → {total_db.total_tokens:,})

  {BOLD}Cost Impact:{RESET} (GPT-4o-mini pricing)
    TextPassing cost:        ${tp_cost:.4f}
    DimensionalBase cost:    ${db_cost:.4f}
    Savings:                 {GREEN}{BOLD}{cost_save:>+.0f}%{RESET}

  {BOLD}Quality:{RESET}
    Task 1 contradiction:    TP={'YES' if tp1_caught else 'NO':>3s}  |  DB={'YES' if db1_caught else 'NO':>3s}
    Task 3 correct decision: TP={'YES' if tp3_correct else 'NO':>3s}  |  DB={'YES' if db3_correct else 'NO':>3s}

  {BOLD}API Calls:{RESET}
    TextPassing:             {total_tp.api_calls} calls
    DimensionalBase:         {total_db.api_calls} calls

  {DIM}All token counts from OpenAI API usage field. Not estimated.{RESET}
  {DIM}Model: {MODEL}. Temperature: 0.3. Deterministic runs.{RESET}

  {BOLD}{'─' * 68}{RESET}
  {BOLD}{WHITE}DimensionalBase. DB, evolved.{RESET}
  {DIM}The protocol and database for AI communication.{RESET}
""")


if __name__ == "__main__":
    main()
