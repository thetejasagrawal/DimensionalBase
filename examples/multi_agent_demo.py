"""
DimensionalBase — Multi-Agent Coordination Demo

Three agents working on a deployment task:
  - Planner: creates and updates the plan
  - Backend: investigates and fixes issues
  - Monitor: watches for conflicts and gaps

DimensionalBase is the fourth participant — it detects contradictions,
flags gaps, and keeps everyone coherent.
"""

from dimensionalbase import DimensionalBase, EventType

# ============================================================
# SETUP
# ============================================================

db = DimensionalBase()

# Monitor agent watches everything
alert_log = []

def on_alert(event):
    if event.type in (EventType.CONFLICT, EventType.GAP, EventType.STALE):
        alert_log.append(event)
        print(f"  ALERT [{event.type.value.upper()}]: {event.path}")
        if event.type == EventType.CONFLICT:
            print(f"    {event.data.get('new_entry_owner')} says: {event.data.get('new_entry_value', '')[:80]}")
            print(f"    {event.data.get('existing_entry_owner')} says: {event.data.get('existing_entry_value', '')[:80]}")

db.subscribe("**", "monitor-agent", on_alert)

print("=" * 60)
print("MULTI-AGENT DEPLOYMENT DEMO")
print("=" * 60)

# ============================================================
# PHASE 1: PLANNING
# ============================================================

print("\n--- Phase 1: Planning ---")

db.put(
    path="deploy/plan",
    value="Deploy auth-service v2.3: build -> test -> stage -> prod",
    owner="planner",
    type="plan",
    confidence=0.95,
    refs=["deploy/build", "deploy/test", "deploy/stage", "deploy/prod"],
)
print("Planner: Created deployment plan with 4 steps")

# Planner reads to verify
plan = db.get(scope="deploy/plan", budget=200)
print(f"Planner: Verified plan ({plan.tokens_used} tokens)")

# ============================================================
# PHASE 2: EXECUTION
# ============================================================

print("\n--- Phase 2: Execution ---")

# Backend builds
db.put(
    path="deploy/build",
    value="Docker image built: auth-service:v2.3.1-sha.abc123",
    owner="backend",
    type="observation",
    confidence=1.0,
)
print("Backend: Build complete")

# Backend runs tests
db.put(
    path="deploy/test",
    value="Tests passing: 142/142. Coverage: 87%.",
    owner="backend",
    type="observation",
    confidence=1.0,
)
print("Backend: Tests passing")

# Backend deploys to staging
db.put(
    path="deploy/stage",
    value="Deployed to staging. Health check passing.",
    owner="backend",
    type="observation",
    confidence=0.95,
)
print("Backend: Staged")

# ============================================================
# PHASE 3: CONFLICT
# ============================================================

print("\n--- Phase 3: Conflict Detection ---")

# Backend says staging is healthy
db.put(
    path="deploy/stage/health",
    value="Staging environment healthy. Latency: 45ms p99.",
    owner="backend",
    type="fact",
    confidence=0.92,
)

# But QA says staging is broken
db.put(
    path="deploy/stage/status",
    value="Staging environment returning 500 errors on auth endpoints.",
    owner="qa-agent",
    type="fact",
    confidence=0.88,
)

print(f"\nAlerts fired: {len(alert_log)}")

# ============================================================
# PHASE 4: RESOLUTION
# ============================================================

print("\n--- Phase 4: Resolution ---")

# Read all deployment context
full_context = db.get(
    scope="deploy/**",
    budget=2000,
    query="What is the current deployment status? Any issues?",
)
print(f"Full context: {full_context.total_matched} entries, {full_context.tokens_used} tokens")
print()

# Print context that would go to an LLM
print("CONTEXT FOR LLM:")
print("-" * 40)
print(full_context.text)
print("-" * 40)

# Planner makes a decision based on the full context
db.put(
    path="deploy/decision",
    value="Hold prod deployment. Investigate staging 500 errors on auth endpoints. Backend and QA agents disagree on staging health.",
    owner="planner",
    type="decision",
    confidence=0.90,
    refs=["deploy/stage/health", "deploy/stage/status"],
)
print("\nPlanner: Decision made — hold deployment pending investigation")

# ============================================================
# STATUS
# ============================================================

print("\n--- Final Status ---")
status = db.status()
print(f"Entries: {status['entries']}")
print(f"Channel: {status['channel']}")
print(f"Embeddings: {status['embeddings']}")
print(f"Active alerts: {len(alert_log)}")

print("\n--- Event History ---")
for event in db.events.get_history(limit=10):
    print(f"  {event.type.value:10s} | {event.path}")

db.close()
print("\nDone.")
