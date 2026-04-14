"""
DimensionalBase — Basic Usage

This is the simplest possible example. Four lines of meaningful code.
"""

from dimensionalbase import DimensionalBase

# Initialize — auto-detects channels and embedding providers
db = DimensionalBase()

# ============================================================
# WRITE: Agents share knowledge through the store
# ============================================================

db.put(
    path="task/auth/status",
    value="JWT signing key expired. Auth service returning 401 on all requests.",
    owner="agent-backend",
    type="fact",
    confidence=0.92,
    refs=["task/deploy-api"],
)

db.put(
    path="task/deploy-api",
    value="API deployment blocked. Waiting for auth fix.",
    owner="agent-devops",
    type="observation",
    confidence=0.85,
)

db.put(
    path="task/plan",
    value="1. Fix auth signing key, 2. Re-run integration tests, 3. Deploy API v2.1",
    owner="agent-planner",
    type="plan",
    confidence=0.90,
    refs=["task/auth/status", "task/deploy-api"],
)

# ============================================================
# READ: Get relevant context within a token budget
# ============================================================

context = db.get(
    scope="task/**",
    budget=500,
    query="What's blocking the deployment?",
)

print(f"Retrieved {len(context.entries)} entries using {context.tokens_used} tokens")
print(f"Channel: {context.channel_used.name}")
print()
print("Context for LLM:")
print(context.text)
print()

# ============================================================
# SUBSCRIBE: Watch for changes
# ============================================================

def on_change(event):
    print(f"EVENT: {event.type.value} at {event.path}")

sub = db.subscribe("task/**", "monitor-agent", on_change)

# This triggers the subscription
db.put(
    path="task/auth/status",
    value="JWT signing key rotated. Auth service healthy.",
    owner="agent-backend",
    type="fact",
    confidence=0.98,
)

# ============================================================
# INTROSPECT
# ============================================================

print()
print(f"Status: {db.status()}")
print(f"Tool definitions: {len(DimensionalBase.tool_definitions())} tools")

# ============================================================
# CLEANUP
# ============================================================

db.unsubscribe(sub)
db.close()
