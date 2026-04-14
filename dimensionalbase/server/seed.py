"""Seed a DimensionalBase instance with realistic demo data for the web dashboard."""

import time


def seed_demo_data(db) -> None:
    """Populate *db* with 14 entries across deploy/, auth/, and monitoring/
    prefixes.  Includes two planted contradictions and one gap trigger
    (a plan that references a path with no observations).
    """

    _put = db.put
    pause = 0.05

    # ── deploy/ prefix ──────────────────────────────────────────

    _put("deploy/target-env", "Production cluster us-east-1 (EKS v1.29)",
         owner="planner-agent", type="fact", confidence=1.0)
    time.sleep(pause)

    _put("deploy/strategy", "Blue-green deployment with 10 % canary window",
         owner="planner-agent", type="decision", confidence=0.95)
    time.sleep(pause)

    _put("deploy/image-tag", "api-server:sha-a3f82c1 built from main@HEAD",
         owner="backend-agent", type="fact", confidence=1.0)
    time.sleep(pause)

    _put("deploy/rollout-steps",
         "1. Push image  2. Canary 10 %  3. Monitor 5 min  4. Full rollout",
         owner="planner-agent", type="plan", confidence=0.9,
         refs=["deploy/image-tag", "deploy/strategy", "monitoring/error-rate"])
    time.sleep(pause)

    _put("deploy/canary-status", "Canary pods healthy; 0 restarts after 3 min",
         owner="qa-agent", type="observation", confidence=0.85)
    time.sleep(pause)

    # ── auth/ prefix ────────────────────────────────────────────

    _put("auth/provider", "OAuth2 via Auth0 tenant prod-dmb",
         owner="backend-agent", type="fact", confidence=1.0)
    time.sleep(pause)

    _put("auth/token-ttl", "Access tokens expire after 15 minutes",
         owner="backend-agent", type="fact", confidence=0.9)
    time.sleep(pause)

    _put("auth/migration-plan",
         "Migrate session store from Redis 6 to Redis 7 cluster mode",
         owner="planner-agent", type="plan", confidence=0.85,
         refs=["auth/provider", "auth/token-ttl"])
    time.sleep(pause)

    _put("auth/session-load", "Current p99 session-lookup latency: 4.2 ms",
         owner="qa-agent", type="observation", confidence=0.8)
    time.sleep(pause)

    # ── monitoring/ prefix ──────────────────────────────────────

    _put("monitoring/stack", "Prometheus + Grafana on dedicated monitoring namespace",
         owner="backend-agent", type="fact", confidence=1.0)
    time.sleep(pause)

    _put("monitoring/alert-rules",
         "Page if error_rate > 1 % or p99_latency > 500 ms for 2 min",
         owner="planner-agent", type="decision", confidence=0.9)
    time.sleep(pause)

    _put("monitoring/dashboard", "Grafana dashboard deploy-overview updated with canary panels",
         owner="qa-agent", type="observation", confidence=0.85)
    time.sleep(pause)

    # ── Contradiction 1 — deploy strategy conflict ──────────────
    # backend-agent overwrites the planner-agent's blue-green decision
    # with an incompatible rolling-update claim -> CONFLICT event
    _put("deploy/strategy", "Rolling update with maxUnavailable=25 %",
         owner="backend-agent", type="decision", confidence=0.8)
    time.sleep(pause)

    # ── Contradiction 2 — token TTL conflict ────────────────────
    # qa-agent reports a different TTL than backend-agent recorded
    _put("auth/token-ttl", "Access tokens expire after 60 minutes",
         owner="qa-agent", type="observation", confidence=0.7)

    # ── Gap trigger ─────────────────────────────────────────────
    # The plan at deploy/rollout-steps refs "monitoring/error-rate",
    # but no entry exists at that path -> GAP event.
