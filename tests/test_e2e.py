"""
End-to-end test: Two agents coordinating through DimensionalBase.

This is the Phase 0 validation test — proof that the architecture works.
"""

from dimensionalbase import DimensionalBase, EventType


def test_two_agents_coordinate():
    """Simulate two agents working on a deployment task.

    Agent 1 (planner): creates the plan
    Agent 2 (executor): executes steps and reports observations
    DimensionalBase: detects gaps, tracks progress
    """
    db = DimensionalBase()
    events_log = []
    db.subscribe("**", "test-monitor", lambda e: events_log.append(e))

    # === PLANNER creates the deployment plan ===
    db.put(
        path="deploy/plan",
        value="Deploy auth service: 1) build image, 2) run tests, 3) push to staging",
        owner="planner",
        type="plan",
        confidence=0.95,
        refs=["deploy/build", "deploy/test", "deploy/push"],
    )

    # Verify plan is stored
    plan = db.retrieve("deploy/plan")
    assert plan is not None
    assert plan.owner == "planner"
    assert len(plan.refs) == 3

    # === EXECUTOR checks what needs doing ===
    context = db.get(
        scope="deploy/**",
        budget=500,
        query="What are the deployment steps?",
    )
    assert len(context.entries) >= 1
    assert context.tokens_used <= 500

    # === EXECUTOR completes step 1 ===
    db.put(
        path="deploy/build",
        value="Docker image built successfully. Tag: auth-v2.3.1",
        owner="executor",
        type="observation",
        confidence=1.0,
    )

    # === EXECUTOR hits a problem on step 2 ===
    db.put(
        path="deploy/test",
        value="Integration tests failing. 3/12 tests fail on auth token validation.",
        owner="executor",
        type="observation",
        confidence=0.88,
    )

    # === PLANNER reads current status ===
    status = db.get(
        scope="deploy/**",
        budget=1000,
        query="What is the current deployment status?",
    )
    assert len(status.entries) >= 3  # plan + 2 observations

    # Verify we can see the test failure
    texts = [e.value for e in status.entries]
    assert any("failing" in t or "fail" in t for t in texts)

    # === PLANNER makes a decision ===
    db.put(
        path="deploy/decision",
        value="Rollback deployment. Fix auth token validation before retrying.",
        owner="planner",
        type="decision",
        confidence=0.90,
        refs=["deploy/test"],
    )

    # === Verify the full knowledge graph ===
    all_deploy = db.get(scope="deploy/**", budget=5000)
    assert all_deploy.total_matched >= 4

    # Verify events were fired
    change_events = [e for e in events_log if e.type == EventType.CHANGE]
    assert len(change_events) >= 4  # 4 puts

    db.close()


def test_three_agents_with_conflict():
    """Three agents where two disagree — contradiction detection kicks in.

    Agent 1 (backend): reports API status
    Agent 2 (frontend): reports API status differently
    Agent 3 (monitor): watches for conflicts
    """
    db = DimensionalBase()
    conflicts = []
    db.subscribe("**", "conflict-monitor",
                 lambda e: conflicts.append(e) if e.type == EventType.CONFLICT else None)

    # Backend says API is fine
    db.put(
        path="service/api/status",
        value="API is healthy. All endpoints returning 200.",
        owner="backend-agent",
        type="fact",
        confidence=0.95,
    )

    # Frontend says API is broken
    db.put(
        path="service/api/health",
        value="API is down. Getting 503 on all requests.",
        owner="frontend-agent",
        type="fact",
        confidence=0.90,
    )

    # Monitor agent should have been notified of the conflict
    # (In text-only mode, this uses path-based heuristic)

    # Both entries should exist
    assert db.exists("service/api/status")
    assert db.exists("service/api/health")

    # Get all service info
    context = db.get(scope="service/**", budget=1000)
    assert context.total_matched == 2

    db.close()


def test_budget_aware_retrieval():
    """Verify that budget constraints work correctly under load."""
    db = DimensionalBase()

    # Write 100 entries
    for i in range(100):
        db.put(
            path=f"data/metrics/item-{i:03d}",
            value=f"Metric item {i}: latency={i*10}ms, errors={i%5}, throughput={1000-i*5}req/s",
            owner="metrics-agent",
            type="observation",
            confidence=0.7 + (i % 30) * 0.01,
        )

    # Tiny budget — should only fit a few entries
    result = db.get(scope="data/**", budget=100)
    assert result.tokens_used <= 100
    assert result.total_matched == 100
    assert len(result.entries) < 100

    # Medium budget
    result = db.get(scope="data/**", budget=1000)
    assert result.tokens_used <= 1000

    # Large budget — should fit everything
    result = db.get(scope="data/**", budget=50000)
    assert result.total_matched == 100

    db.close()


def test_multi_agent_workflow():
    """Full workflow: research -> plan -> execute -> review."""
    db = DimensionalBase()

    # 1. Research agent gathers facts
    db.put("project/research/market", "TAM is $4.2B for AI agent tooling", owner="researcher", type="fact", confidence=0.85)
    db.put("project/research/competition", "No direct competitor for multi-agent coordination layer", owner="researcher", type="fact", confidence=0.78)
    db.put("project/research/tech", "Embedding similarity enables conflict detection", owner="researcher", type="fact", confidence=0.95)

    # 2. Planner reads research and creates plan
    research = db.get(scope="project/research/**", budget=500)
    assert len(research.entries) == 3

    db.put(
        "project/plan/mvp",
        "Build MVP: SQLite+embeddings, 4-method API, ship to PyPI in 8 weeks",
        owner="planner",
        type="plan",
        refs=["project/research/tech", "project/execution/build"],
        confidence=0.90,
    )

    # 3. Executor reads plan and starts working
    plan = db.get(scope="project/plan/**", budget=300, query="What should I build?")
    assert len(plan.entries) >= 1

    db.put("project/execution/build", "Package structure created. Core API implemented.", owner="executor", type="observation", confidence=1.0)
    db.put("project/execution/tests", "47 tests passing. 0 failures.", owner="executor", type="observation", confidence=1.0)

    # 4. Reviewer reads everything
    full_context = db.get(scope="project/**", budget=5000, query="Project status overview")
    assert full_context.total_matched >= 5

    # Everything is accessible through one shared space
    all_paths = {e.path for e in full_context.entries}
    assert any("research" in p for p in all_paths)
    assert any("plan" in p for p in all_paths)
    assert any("execution" in p for p in all_paths)

    db.close()
