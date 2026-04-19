"""Tests for the hardened REST and WebSocket server surface."""

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from dimensionalbase import DimensionalBase
from dimensionalbase.security.acl import AccessController, AgentPolicy
from dimensionalbase.security.auth import APIKeyManager
from dimensionalbase.security.middleware import SecureDimensionalBase
from dimensionalbase.server.app import create_app
from dimensionalbase.embeddings.provider import EmbeddingProvider

import numpy as np


@pytest.fixture
def secured_server():
    db = DimensionalBase()
    mgr = APIKeyManager()
    key = mgr.generate_key("agent-1")

    acl = AccessController()
    acl.register_policy(AgentPolicy(
        agent_id="agent-1",
        allowed_read_patterns=["task/agent-1/**"],
        allowed_write_patterns=["task/agent-1/**"],
    ))

    secure = SecureDimensionalBase(db, key_manager=mgr, acl=acl)
    app = create_app(secure)

    yield {
        "db": db,
        "mgr": mgr,
        "key": key,
        "client": TestClient(app),
    }

    db.close()
    mgr.close()


class TestSecureServer:
    def test_healthz_is_open(self, secured_server):
        response = secured_server["client"].get("/healthz")
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert "version" in body

    def test_status_requires_api_key(self, secured_server):
        response = secured_server["client"].get("/api/v1/status")
        assert response.status_code == 401

    def test_rest_put_enforces_owner_binding(self, secured_server):
        client = secured_server["client"]
        headers = {"X-DMB-API-Key": secured_server["key"]}

        forbidden = client.post(
            "/api/v1/entries",
            headers=headers,
            json={
                "path": "task/agent-1/status",
                "value": "ok",
                "owner": "agent-2",
            },
        )
        assert forbidden.status_code == 403

        allowed = client.post(
            "/api/v1/entries",
            headers=headers,
            json={
                "path": "task/agent-1/status",
                "value": "ok",
                "owner": "agent-1",
            },
        )
        assert allowed.status_code in (200, 201)
        assert allowed.json()["owner"] == "agent-1"

    def test_websocket_respects_requested_patterns(self, secured_server):
        db = secured_server["db"]
        key = secured_server["key"]

        with secured_server["client"].websocket_connect(f"/ws/subscribe?api_key={key}") as ws:
            ws.send_text(json.dumps({"pattern": "task/agent-2/**"}))
            denied = json.loads(ws.receive_text())
            assert denied["ack"] is False

            ws.send_text(json.dumps({"pattern": "task/agent-1/**"}))
            allowed = json.loads(ws.receive_text())
            assert allowed == {"ack": True, "pattern": "task/agent-1/**"}

            db.put("task/agent-1/status", "healthy", owner="agent-1")
            event = json.loads(ws.receive_text())
            assert event["path"] == "task/agent-1/status"
            assert event["type"] == "change"

    def test_internal_exceptions_return_500(self):
        db = DimensionalBase()

        def boom(*args, **kwargs):
            raise RuntimeError("boom")

        db.put = boom  # type: ignore[assignment]
        client = TestClient(create_app(db), raise_server_exceptions=False)
        response = client.post(
            "/api/v1/entries",
            json={"path": "task/x", "value": "value", "owner": "agent-1"},
        )
        assert response.status_code == 500
        body = response.json()
        if "error" in body:
            assert body["error"]["type"] == "internal_error"
        else:
            assert body["detail"] == "Internal server error"
        db.close()

    def test_body_limit_rejects_large_payloads(self):
        db = DimensionalBase()
        client = TestClient(
            create_app(db, server_config={"max_request_body_bytes": 32}),
            raise_server_exceptions=False,
        )
        response = client.post(
            "/api/v1/entries",
            json={"path": "task/x", "value": "x" * 128, "owner": "agent-1"},
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "payload_too_large"
        db.close()

    def test_compose_filters_disallowed_neighbors(self):
        class FixedEmbeddingProvider(EmbeddingProvider):
            def embed(self, text: str) -> np.ndarray:
                return np.array([1.0, 0.0], dtype=np.float32)

            def embed_batch(self, texts):
                return [self.embed(text) for text in texts]

            def dimension(self) -> int:
                return 2

            @property
            def name(self) -> str:
                return "fixed"

        db = DimensionalBase(embedding_provider=FixedEmbeddingProvider())
        mgr = APIKeyManager()
        key = mgr.generate_key("agent-1")

        acl = AccessController()
        acl.register_policy(AgentPolicy(
            agent_id="agent-1",
            allowed_read_patterns=["task/agent-1/**"],
            allowed_write_patterns=["task/agent-1/**"],
        ))

        secure = SecureDimensionalBase(db, key_manager=mgr, acl=acl)
        client = TestClient(create_app(secure))
        headers = {"X-DMB-API-Key": key}

        db.put("task/agent-1/a", "allowed a", owner="agent-1")
        db.put("task/agent-1/b", "allowed b", owner="agent-1")
        db.put("task/agent-2/secret", "secret", owner="agent-2")

        response = client.post(
            "/api/v1/compose",
            headers=headers,
            json={"paths": ["task/agent-1/a", "task/agent-1/b"], "mode": "attentive", "k": 5},
        )
        assert response.status_code == 200
        paths = [item["path"] for item in response.json()["nearest"]]
        assert "task/agent-2/secret" not in paths

        secure.close()
