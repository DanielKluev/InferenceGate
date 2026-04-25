"""
Tests for the ``/gate/*`` admin namespace.

Covers redaction helpers, GET/POST config, index reload, stats, 404 handling
for unknown gate endpoints, and verifies the ``/gate/`` namespace is reserved
from being routed upstream.
"""

import pytest
from aiohttp.test_utils import TestClient, TestServer

from inference_gate.inflow.admin import _redact_url, redact, redact_headers
from inference_gate.inflow.server import InflowServer
from inference_gate.modes import Mode
from inference_gate.recording.storage import CacheStorage
from inference_gate.router.router import Router


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture
def replay_router(temp_cache_dir):
    """Create a Router in replay-only mode for admin tests."""
    storage = CacheStorage(temp_cache_dir)
    return Router(mode=Mode.REPLAY_ONLY, storage=storage, fuzzy_model=True, fuzzy_sampling="soft")


@pytest.fixture
async def admin_client(replay_router):
    """Create a test client wired with the admin namespace."""
    server = InflowServer(host="127.0.0.1", port=0, router=replay_router)
    app = server._create_app()
    async with TestClient(TestServer(app)) as client:
        yield client


class TestRedactionHelpers:
    """Unit tests for the pure-function redaction helpers."""

    def test_redact_url_with_userinfo(self):
        """URL with embedded credentials replaces userinfo with ``***@``."""
        assert _redact_url("https://user:pass@example.com:8080/path") == "https://***@example.com:8080/path"

    def test_redact_url_without_userinfo(self):
        """URL without userinfo passes through unchanged."""
        assert _redact_url("https://example.com:8080/path") == "https://example.com:8080/path"

    def test_redact_url_non_url_string(self):
        """Non-URL strings pass through unchanged."""
        assert _redact_url("not a url") == "not a url"
        assert _redact_url("") == ""

    def test_redact_headers_masks_authorization(self):
        """Authorization-style headers are masked, others preserved."""
        result = redact_headers({"Authorization": "Bearer secret", "Content-Type": "application/json", "X-API-Key": "abc"})
        assert result["Authorization"] == "***"
        assert result["X-API-Key"] == "***"
        assert result["Content-Type"] == "application/json"

    def test_redact_recursive_masks_secret_keys(self):
        """Recursive redactor masks nested keys whose name suggests a secret."""
        payload = {
            "model": "gpt-4",
            "api_key": "sk-abc",
            "nested": {"password": "p", "ok": "v"},
            "items": [{"token": "t"}],
        }
        result = redact(payload)
        assert result["api_key"] == "***"
        assert result["nested"]["password"] == "***"
        assert result["nested"]["ok"] == "v"
        assert result["items"][0]["token"] == "***"
        assert result["model"] == "gpt-4"


class TestGateHealth:
    """Tests for ``GET /gate/health``."""

    async def test_health_returns_mode_and_fuzzy(self, admin_client):
        """Gate health echoes mode and the live fuzzy settings."""
        resp = await admin_client.get("/gate/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "healthy"
        assert data["mode"] == "replay-only"
        assert data["fuzzy_model"] is True
        assert data["fuzzy_sampling"] == "soft"


class TestGateConfig:
    """Tests for GET/POST ``/gate/config``."""

    async def test_get_config_snapshot(self, admin_client, replay_router):
        """GET returns a redacted snapshot reflecting the live router."""
        resp = await admin_client.get("/gate/config")
        assert resp.status == 200
        data = await resp.json()
        assert data["mode"] == "replay-only"
        assert data["fuzzy_model"] is True
        assert data["fuzzy_sampling"] == "soft"
        assert data["max_non_greedy_replies"] == replay_router.max_non_greedy_replies
        assert data["outflow"] is None  # replay-only has no outflow
        assert "stats" in data and "cassettes" in data["stats"]

    async def test_post_config_updates_fuzzy_settings(self, admin_client, replay_router):
        """POST applies new fuzzy_model/fuzzy_sampling and reflects in router state."""
        resp = await admin_client.post("/gate/config", json={"fuzzy_model": False, "fuzzy_sampling": "off"})
        assert resp.status == 200
        data = await resp.json()
        assert data["fuzzy_model"] is False
        assert data["fuzzy_sampling"] == "off"
        assert replay_router.fuzzy_model is False
        assert replay_router.fuzzy_sampling == "off"

    async def test_post_config_rejects_unknown_keys(self, admin_client):
        """POST with unknown keys returns 400 listing accepted set."""
        resp = await admin_client.post("/gate/config", json={"made_up": 1})
        assert resp.status == 400
        data = await resp.json()
        assert data["error"]["type"] == "invalid_request"
        assert "made_up" in data["error"]["message"]

    async def test_post_config_rejects_invalid_mode(self, admin_client):
        """POST with bogus mode value returns 400."""
        resp = await admin_client.post("/gate/config", json={"mode": "warp"})
        assert resp.status == 400
        data = await resp.json()
        assert data["error"]["type"] == "invalid_request"

    async def test_post_config_rejects_invalid_fuzzy_sampling(self, admin_client):
        """POST with bogus fuzzy_sampling returns 400."""
        resp = await admin_client.post("/gate/config", json={"fuzzy_sampling": "extreme"})
        assert resp.status == 400

    async def test_post_config_rejects_non_object_body(self, admin_client):
        """POST with a non-object body is rejected."""
        resp = await admin_client.post("/gate/config", data="not json", headers={"content-type": "application/json"})
        assert resp.status == 400


class TestGateIndexReload:
    """Tests for ``POST /gate/index/reload``."""

    async def test_reload_returns_cassette_count(self, admin_client):
        """Reload returns the new cassette count from the rebuilt index."""
        resp = await admin_client.post("/gate/index/reload")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "reloaded"
        assert data["cassettes"] == 0


class TestGateStats:
    """Tests for ``GET /gate/stats``."""

    async def test_stats_returns_zero_for_empty_cache(self, admin_client):
        """Stats endpoint reports zero cassettes for an empty cache directory."""
        resp = await admin_client.get("/gate/stats")
        assert resp.status == 200
        data = await resp.json()
        assert data["cassettes"] == 0
        assert data["mode"] == "replay-only"


class TestGateNamespaceReserved:
    """The /gate/ namespace must never fall through to the proxy handler."""

    async def test_unknown_gate_path_returns_404(self, admin_client):
        """Unknown ``/gate/*`` paths return a structured 404, not a 503 proxy miss."""
        resp = await admin_client.get("/gate/does-not-exist")
        assert resp.status == 404
        data = await resp.json()
        assert data["error"]["type"] == "not_found"

    async def test_unknown_gate_post_returns_404(self, admin_client):
        """POST to unknown ``/gate/*`` path also returns 404."""
        resp = await admin_client.post("/gate/whatever", json={})
        assert resp.status == 404
