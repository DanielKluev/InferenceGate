"""Tests for InferenceGate inflow server module."""

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, TestClient, TestServer

from inference_gate.inflow.server import InflowServer
from inference_gate.modes import Mode
from inference_gate.recording.storage import CachedRequest, CachedResponse, CacheEntry, CacheStorage
from inference_gate.router.router import Router


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture
def replay_router(temp_cache_dir):
    """Create a Router in replay-only mode."""
    storage = CacheStorage(temp_cache_dir)
    return Router(mode=Mode.REPLAY_ONLY, storage=storage)


@pytest.fixture
def replay_server(replay_router):
    """Create an InflowServer in replay-only mode."""
    return InflowServer(host="127.0.0.1", port=0, router=replay_router)


@pytest.fixture
async def replay_client(replay_server):
    """Create an aiohttp test client for the replay-only server."""
    app = replay_server._create_app()
    async with TestClient(TestServer(app)) as client:
        yield client


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    async def test_health_replay_mode(self, replay_client):
        """Test that health endpoint returns healthy status and correct mode."""
        resp = await replay_client.get("/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "healthy"
        assert data["mode"] == "replay-only"


class TestReplayMode:
    """Tests for replay mode behavior."""

    async def test_cache_miss_returns_503(self, replay_client):
        """Test that cache miss in replay-only mode returns 503 with error details."""
        resp = await replay_client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]})
        assert resp.status == 503
        data = await resp.json()
        assert "error" in data
        assert "no_cached_response" in data["error"]["code"]

    async def test_cache_hit_returns_cached(self, temp_cache_dir, replay_client, replay_router):
        """Test that cache hit in replay-only mode returns the cached response."""
        # Pre-populate cache
        storage = replay_router.storage
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
        )
        response = CachedResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body={"id": "chatcmpl-123", "choices": [{"message": {"role": "assistant", "content": "Hi!"}}]},
        )
        entry = CacheEntry(request=request, response=response, model="gpt-4")
        storage.put(entry)

        # Now request should hit cache
        resp = await replay_client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]})
        assert resp.status == 200
        data = await resp.json()
        assert data["id"] == "chatcmpl-123"
        assert data["choices"][0]["message"]["content"] == "Hi!"


class TestStreamingReplay:
    """Tests for streaming response replay."""

    async def test_streaming_cache_hit(self, replay_client, replay_router):
        """Test that streaming responses are replayed correctly with all chunks present."""
        # Pre-populate cache with streaming response
        storage = replay_router.storage
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={"model": "gpt-4", "messages": [{"role": "user", "content": "Stream test"}], "stream": True},
        )
        chunks = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":" World"}}]}\n\n',
            "data: [DONE]\n\n",
        ]
        response = CachedResponse(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            chunks=chunks,
            is_streaming=True,
        )
        entry = CacheEntry(request=request, response=response, model="gpt-4")
        storage.put(entry)

        # Request streaming response
        resp = await replay_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Stream test"}], "stream": True},
        )
        assert resp.status == 200
        content = await resp.text()
        assert "Hello" in content
        assert "World" in content
