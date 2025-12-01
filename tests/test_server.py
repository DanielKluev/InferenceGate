"""Tests for InferenceReplay server module."""

import pytest
from fastapi.testclient import TestClient

from inference_replay.modes import Mode
from inference_replay.server import ProxyConfig, create_app
from inference_replay.storage import (
    CachedRequest,
    CachedResponse,
    CacheEntry,
    CacheStorage,
)


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture
def replay_client(temp_cache_dir):
    """Create a test client in replay mode."""
    config = ProxyConfig(
        mode=Mode.REPLAY,
        cache_dir=temp_cache_dir,
    )
    app = create_app(config)
    return TestClient(app)


@pytest.fixture
def dev_client(temp_cache_dir):
    """Create a test client in development mode."""
    config = ProxyConfig(
        mode=Mode.DEVELOPMENT,
        cache_dir=temp_cache_dir,
        upstream_base_url="http://localhost:9999",  # Won't be used with cache hits
    )
    app = create_app(config)
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_replay_mode(self, replay_client):
        """Test health endpoint in replay mode."""
        response = replay_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["mode"] == "replay"


class TestReplayMode:
    """Tests for replay mode behavior."""

    def test_cache_miss_returns_503(self, replay_client):
        """Test that cache miss in replay mode returns 503."""
        response = replay_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 503
        assert "No cached response" in response.json()["detail"]

    def test_cache_hit_returns_cached(self, temp_cache_dir, replay_client):
        """Test that cache hit returns cached response."""
        # Pre-populate cache
        storage = CacheStorage(temp_cache_dir)
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json", "accept": "*/*"},
            body={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
        )
        response = CachedResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body={
                "id": "chatcmpl-123",
                "choices": [{"message": {"role": "assistant", "content": "Hi!"}}],
            },
        )
        entry = CacheEntry(request=request, response=response, model="gpt-4")
        storage.put(entry)

        # Now request should hit cache
        response = replay_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "chatcmpl-123"
        assert data["choices"][0]["message"]["content"] == "Hi!"


class TestDevelopmentMode:
    """Tests for development mode behavior."""

    def test_cache_hit_in_dev_mode(self, temp_cache_dir, dev_client):
        """Test that cache hit in dev mode returns cached response."""
        # Pre-populate cache
        storage = CacheStorage(temp_cache_dir)
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json", "accept": "*/*"},
            body={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Test"}]},
        )
        response = CachedResponse(
            status_code=200,
            headers={},
            body={"choices": [{"message": {"content": "Cached response"}}]},
        )
        entry = CacheEntry(request=request, response=response)
        storage.put(entry)

        response = dev_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Test"}]},
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "Cached response"


class TestStreamingReplay:
    """Tests for streaming response replay."""

    def test_streaming_cache_hit(self, temp_cache_dir, replay_client):
        """Test that streaming responses are replayed correctly."""
        # Pre-populate cache with streaming response
        storage = CacheStorage(temp_cache_dir)
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json", "accept": "*/*"},
            body={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Stream test"}],
                "stream": True,
            },
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
        with replay_client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Stream test"}],
                "stream": True,
            },
        ) as r:
            assert r.status_code == 200
            content = b"".join(r.iter_bytes()).decode("utf-8")
            assert "Hello" in content
            assert "World" in content
