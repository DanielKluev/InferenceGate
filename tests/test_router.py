"""Tests for InferenceGate router module."""

import pytest

from inference_gate.modes import Mode
from inference_gate.outflow.client import OutflowClient
from inference_gate.recording.storage import CachedRequest, CachedResponse, CacheEntry, CacheStorage
from inference_gate.router.router import Router


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture
def storage(temp_cache_dir):
    """Create a CacheStorage instance."""
    return CacheStorage(temp_cache_dir)


@pytest.fixture
def replay_router(storage):
    """Create a Router in replay-only mode."""
    return Router(mode=Mode.REPLAY_ONLY, storage=storage)


class TestRouterInit:
    """Tests for Router initialization."""

    def test_replay_only_no_outflow_needed(self, storage):
        """Test that REPLAY_ONLY mode works without an OutflowClient."""
        router = Router(mode=Mode.REPLAY_ONLY, storage=storage)
        assert router.mode == Mode.REPLAY_ONLY

    def test_record_and_replay_requires_outflow(self, storage):
        """Test that RECORD_AND_REPLAY mode raises ValueError without OutflowClient."""
        with pytest.raises(ValueError, match="OutflowClient is required"):
            Router(mode=Mode.RECORD_AND_REPLAY, storage=storage)


class TestReplayOnlyRouting:
    """Tests for routing in replay-only mode."""

    async def test_cache_miss_returns_503(self, replay_router):
        """Test that cache miss in replay-only mode returns 503 error response."""
        response = await replay_router.route_request(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 503
        assert response.body is not None
        assert "error" in response.body

    async def test_cache_hit_returns_cached(self, replay_router, storage):
        """Test that cache hit in replay-only mode returns the stored response."""
        # Pre-populate cache
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
        )
        cached_resp = CachedResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body={"id": "chatcmpl-123", "choices": [{"message": {"content": "Hi!"}}]},
        )
        entry = CacheEntry(request=request, response=cached_resp, model="gpt-4")
        storage.put(entry)

        # Route should hit cache
        response = await replay_router.route_request(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 200
        assert response.body["id"] == "chatcmpl-123"

    async def test_streaming_cache_hit(self, replay_router, storage):
        """Test that streaming cached response is returned correctly."""
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={"model": "gpt-4", "messages": [{"role": "user", "content": "Stream"}], "stream": True},
        )
        chunks = ['data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n', "data: [DONE]\n\n"]
        cached_resp = CachedResponse(status_code=200, headers={}, chunks=chunks, is_streaming=True)
        entry = CacheEntry(request=request, response=cached_resp)
        storage.put(entry)

        response = await replay_router.route_request(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={"model": "gpt-4", "messages": [{"role": "user", "content": "Stream"}], "stream": True},
        )
        assert response.is_streaming
        assert len(response.chunks) == 2
