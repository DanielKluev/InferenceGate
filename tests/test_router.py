"""Tests for InferenceGate router module."""

from unittest.mock import AsyncMock, MagicMock

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


@pytest.fixture
def mock_outflow():
    """Create a mock OutflowClient for testing record-and-replay mode."""
    outflow = MagicMock(spec=OutflowClient)
    outflow.forward_request = AsyncMock(return_value=CachedResponse(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        chunks=['data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n', 'data: [DONE]\n\n'],
        is_streaming=True,
    ))
    return outflow


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

    def test_non_streaming_models_default_empty(self, storage):
        """Test that non_streaming_models defaults to empty list."""
        router = Router(mode=Mode.REPLAY_ONLY, storage=storage)
        assert router.non_streaming_models == []

    def test_non_streaming_models_accepted(self, storage):
        """Test that non_streaming_models can be passed to constructor."""
        router = Router(mode=Mode.REPLAY_ONLY, storage=storage, non_streaming_models=["o1-preview"])
        assert router.non_streaming_models == ["o1-preview"]


class TestReplayOnlyRouting:
    """Tests for routing in replay-only mode."""

    async def test_cache_miss_returns_503(self, replay_router):
        """Test that cache miss in replay-only mode returns 503 error response."""
        response = await replay_router.route_request(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "Hello"
                }]
            },
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
            body={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "Hello"
                }]
            },
        )
        cached_resp = CachedResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body={
                "id": "chatcmpl-123",
                "choices": [{
                    "message": {
                        "content": "Hi!"
                    }
                }]
            },
        )
        entry = CacheEntry(request=request, response=cached_resp, model="gpt-4")
        storage.put(entry)

        # Route should hit cache
        response = await replay_router.route_request(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "Hello"
                }]
            },
        )
        assert response.status_code == 200
        assert response.body["id"] == "chatcmpl-123"

    async def test_streaming_cache_hit(self, replay_router, storage):
        """Test that streaming cached response is returned correctly."""
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "Stream"
                }],
                "stream": True
            },
        )
        chunks = ['data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n', "data: [DONE]\n\n"]
        cached_resp = CachedResponse(status_code=200, headers={}, chunks=chunks, is_streaming=True)
        entry = CacheEntry(request=request, response=cached_resp)
        storage.put(entry)

        response = await replay_router.route_request(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "Stream"
                }],
                "stream": True
            },
        )
        assert response.is_streaming
        assert len(response.chunks) == 2

    async def test_non_streaming_client_hits_streaming_cache(self, replay_router, storage):
        """Test that a non-streaming client request hits a streaming cassette via normalized cache key."""
        # Store a streaming cassette (as the router would after forcing streaming)
        request_stored = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "Hello"
                }],
                "stream": True
            },
        )
        chunks = ['data: {"choices":[{"delta":{"content":"Hi!"}}]}\n\n', "data: [DONE]\n\n"]
        cached_resp = CachedResponse(status_code=200, headers={}, chunks=chunks, is_streaming=True)
        entry = CacheEntry(request=request_stored, response=cached_resp)
        storage.put(entry)

        # Non-streaming client request (stream=False) should find the same cassette
        response = await replay_router.route_request(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "Hello"
                }],
                "stream": False
            },
        )
        assert response.is_streaming
        assert response.chunks is not None


class TestForceStreaming:
    """Tests for the forced streaming behavior in RECORD_AND_REPLAY mode."""

    async def test_forces_streaming_on_upstream(self, storage, mock_outflow):
        """Test that router forces stream=True on upstream requests even when client sends stream=False."""
        router = Router(mode=Mode.RECORD_AND_REPLAY, storage=storage, outflow=mock_outflow)

        await router.route_request(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "Hello"
                }],
                "stream": False
            },
        )

        # Verify the request forwarded to outflow had stream=True
        call_args = mock_outflow.forward_request.call_args
        forwarded_request = call_args[0][0]
        assert forwarded_request.body["stream"] is True

    async def test_forces_stream_options_include_usage(self, storage, mock_outflow):
        """Test that router adds stream_options.include_usage=True when forcing streaming."""
        router = Router(mode=Mode.RECORD_AND_REPLAY, storage=storage, outflow=mock_outflow)

        await router.route_request(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "Hello"
                }]
            },
        )

        call_args = mock_outflow.forward_request.call_args
        forwarded_request = call_args[0][0]
        assert forwarded_request.body["stream_options"] == {"include_usage": True}

    async def test_preserves_existing_stream_options(self, storage, mock_outflow):
        """Test that router preserves existing stream_options while adding include_usage."""
        router = Router(mode=Mode.RECORD_AND_REPLAY, storage=storage, outflow=mock_outflow)

        await router.route_request(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "Hello"
                }],
                "stream_options": {
                    "some_other": "option"
                }
            },
        )

        call_args = mock_outflow.forward_request.call_args
        forwarded_request = call_args[0][0]
        assert forwarded_request.body["stream_options"]["include_usage"] is True
        assert forwarded_request.body["stream_options"]["some_other"] == "option"

    async def test_non_streaming_model_not_forced(self, storage, mock_outflow):
        """Test that models in non_streaming_models list are not forced to stream."""
        # Configure outflow to return a non-streaming response for this model
        mock_outflow.forward_request = AsyncMock(return_value=CachedResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body={
                "id": "chatcmpl-1",
                "choices": [{
                    "message": {
                        "content": "Hi"
                    }
                }]
            },
            is_streaming=False,
        ))
        router = Router(mode=Mode.RECORD_AND_REPLAY, storage=storage, outflow=mock_outflow, non_streaming_models=["o1-preview"])

        await router.route_request(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "o1-preview",
                "messages": [{
                    "role": "user",
                    "content": "Reason about X"
                }],
                "stream": False
            },
        )

        call_args = mock_outflow.forward_request.call_args
        forwarded_request = call_args[0][0]
        assert forwarded_request.body["stream"] is False

    async def test_stores_original_client_streaming_metadata(self, storage, mock_outflow):
        """Test that original_client_streaming is recorded in CacheEntry metadata."""
        router = Router(mode=Mode.RECORD_AND_REPLAY, storage=storage, outflow=mock_outflow)

        await router.route_request(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "Hi"
                }],
                "stream": False
            },
        )

        # Check stored entry has original_client_streaming=False
        entries = storage.list_entries()
        assert len(entries) == 1
        _, entry = entries[0]
        assert entry.original_client_streaming is False

    async def test_cache_hit_does_not_forward(self, storage, mock_outflow):
        """Test that cache hit does not forward to upstream."""
        router = Router(mode=Mode.RECORD_AND_REPLAY, storage=storage, outflow=mock_outflow)

        # Pre-populate cache
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "Cached"
                }],
                "stream": True
            },
        )
        cached_resp = CachedResponse(status_code=200, headers={}, chunks=["data: test\n\n"], is_streaming=True)
        entry = CacheEntry(request=request, response=cached_resp)
        storage.put(entry)

        await router.route_request(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "Cached"
                }]
            },
        )

        mock_outflow.forward_request.assert_not_called()
