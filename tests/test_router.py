"""Tests for InferenceGate router module (v2 with tiered fuzzy matching and multi-reply)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from inference_gate.modes import Mode
from inference_gate.outflow.client import OutflowClient
from inference_gate.recording.storage import CachedRequest, CachedResponse, CacheEntry, CacheStorage
from inference_gate.router.router import ReplayCounter, Router


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


class TestReplayCounter:
    """Tests for ReplayCounter round-robin, random, and first strategies."""

    def test_round_robin_cycling(self):
        """Test that round-robin cycles through replies sequentially."""
        counter = ReplayCounter()
        results = [counter.next_reply("hash1", 3) for _ in range(6)]
        assert results == [1, 2, 3, 1, 2, 3]

    def test_first_always_returns_1(self):
        """Test that 'first' strategy always returns reply 1."""
        counter = ReplayCounter()
        results = [counter.next_reply("hash1", 5, strategy="first") for _ in range(3)]
        assert results == [1, 1, 1]

    def test_random_within_range(self):
        """Test that 'random' strategy returns values within valid range."""
        counter = ReplayCounter()
        for _ in range(20):
            r = counter.next_reply("hash1", 3, strategy="random")
            assert 1 <= r <= 3


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
                }],
                "temperature": 0,
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
                }],
                "temperature": 0,
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
                }],
                "temperature": 0,
            },
        )
        assert response.status_code == 200

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
                "stream": True,
                "temperature": 0,
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
                "stream": True,
                "temperature": 0,
            },
        )
        assert response.is_streaming
        assert response.chunks is not None

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
                "stream": True,
                "temperature": 0,
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
                "stream": False,
                "temperature": 0,
            },
        )
        assert response.status_code == 200


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
                "stream": False,
                "temperature": 0,
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
                }],
                "temperature": 0,
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
                },
                "temperature": 0,
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
                "stream": False,
                "temperature": 0,
            },
        )

        call_args = mock_outflow.forward_request.call_args
        forwarded_request = call_args[0][0]
        assert forwarded_request.body["stream"] is False

    async def test_stores_original_client_streaming_metadata(self, storage, mock_outflow):
        """Test that storage records the entry after upstream forward."""
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
                "stream": False,
                "temperature": 0,
            },
        )

        # Check stored entry exists in index
        entries = storage.list_entries()
        assert len(entries) == 1

    async def test_cache_hit_does_not_forward(self, storage, mock_outflow):
        """Test that cache hit does not forward to upstream."""
        router = Router(mode=Mode.RECORD_AND_REPLAY, storage=storage, outflow=mock_outflow)

        # Pre-populate cache with greedy request
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
                "stream": True,
                "temperature": 0,
            },
        )
        cached_resp = CachedResponse(
            status_code=200, headers={},
            chunks=["data: {\"choices\":[{\"delta\":{\"content\":\"test\"}}]}\n\n", "data: [DONE]\n\n"],
            is_streaming=True)
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
                }],
                "temperature": 0,
            },
        )

        mock_outflow.forward_request.assert_not_called()


class TestFuzzyModelMatching:
    """Tests for the fuzzy model matching behavior (v2 tiered lookup)."""

    async def test_fuzzy_model_disabled_by_default(self, storage):
        """Test that fuzzy model matching is disabled by default."""
        router = Router(mode=Mode.REPLAY_ONLY, storage=storage)
        assert router.fuzzy_model is False

    async def test_fuzzy_model_enabled_via_constructor(self, storage):
        """Test that fuzzy model matching can be enabled via constructor."""
        router = Router(mode=Mode.REPLAY_ONLY, storage=storage, fuzzy_model=True)
        assert router.fuzzy_model is True

    async def test_fuzzy_model_replay_only_returns_cached(self, storage):
        """Test that fuzzy model matching returns cached response from a different model in replay-only mode."""
        # Store entry with model-a
        messages = [{"role": "user", "content": "Hello fuzzy"}]
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "model-a",
                "messages": messages,
                "temperature": 0,
            },
        )
        cached_resp = CachedResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body={
                "id": "chatcmpl-fuzzy",
                "choices": [{
                    "message": {
                        "content": "Hi from model-a!"
                    }
                }]
            },
        )
        entry = CacheEntry(request=request, response=cached_resp, model="model-a")
        storage.put(entry)

        # Request with model-b and fuzzy model matching enabled
        router = Router(mode=Mode.REPLAY_ONLY, storage=storage, fuzzy_model=True)
        response = await router.route_request(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "model-b",
                "messages": messages,
                "temperature": 0,
            },
        )
        assert response.status_code == 200

    async def test_fuzzy_model_disabled_returns_503(self, storage):
        """Test that without fuzzy model matching, a cache miss with different model returns 503."""
        # Store entry with model-a
        messages = [{"role": "user", "content": "Hello no-fuzzy"}]
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "model-a",
                "messages": messages,
                "temperature": 0,
            },
        )
        cached_resp = CachedResponse(status_code=200, headers={}, body={"choices": [{"message": {"content": "Hi!"}}]})
        entry = CacheEntry(request=request, response=cached_resp, model="model-a")
        storage.put(entry)

        # Request with model-b but fuzzy matching disabled (default)
        router = Router(mode=Mode.REPLAY_ONLY, storage=storage, fuzzy_model=False)
        response = await router.route_request(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "model-b",
                "messages": messages,
                "temperature": 0,
            },
        )
        assert response.status_code == 503

    async def test_exact_match_preferred_over_fuzzy(self, storage):
        """Test that an exact cache key match is preferred over fuzzy model matching."""
        messages = [{"role": "user", "content": "Exact vs fuzzy"}]

        # Store entry with model-a
        request_a = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "model-a",
                "messages": messages,
                "temperature": 0,
            },
        )
        resp_a = CachedResponse(status_code=200, headers={}, body={"id": "from-model-a", "choices": [{"message": {"content": "A"}}]})
        entry_a = CacheEntry(request=request_a, response=resp_a, model="model-a")
        storage.put(entry_a)

        # Store entry with model-b
        request_b = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "model-b",
                "messages": messages,
                "temperature": 0,
            },
        )
        resp_b = CachedResponse(status_code=200, headers={}, body={"id": "from-model-b", "choices": [{"message": {"content": "B"}}]})
        entry_b = CacheEntry(request=request_b, response=resp_b, model="model-b")
        storage.put(entry_b)

        # Request with model-b should return the exact match, not the fuzzy match from model-a
        router = Router(mode=Mode.REPLAY_ONLY, storage=storage, fuzzy_model=True)
        response = await router.route_request(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "model-b",
                "messages": messages,
                "temperature": 0,
            },
        )
        assert response.status_code == 200

    async def test_fuzzy_model_record_and_replay_returns_cached(self, storage, mock_outflow):
        """Test that fuzzy model matching works in record-and-replay mode (avoids upstream call)."""
        messages = [{"role": "user", "content": "Hello record fuzzy"}]

        # Store entry with model-a (greedy)
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "model-a",
                "messages": messages,
                "temperature": 0,
            },
        )
        cached_resp = CachedResponse(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            chunks=['data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n', 'data: [DONE]\n\n'],
            is_streaming=True,
        )
        entry = CacheEntry(request=request, response=cached_resp, model="model-a")
        storage.put(entry)

        # Request with model-b and fuzzy model matching enabled
        router = Router(mode=Mode.RECORD_AND_REPLAY, storage=storage, outflow=mock_outflow, fuzzy_model=True)
        response = await router.route_request(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "model-b",
                "messages": messages,
                "temperature": 0,
            },
        )
        # Should use the fuzzy match, not forward to upstream
        assert response.status_code == 200
        mock_outflow.forward_request.assert_not_called()

    async def test_fuzzy_match_no_body(self, storage):
        """Test that fuzzy matching gracefully handles requests without a body."""
        router = Router(mode=Mode.REPLAY_ONLY, storage=storage, fuzzy_model=True)
        response = await router.route_request(
            method="GET",
            path="/v1/models",
            headers={},
        )
        assert response.status_code == 503


class TestFuzzySamplingMatching:
    """Tests for sampling parameter fuzzy matching (v2 tiered lookup)."""

    async def test_fuzzy_sampling_soft_matches_non_greedy(self, storage):
        """Test that soft fuzzy sampling matches a non-greedy cassette with different temperature."""
        messages = [{"role": "user", "content": "Hello sampling"}]

        # Store with temperature=0.7
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={"model": "gpt-4", "messages": messages, "temperature": 0.7},
        )
        resp = CachedResponse(status_code=200, headers={}, body={"choices": [{"message": {"content": "Hi"}}]})
        entry = CacheEntry(request=request, response=resp, model="gpt-4")
        storage.put(entry)

        # Request with temperature=0.9 and soft matching — both are non-greedy, should match
        router = Router(mode=Mode.REPLAY_ONLY, storage=storage, fuzzy_sampling="soft")
        response = await router.route_request(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={"model": "gpt-4", "messages": messages, "temperature": 0.9},
        )
        assert response.status_code == 200

    async def test_fuzzy_sampling_soft_rejects_greedy_vs_nongreedy(self, storage):
        """Test that soft fuzzy sampling does NOT match greedy with non-greedy."""
        messages = [{"role": "user", "content": "Hello greedy vs non-greedy"}]

        # Store with temperature=0 (greedy)
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={"model": "gpt-4", "messages": messages, "temperature": 0},
        )
        resp = CachedResponse(status_code=200, headers={}, body={"choices": [{"message": {"content": "Hi"}}]})
        entry = CacheEntry(request=request, response=resp, model="gpt-4")
        storage.put(entry)

        # Request with temperature=0.7 (non-greedy) and soft matching — should NOT match
        router = Router(mode=Mode.REPLAY_ONLY, storage=storage, fuzzy_sampling="soft")
        response = await router.route_request(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={"model": "gpt-4", "messages": messages, "temperature": 0.7},
        )
        assert response.status_code == 503

    async def test_fuzzy_sampling_aggressive_matches_greedy_with_nongreedy(self, storage):
        """Test that aggressive fuzzy sampling matches greedy with non-greedy cassettes."""
        messages = [{"role": "user", "content": "Hello aggressive"}]

        # Store with temperature=0.7 (non-greedy)
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={"model": "gpt-4", "messages": messages, "temperature": 0.7},
        )
        resp = CachedResponse(status_code=200, headers={}, body={"choices": [{"message": {"content": "Hi"}}]})
        entry = CacheEntry(request=request, response=resp, model="gpt-4")
        storage.put(entry)

        # Request with temperature=0 (greedy) and aggressive matching — should match
        router = Router(mode=Mode.REPLAY_ONLY, storage=storage, fuzzy_sampling="aggressive")
        response = await router.route_request(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={"model": "gpt-4", "messages": messages, "temperature": 0},
        )
        assert response.status_code == 200


class TestMultiReply:
    """Tests for multi-reply cassette behavior in RECORD_AND_REPLAY mode."""

    async def test_non_greedy_collects_multiple_replies(self, storage, mock_outflow):
        """Test that non-greedy requests are forwarded until max_replies reached."""
        router = Router(mode=Mode.RECORD_AND_REPLAY, storage=storage, outflow=mock_outflow, max_non_greedy_replies=3)
        body = {"model": "gpt-4", "messages": [{"role": "user", "content": "Non-greedy multi"}], "temperature": 0.7}

        # First request: cache miss, forward to upstream and create cassette
        await router.route_request(method="POST", path="/v1/chat/completions", headers={}, body=body.copy())
        assert mock_outflow.forward_request.call_count == 1

        # Second request: cassette exists but not full (1/3), forward again
        await router.route_request(method="POST", path="/v1/chat/completions", headers={}, body=body.copy())
        assert mock_outflow.forward_request.call_count == 2

        # Third request: cassette has 2/3, forward one more
        await router.route_request(method="POST", path="/v1/chat/completions", headers={}, body=body.copy())
        assert mock_outflow.forward_request.call_count == 3

        # Fourth request: cassette is now full (3/3), should replay
        await router.route_request(method="POST", path="/v1/chat/completions", headers={}, body=body.copy())
        # No new calls — reply served from cache
        assert mock_outflow.forward_request.call_count == 3

    async def test_greedy_single_reply(self, storage, mock_outflow):
        """Test that greedy requests (temperature=0) only record a single reply."""
        router = Router(mode=Mode.RECORD_AND_REPLAY, storage=storage, outflow=mock_outflow, max_non_greedy_replies=5)
        body = {"model": "gpt-4", "messages": [{"role": "user", "content": "Greedy single"}], "temperature": 0}

        # First request: cache miss, forward
        await router.route_request(method="POST", path="/v1/chat/completions", headers={}, body=body.copy())
        assert mock_outflow.forward_request.call_count == 1

        # Second request: greedy cassette full with 1 reply, should replay
        await router.route_request(method="POST", path="/v1/chat/completions", headers={}, body=body.copy())
        assert mock_outflow.forward_request.call_count == 1  # No additional calls


class TestPathBasedStreamingControl:
    """Tests for path-based streaming control — streaming is only forced on generation endpoints."""

    async def test_tokenize_not_forced_to_stream(self, storage, mock_outflow):
        """Test that /tokenize requests are NOT forced to stream upstream."""
        mock_outflow.forward_request = AsyncMock(return_value=CachedResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body={"tokens": [1, 2, 3]},
            is_streaming=False,
        ))
        router = Router(mode=Mode.RECORD_AND_REPLAY, storage=storage, outflow=mock_outflow)

        await router.route_request(
            method="POST",
            path="/tokenize",
            headers={"content-type": "application/json"},
            body={
                "model": "qwen3-4b-it",
                "prompt": "Hello world"
            },
        )

        call_args = mock_outflow.forward_request.call_args
        forwarded_request = call_args[0][0]
        assert "stream" not in forwarded_request.body

    async def test_detokenize_not_forced_to_stream(self, storage, mock_outflow):
        """Test that /detokenize requests are NOT forced to stream upstream."""
        mock_outflow.forward_request = AsyncMock(return_value=CachedResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body={"prompt": "Hello world"},
            is_streaming=False,
        ))
        router = Router(mode=Mode.RECORD_AND_REPLAY, storage=storage, outflow=mock_outflow)

        await router.route_request(
            method="POST",
            path="/detokenize",
            headers={"content-type": "application/json"},
            body={
                "model": "qwen3-4b-it",
                "tokens": [1, 2, 3]
            },
        )

        call_args = mock_outflow.forward_request.call_args
        forwarded_request = call_args[0][0]
        assert "stream" not in forwarded_request.body

    async def test_v1_completions_not_forced_to_stream(self, storage, mock_outflow):
        """Test that /v1/completions requests are NOT forced to stream upstream."""
        mock_outflow.forward_request = AsyncMock(return_value=CachedResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body={
                "id": "cmpl-1",
                "choices": [{
                    "text": "result",
                    "prompt_logprobs": [None, {
                        "1": {
                            "logprob": -0.5
                        }
                    }]
                }],
            },
            is_streaming=False,
        ))
        router = Router(mode=Mode.RECORD_AND_REPLAY, storage=storage, outflow=mock_outflow)

        await router.route_request(
            method="POST",
            path="/v1/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "qwen3-4b-it",
                "prompt": "Hello",
                "prompt_logprobs": 3,
                "max_tokens": 1
            },
        )

        call_args = mock_outflow.forward_request.call_args
        forwarded_request = call_args[0][0]
        assert forwarded_request.body.get("stream") is not True

    async def test_chat_completions_still_forced_to_stream(self, storage, mock_outflow):
        """Test that /v1/chat/completions requests ARE still forced to stream upstream."""
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
        assert forwarded_request.body["stream"] is True

    async def test_responses_api_forced_to_stream(self, storage, mock_outflow):
        """Test that /v1/responses requests ARE forced to stream upstream."""
        router = Router(mode=Mode.RECORD_AND_REPLAY, storage=storage, outflow=mock_outflow)

        await router.route_request(
            method="POST",
            path="/v1/responses",
            headers={"content-type": "application/json"},
            body={
                "model": "gpt-4",
                "input": "Hello"
            },
        )

        call_args = mock_outflow.forward_request.call_args
        forwarded_request = call_args[0][0]
        assert forwarded_request.body["stream"] is True

    async def test_tokenize_response_body_preserved(self, storage, mock_outflow):
        """Test that tokenize response body is preserved as-is (non-streaming JSON)."""
        expected_body = {"tokens": [100, 200, 300, 400]}
        mock_outflow.forward_request = AsyncMock(return_value=CachedResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=expected_body,
            is_streaming=False,
        ))
        router = Router(mode=Mode.RECORD_AND_REPLAY, storage=storage, outflow=mock_outflow)

        response = await router.route_request(
            method="POST",
            path="/tokenize",
            headers={"content-type": "application/json"},
            body={
                "model": "qwen3-4b-it",
                "prompt": "Hello world"
            },
        )

        assert response.status_code == 200
        assert response.is_streaming is False
        assert response.body == expected_body
