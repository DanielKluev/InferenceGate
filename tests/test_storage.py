"""Tests for InferenceGate recording/storage module."""

import tempfile
from pathlib import Path

import pytest

from inference_gate.recording.storage import CachedRequest, CachedResponse, CacheEntry, CacheStorage


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def storage(temp_cache_dir):
    """Create a cache storage instance."""
    return CacheStorage(temp_cache_dir)


class TestCacheStorage:
    """Tests for CacheStorage class."""

    def test_init_creates_directory(self, temp_cache_dir):
        """Test that CacheStorage constructor creates the cache directory if it doesn't exist."""
        cache_dir = Path(temp_cache_dir) / "new_cache"
        CacheStorage(cache_dir)  # Side-effect: creates directory
        assert cache_dir.exists()

    def test_put_and_get(self, storage):
        """Test that storing an entry and retrieving it returns the same data."""
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
        response = CachedResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body={"choices": [{
                "message": {
                    "content": "Hi there!"
                }
            }]},
        )
        entry = CacheEntry(request=request, response=response, model="gpt-4", temperature=0.7)

        cache_key = storage.put(entry)
        assert cache_key is not None

        retrieved = storage.get(request)
        assert retrieved is not None
        assert retrieved.model == "gpt-4"
        assert retrieved.response.status_code == 200
        assert retrieved.response.body == response.body

    def test_get_nonexistent(self, storage):
        """Test that getting a non-existent entry returns None."""
        request = CachedRequest(method="GET", path="/v1/models", headers={})
        result = storage.get(request)
        assert result is None

    def test_exists(self, storage):
        """Test that exists() correctly reports whether a request is cached."""
        request = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={"model": "gpt-4"})

        assert not storage.exists(request)

        response = CachedResponse(status_code=200, headers={}, body={})
        entry = CacheEntry(request=request, response=response)
        storage.put(entry)

        assert storage.exists(request)

    def test_clear(self, storage):
        """Test that clear() removes all entries and returns the correct count."""
        # Add some entries
        for i in range(3):
            request = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={"model": f"model-{i}"})
            response = CachedResponse(status_code=200, headers={}, body={})
            entry = CacheEntry(request=request, response=response)
            storage.put(entry)

        count = storage.clear()
        assert count == 3
        assert storage.list_entries() == []

    def test_list_entries(self, storage):
        """Test that list_entries() returns all stored entries."""
        # Add entries
        for i in range(2):
            request = CachedRequest(method="POST", path=f"/v1/path-{i}", headers={}, body={})
            response = CachedResponse(status_code=200, headers={}, body={})
            entry = CacheEntry(request=request, response=response)
            storage.put(entry)

        entries = storage.list_entries()
        assert len(entries) == 2

    def test_compute_prompt_hash(self):
        """Test that prompt hash is deterministic and different for different inputs."""
        messages = [
            {
                "role": "system",
                "content": "You are helpful."
            },
            {
                "role": "user",
                "content": "Hello"
            },
        ]
        hash1 = CacheStorage.compute_prompt_hash(messages)
        hash2 = CacheStorage.compute_prompt_hash(messages)
        assert hash1 == hash2

        different_messages = [{"role": "user", "content": "Different"}]
        hash3 = CacheStorage.compute_prompt_hash(different_messages)
        assert hash1 != hash3

    def test_streaming_response(self, storage):
        """Test that streaming responses with chunks are stored and retrieved correctly."""
        request = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={"model": "gpt-4", "stream": True})
        response = CachedResponse(
            status_code=200,
            headers={},
            chunks=["data: chunk1\n\n", "data: chunk2\n\n", "data: [DONE]\n\n"],
            is_streaming=True,
        )
        entry = CacheEntry(request=request, response=response)
        storage.put(entry)

        retrieved = storage.get(request)
        assert retrieved is not None
        assert retrieved.response.is_streaming
        assert len(retrieved.response.chunks) == 3

    def test_cache_key_ignores_stream_field(self, storage):
        """Test that stream=True and stream=False requests share the same cache key."""
        body_streaming = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "stream": True}
        body_non_streaming = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "stream": False}
        body_no_stream = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}

        req_streaming = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body=body_streaming)
        req_non_streaming = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body=body_non_streaming)
        req_no_stream = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body=body_no_stream)

        key1 = storage._compute_cache_key(req_streaming)
        key2 = storage._compute_cache_key(req_non_streaming)
        key3 = storage._compute_cache_key(req_no_stream)

        assert key1 == key2, "stream=True and stream=False should produce the same cache key"
        assert key1 == key3, "stream=True and no stream field should produce the same cache key"

    def test_cache_key_differs_for_different_content(self, storage):
        """Test that different prompt content produces different cache keys even after stream stripping."""
        body1 = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "stream": True}
        body2 = {"model": "gpt-4", "messages": [{"role": "user", "content": "Goodbye"}], "stream": True}

        req1 = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body=body1)
        req2 = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body=body2)

        assert storage._compute_cache_key(req1) != storage._compute_cache_key(req2)

    def test_streaming_request_hits_non_streaming_cache(self, storage):
        """Test that a streaming request can find a cassette stored by a non-streaming request (same cache key)."""
        # Store with stream=True and stream_options (as recorded by forced-streaming router)
        request_stored = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "Test"
                }],
                "stream": True,
                "stream_options": {
                    "include_usage": True
                },
            },
        )
        response = CachedResponse(status_code=200, headers={}, chunks=["data: chunk\n\n"], is_streaming=True)
        entry = CacheEntry(request=request_stored, response=response)
        storage.put(entry)

        # Look up with stream=False body (as the non-streaming client would send)
        request_lookup = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "Test"
                }],
                "stream": False
            },
        )
        retrieved = storage.get(request_lookup)
        assert retrieved is not None
        assert retrieved.response.is_streaming

        # Also look up with NO stream field at all (as a plain client would send)
        request_plain = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "Test"
                }],
            },
        )
        retrieved2 = storage.get(request_plain)
        assert retrieved2 is not None
        assert retrieved2.response.is_streaming

    def test_cache_key_ignores_stream_options_field(self, storage):
        """Test that stream_options field is excluded from cache key computation."""
        body_with = {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": "Hello"
            }],
            "stream": True,
            "stream_options": {
                "include_usage": True
            },
        }
        body_without = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}

        req_with = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body=body_with)
        req_without = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body=body_without)

        key_with = storage._compute_cache_key(req_with)
        key_without = storage._compute_cache_key(req_without)

        assert key_with == key_without, "stream_options should be excluded from cache key"

    def test_original_client_streaming_metadata(self, storage):
        """Test that original_client_streaming metadata is stored and retrieved correctly."""
        request = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={"model": "gpt-4", "stream": True})
        response = CachedResponse(status_code=200, headers={}, chunks=["data: test\n\n"], is_streaming=True)
        entry = CacheEntry(request=request, response=response, original_client_streaming=False)
        storage.put(entry)

        retrieved = storage.get(request)
        assert retrieved is not None
        assert retrieved.original_client_streaming is False
