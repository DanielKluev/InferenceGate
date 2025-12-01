"""Tests for InferenceReplay storage module."""

import tempfile
from pathlib import Path

import pytest

from inference_replay.storage import (
    CachedRequest,
    CachedResponse,
    CacheEntry,
    CacheStorage,
)


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
        """Test that init creates the cache directory."""
        cache_dir = Path(temp_cache_dir) / "new_cache"
        CacheStorage(cache_dir)  # Side-effect: creates directory
        assert cache_dir.exists()

    def test_put_and_get(self, storage):
        """Test storing and retrieving a cache entry."""
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
        )
        response = CachedResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body={"choices": [{"message": {"content": "Hi there!"}}]},
        )
        entry = CacheEntry(
            request=request,
            response=response,
            model="gpt-4",
            temperature=0.7,
        )

        cache_key = storage.put(entry)
        assert cache_key is not None

        retrieved = storage.get(request)
        assert retrieved is not None
        assert retrieved.model == "gpt-4"
        assert retrieved.response.status_code == 200
        assert retrieved.response.body == response.body

    def test_get_nonexistent(self, storage):
        """Test getting a non-existent entry."""
        request = CachedRequest(
            method="GET",
            path="/v1/models",
            headers={},
        )
        result = storage.get(request)
        assert result is None

    def test_exists(self, storage):
        """Test exists method."""
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={"model": "gpt-4"},
        )

        assert not storage.exists(request)

        response = CachedResponse(status_code=200, headers={}, body={})
        entry = CacheEntry(request=request, response=response)
        storage.put(entry)

        assert storage.exists(request)

    def test_clear(self, storage):
        """Test clearing cache."""
        # Add some entries
        for i in range(3):
            request = CachedRequest(
                method="POST",
                path="/v1/chat/completions",
                headers={},
                body={"model": f"model-{i}"},
            )
            response = CachedResponse(status_code=200, headers={}, body={})
            entry = CacheEntry(request=request, response=response)
            storage.put(entry)

        count = storage.clear()
        assert count == 3
        assert storage.list_entries() == []

    def test_list_entries(self, storage):
        """Test listing cache entries."""
        # Add entries
        for i in range(2):
            request = CachedRequest(
                method="POST",
                path=f"/v1/path-{i}",
                headers={},
                body={},
            )
            response = CachedResponse(status_code=200, headers={}, body={})
            entry = CacheEntry(request=request, response=response)
            storage.put(entry)

        entries = storage.list_entries()
        assert len(entries) == 2

    def test_compute_prompt_hash(self):
        """Test prompt hash computation."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        hash1 = CacheStorage.compute_prompt_hash(messages)
        hash2 = CacheStorage.compute_prompt_hash(messages)
        assert hash1 == hash2

        different_messages = [
            {"role": "user", "content": "Different"},
        ]
        hash3 = CacheStorage.compute_prompt_hash(different_messages)
        assert hash1 != hash3

    def test_streaming_response(self, storage):
        """Test caching streaming responses."""
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={"model": "gpt-4", "stream": True},
        )
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
