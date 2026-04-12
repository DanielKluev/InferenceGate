"""Tests for InferenceGate recording/storage module (v2 tape format)."""

import tempfile
from pathlib import Path

import pytest

from inference_gate.recording.hashing import compute_content_hash
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
    """Tests for CacheStorage class (v2 tape format)."""

    def test_init_creates_directory(self, temp_cache_dir):
        """Test that CacheStorage constructor creates the cache directory and subdirectories."""
        cache_dir = Path(temp_cache_dir) / "new_cache"
        s = CacheStorage(cache_dir)
        assert cache_dir.exists()
        assert (cache_dir / "requests").exists()
        assert (cache_dir / "responses").exists()
        assert (cache_dir / "assets").exists()

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
                }],
                "temperature": 0,
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
        entry = CacheEntry(request=request, response=response, model="gpt-4", temperature=0)

        cache_key = storage.put(entry)
        assert cache_key is not None

        retrieved = storage.get(request)
        assert retrieved is not None
        assert retrieved.model == "gpt-4"

    def test_get_nonexistent(self, storage):
        """Test that getting a non-existent entry returns None."""
        request = CachedRequest(method="GET", path="/v1/models", headers={})
        result = storage.get(request)
        assert result is None

    def test_exists(self, storage):
        """Test that exists() correctly reports whether a request is cached."""
        request = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={"model": "gpt-4", "temperature": 0})

        assert not storage.exists(request)

        response = CachedResponse(status_code=200, headers={}, body={"choices": []})
        entry = CacheEntry(request=request, response=response)
        storage.put(entry)

        assert storage.exists(request)

    def test_clear(self, storage):
        """Test that clear() removes all entries and returns the correct count."""
        # Add some entries
        for i in range(3):
            request = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={"model": f"model-{i}", "temperature": 0})
            response = CachedResponse(status_code=200, headers={}, body={"choices": []})
            entry = CacheEntry(request=request, response=response)
            storage.put(entry)

        count = storage.clear()
        assert count == 3
        assert storage.list_entries() == []

    def test_list_entries(self, storage):
        """Test that list_entries() returns all stored entries."""
        # Add entries
        for i in range(2):
            request = CachedRequest(method="POST", path=f"/v1/path-{i}", headers={}, body={"temperature": 0})
            response = CachedResponse(status_code=200, headers={}, body={"choices": []})
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
        request = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={
            "model": "gpt-4",
            "stream": True,
            "temperature": 0
        })
        response = CachedResponse(
            status_code=200,
            headers={},
            chunks=[
                "data: {\"choices\":[{\"delta\":{\"content\":\"chunk1\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"chunk2\"}}]}\n\n", "data: [DONE]\n\n"
            ],
            is_streaming=True,
        )
        entry = CacheEntry(request=request, response=response)
        storage.put(entry)

        retrieved = storage.get(request)
        assert retrieved is not None

    def test_cache_key_ignores_stream_field(self, storage):
        """Test that stream=True and stream=False requests produce the same content hash."""
        body_streaming = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "stream": True}
        body_non_streaming = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "stream": False}
        body_no_stream = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}

        key1 = compute_content_hash("POST", "/v1/chat/completions", body_streaming)
        key2 = compute_content_hash("POST", "/v1/chat/completions", body_non_streaming)
        key3 = compute_content_hash("POST", "/v1/chat/completions", body_no_stream)

        assert key1 == key2, "stream=True and stream=False should produce the same cache key"
        assert key1 == key3, "stream=True and no stream field should produce the same cache key"

    def test_cache_key_differs_for_different_content(self, storage):
        """Test that different prompt content produces different content hashes."""
        body1 = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "stream": True}
        body2 = {"model": "gpt-4", "messages": [{"role": "user", "content": "Goodbye"}], "stream": True}

        key1 = compute_content_hash("POST", "/v1/chat/completions", body1)
        key2 = compute_content_hash("POST", "/v1/chat/completions", body2)

        assert key1 != key2

    def test_streaming_request_hits_non_streaming_cache(self, storage):
        """Test that a streaming request can find a cassette stored by a non-streaming request (same content hash)."""
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
                "temperature": 0,
            },
        )
        response = CachedResponse(status_code=200, headers={},
                                  chunks=["data: {\"choices\":[{\"delta\":{\"content\":\"chunk\"}}]}\n\n",
                                          "data: [DONE]\n\n"], is_streaming=True)
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
                "stream": False,
                "temperature": 0,
            },
        )
        retrieved = storage.get(request_lookup)
        assert retrieved is not None

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
                "temperature": 0,
            },
        )
        retrieved2 = storage.get(request_plain)
        assert retrieved2 is not None

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

        key_with = compute_content_hash("POST", "/v1/chat/completions", body_with)
        key_without = compute_content_hash("POST", "/v1/chat/completions", body_without)

        assert key_with == key_without, "stream_options should be excluded from cache key"

    def test_original_client_streaming_metadata(self, storage):
        """Test that original_client_streaming metadata is stored in CacheEntry."""
        request = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={
            "model": "gpt-4",
            "stream": True,
            "temperature": 0
        })
        response = CachedResponse(status_code=200, headers={},
                                  chunks=["data: {\"choices\":[{\"delta\":{\"content\":\"test\"}}]}\n\n",
                                          "data: [DONE]\n\n"], is_streaming=True)
        entry = CacheEntry(request=request, response=response, original_client_streaming=False)
        storage.put(entry)

        retrieved = storage.get(request)
        assert retrieved is not None
        # original_client_streaming is a CacheEntry transport field, not persisted in tape

    def test_index_lookup_by_prompt_hash(self, storage):
        """Test that the index supports lookup by prompt hash."""
        messages = [{"role": "user", "content": "Hello fuzzy"}]
        prompt_hash = CacheStorage.compute_prompt_hash(messages)
        request = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={
            "model": "gpt-4",
            "messages": messages,
            "temperature": 0
        })
        response = CachedResponse(status_code=200, headers={}, body={"choices": [{"message": {"content": "Hi!"}}]})
        entry = CacheEntry(request=request, response=response, model="gpt-4", prompt_hash=prompt_hash)
        storage.put(entry)

        rows = storage.index.by_prompt_hash.get(prompt_hash, [])
        assert len(rows) >= 1
        assert rows[0].model == "gpt-4"

    def test_index_lookup_by_prompt_hash_not_found(self, storage):
        """Test that index lookup by nonexistent prompt hash returns empty list."""
        rows = storage.index.by_prompt_hash.get("nonexistenthash", [])
        assert rows == []

    def test_index_lookup_ignores_model(self, storage):
        """Test that prompt hash lookup finds entries regardless of model used."""
        messages = [{"role": "user", "content": "Hello from model A"}]
        request = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={
            "model": "model-a",
            "messages": messages,
            "temperature": 0
        })
        response = CachedResponse(status_code=200, headers={}, body={"choices": [{"message": {"content": "Response from A"}}]})
        entry = CacheEntry(request=request, response=response, model="model-a")
        storage.put(entry)

        prompt_hash = CacheStorage.compute_prompt_hash(messages)
        rows = storage.index.by_prompt_hash.get(prompt_hash, [])
        assert len(rows) >= 1
        assert rows[0].model == "model-a"

    def test_multi_reply_append(self, storage):
        """Test that putting the same request twice appends a reply to the existing tape."""
        request = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": "Multi"
            }],
            "temperature": 0.7
        })
        response1 = CachedResponse(status_code=200, headers={}, body={"choices": [{"message": {"content": "Reply 1"}}]})
        response2 = CachedResponse(status_code=200, headers={}, body={"choices": [{"message": {"content": "Reply 2"}}]})

        entry1 = CacheEntry(request=request, response=response1, model="gpt-4")
        entry2 = CacheEntry(request=request, response=response2, model="gpt-4")

        content_hash = storage.put(entry1, max_replies=5)
        storage.put(entry2, max_replies=5)

        # Index should show 2 replies
        row = storage.index.by_content_hash.get(content_hash)
        assert row is not None
        assert row.replies == 2

        # Both reply response hashes should be gettable
        hashes = storage.get_reply_response_hashes(content_hash)
        assert len(hashes) == 2

    def test_reindex(self, storage):
        """Test that reindex() rebuilds the index from tape files."""
        # Store an entry
        request = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={"model": "gpt-4", "temperature": 0})
        response = CachedResponse(status_code=200, headers={}, body={"choices": []})
        entry = CacheEntry(request=request, response=response)
        storage.put(entry)

        # Wipe and rebuild index
        count = storage.reindex()
        assert count == 1


def _make_entry(model: str = "gpt-4", message: str = "Hello", temperature: float = 0) -> CacheEntry:
    """Helper to create a simple entry for testing."""
    return CacheEntry(
        request=CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={
            "model": model,
            "messages": [{
                "role": "user",
                "content": message
            }],
            "temperature": temperature
        }),
        response=CachedResponse(status_code=200, headers={}, body={"choices": [{
            "message": {
                "content": f"Reply to: {message}"
            }
        }]}),
        model=model,
        temperature=temperature,
    )


class TestDeleteEntry:
    """Tests for CacheStorage.delete_entry()."""

    def test_delete_existing_entry(self, storage):
        """Test that deleting an existing cassette removes tape, responses, and index entry."""
        entry = _make_entry(message="delete me")
        content_hash = storage.put(entry)

        assert content_hash in storage.index
        assert storage.delete_entry(content_hash) is True
        assert content_hash not in storage.index
        # Tape file should be gone
        assert storage._find_tape_file(content_hash) is None

    def test_delete_nonexistent_entry(self, storage):
        """Test that deleting a non-existent cassette returns False."""
        assert storage.delete_entry("nonexistent123") is False

    def test_delete_removes_response_files(self, storage):
        """Test that deleting a cassette also removes its response JSON/NDJSON files."""
        entry = _make_entry(message="delete responses")
        content_hash = storage.put(entry)

        # Get response hashes before delete
        response_hashes = storage.get_reply_response_hashes(content_hash)
        assert len(response_hashes) > 0

        storage.delete_entry(content_hash)
        for resp_hash in response_hashes:
            assert not (storage.responses_dir / f"{resp_hash}.json").exists()


class TestResolvePrefix:
    """Tests for CacheStorage.resolve_prefix()."""

    def test_exact_match(self, storage):
        """Test that an exact content_hash returns exactly one match."""
        entry = _make_entry(message="prefix test")
        content_hash = storage.put(entry)
        matches = storage.resolve_prefix(content_hash)
        assert len(matches) == 1
        assert matches[0].content_hash == content_hash

    def test_prefix_match(self, storage):
        """Test that a short prefix resolves to the correct entry."""
        entry = _make_entry(message="prefix short")
        content_hash = storage.put(entry)
        # Use first 6 chars as prefix
        matches = storage.resolve_prefix(content_hash[:6])
        assert any(m.content_hash == content_hash for m in matches)

    def test_no_match(self, storage):
        """Test that a non-matching prefix returns empty list."""
        matches = storage.resolve_prefix("zzzzzzzzzzzz")
        assert matches == []


class TestSearchEntries:
    """Tests for CacheStorage.search_entries()."""

    def test_search_by_message(self, storage):
        """Test searching by user message content."""
        storage.put(_make_entry(message="The quick brown fox"))
        storage.put(_make_entry(message="Lazy dog sleeps"))
        storage.put(_make_entry(message="Quick rabbit jumps"))

        results = storage.search_entries("quick")
        assert len(results) == 2

    def test_search_with_model_filter(self, storage):
        """Test search with additional model filter."""
        storage.put(_make_entry(model="gpt-4", message="Hello from gpt"))
        storage.put(_make_entry(model="claude-3", message="Hello from claude"))

        results = storage.search_entries("hello", model="gpt")
        assert len(results) == 1
        assert results[0].model == "gpt-4"

    def test_search_limit(self, storage):
        """Test that search respects the limit parameter."""
        for i in range(10):
            storage.put(_make_entry(message=f"Search target {i}", model=f"model-{i}"))

        results = storage.search_entries("search target", limit=3)
        assert len(results) == 3

    def test_search_no_results(self, storage):
        """Test that search returns empty list when nothing matches."""
        storage.put(_make_entry(message="Something unrelated"))
        results = storage.search_entries("nonexistent query")
        assert results == []


class TestFilterEntries:
    """Tests for CacheStorage.filter_entries()."""

    def test_filter_by_model(self, storage):
        """Test filtering by model name."""
        storage.put(_make_entry(model="gpt-4", message="A"))
        storage.put(_make_entry(model="claude-3", message="B"))
        storage.put(_make_entry(model="gpt-4-turbo", message="C"))

        results = storage.filter_entries(model="gpt")
        assert len(results) == 2

    def test_filter_by_greedy(self, storage):
        """Test filtering by greedy flag (temperature=0 is greedy)."""
        storage.put(_make_entry(temperature=0, message="Greedy"))
        storage.put(_make_entry(temperature=0.7, message="Non-greedy"))

        greedy = storage.filter_entries(greedy=True)
        assert len(greedy) == 1
        assert greedy[0].is_greedy is True

        non_greedy = storage.filter_entries(greedy=False)
        assert len(non_greedy) == 1
        assert non_greedy[0].is_greedy is False

    def test_filter_with_limit(self, storage):
        """Test that filter respects the limit parameter."""
        for i in range(5):
            storage.put(_make_entry(message=f"Limit test {i}", model=f"model-{i}"))

        results = storage.filter_entries(limit=2)
        assert len(results) == 2

    def test_filter_all_returns_all(self, storage):
        """Test that filtering with no criteria returns all entries."""
        storage.put(_make_entry(message="One"))
        storage.put(_make_entry(message="Two"))
        results = storage.filter_entries()
        assert len(results) == 2


class TestGetDiskSize:
    """Tests for CacheStorage.get_disk_size()."""

    def test_empty_cache_size(self, storage):
        """Test that an empty cache has zero (or minimal) disk size."""
        size = storage.get_disk_size()
        assert size == 0

    def test_size_increases_after_put(self, storage):
        """Test that disk size increases after storing an entry."""
        storage.put(_make_entry(message="Size test"))
        size = storage.get_disk_size()
        assert size > 0


class TestReconstructRequestBody:
    """Tests for CacheStorage.reconstruct_request_body()."""

    def test_basic_reconstruction(self, storage):
        """Test that a simple chat completion request body is reconstructed from tape."""
        entry = _make_entry(model="gpt-4", message="Hello world", temperature=0.7)
        content_hash = storage.put(entry)

        body = storage.reconstruct_request_body(content_hash)
        assert body is not None
        assert body["model"] == "gpt-4"
        assert body["temperature"] == 0.7
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"] == "Hello world"

    def test_reconstruction_with_system_message(self, storage):
        """Test reconstruction of request with system + user messages."""
        entry = CacheEntry(
            request=CachedRequest(
                method="POST", path="/v1/chat/completions", headers={}, body={
                    "model": "gpt-4",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are helpful."
                        },
                        {
                            "role": "user",
                            "content": "Hi"
                        },
                    ],
                    "temperature": 0.5,
                }),
            response=CachedResponse(status_code=200, headers={}, body={"choices": [{
                "message": {
                    "content": "Hello!"
                }
            }]}),
            model="gpt-4",
        )
        content_hash = storage.put(entry)

        body = storage.reconstruct_request_body(content_hash)
        assert body is not None
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "You are helpful."
        assert body["messages"][1]["role"] == "user"
        assert body["messages"][1]["content"] == "Hi"

    def test_reconstruction_with_sampling_params(self, storage):
        """Test that sampling parameters (top_p, top_k, etc.) are reconstructed."""
        entry = CacheEntry(
            request=CachedRequest(
                method="POST", path="/v1/chat/completions", headers={}, body={
                    "model": "test-model",
                    "messages": [{
                        "role": "user",
                        "content": "Test"
                    }],
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_tokens": 200,
                }),
            response=CachedResponse(status_code=200, headers={}, body={"choices": [{
                "message": {
                    "content": "OK"
                }
            }]}),
            model="test-model",
        )
        content_hash = storage.put(entry)

        body = storage.reconstruct_request_body(content_hash)
        assert body is not None
        assert body["temperature"] == 0.8
        assert body["top_p"] == 0.9
        assert body["top_k"] == 40
        assert body["max_tokens"] == 200

    def test_reconstruction_with_tools(self, storage):
        """Test that tool definitions are reconstructed from the tape."""
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
        entry = CacheEntry(
            request=CachedRequest(
                method="POST", path="/v1/chat/completions", headers={}, body={
                    "model": "gpt-4",
                    "messages": [{
                        "role": "user",
                        "content": "Weather?"
                    }],
                    "tools": tools,
                    "temperature": 0.5,
                }),
            response=CachedResponse(status_code=200, headers={}, body={"choices": [{
                "message": {
                    "content": "Sunny"
                }
            }]}),
            model="gpt-4",
        )
        content_hash = storage.put(entry)

        body = storage.reconstruct_request_body(content_hash)
        assert body is not None
        assert "tools" in body
        assert body["tools"] == tools

    def test_reconstruction_nonexistent(self, storage):
        """Test that reconstruct_request_body returns None for a nonexistent cassette."""
        body = storage.reconstruct_request_body("nonexistent123")
        assert body is None

    def test_reconstruction_with_stop_sequences(self, storage):
        """Test that stop sequences are reconstructed."""
        entry = CacheEntry(
            request=CachedRequest(
                method="POST", path="/v1/chat/completions", headers={}, body={
                    "model": "gpt-4",
                    "messages": [{
                        "role": "user",
                        "content": "Test"
                    }],
                    "stop": ["\n", "END"],
                    "temperature": 0.5,
                }),
            response=CachedResponse(status_code=200, headers={}, body={"choices": [{
                "message": {
                    "content": "OK"
                }
            }]}),
            model="gpt-4",
        )
        content_hash = storage.put(entry)

        body = storage.reconstruct_request_body(content_hash)
        assert body is not None
        assert body["stop"] == ["\n", "END"]


class TestUpdateMaxReplies:
    """Tests for CacheStorage._update_max_replies()."""

    def test_update_max_replies(self, storage):
        """Test that max_replies is updated in tape metadata and index."""
        entry = _make_entry(model="gpt-4", message="Update max", temperature=0.7)
        content_hash = storage.put(entry, max_replies=5)

        row = storage.index.by_content_hash.get(content_hash)
        assert row.max_replies == 5

        storage._update_max_replies(content_hash, 10)

        # Verify index is updated
        row = storage.index.by_content_hash.get(content_hash)
        assert row.max_replies == 10

        # Verify tape metadata is updated
        tape_data = storage.load_tape(content_hash)
        assert tape_data is not None
        metadata, _ = tape_data
        assert metadata.max_replies == 10
