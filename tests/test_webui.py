"""Tests for WebUI server and API endpoints."""

import pytest
from aiohttp.test_utils import TestClient, TestServer

from inference_gate.modes import Mode
from inference_gate.recording.storage import CachedRequest, CachedResponse, CacheEntry, CacheStorage
from inference_gate.webui.server import WebUIServer


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture
def storage_with_entries(temp_cache_dir):
    """Create a CacheStorage instance with some test entries."""
    storage = CacheStorage(temp_cache_dir)

    # Add a couple of test entries
    request1 = CachedRequest(
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
    response1 = CachedResponse(
        status_code=200,
        headers={"content-type": "application/json"},
        body={
            "id": "chatcmpl-123",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hi!"
                }
            }]
        },
    )
    entry1 = CacheEntry(request=request1, response=response1, model="gpt-4", temperature=0.7)
    storage.put(entry1)

    request2 = CachedRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={"content-type": "application/json"},
        body={
            "model": "gpt-3.5-turbo",
            "messages": [{
                "role": "user",
                "content": "Test"
            }],
            "stream": True
        },
    )
    chunks = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
        'data: {"choices":[{"delta":{"content":" World"}}]}\n\n',
        "data: [DONE]\n\n",
    ]
    response2 = CachedResponse(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        chunks=chunks,
        is_streaming=True,
    )
    entry2 = CacheEntry(request=request2, response=response2, model="gpt-3.5-turbo", temperature=1.0)
    storage.put(entry2)

    return storage


@pytest.fixture
def webui_server(storage_with_entries, temp_cache_dir):
    """Create a WebUIServer instance."""
    return WebUIServer(
        host="127.0.0.1",
        port=0,
        storage=storage_with_entries,
        mode=Mode.RECORD_AND_REPLAY,
        cache_dir=temp_cache_dir,
        upstream_base_url="https://api.openai.com",
        proxy_host="127.0.0.1",
        proxy_port=8080,
    )


@pytest.fixture
async def webui_client(webui_server):
    """Create an aiohttp test client for the WebUI server."""
    app = webui_server._create_app()
    async with TestClient(TestServer(app)) as client:
        yield client


class TestWebUIAPI:
    """Tests for WebUI API endpoints."""

    async def test_get_cache_list(self, webui_client):
        """Test GET /api/cache returns list of cached entries."""
        resp = await webui_client.get("/api/cache")
        assert resp.status == 200
        data = await resp.json()
        assert isinstance(data, list)
        assert len(data) == 2

        # Check structure of returned entries
        entry = data[0]
        assert "id" in entry
        assert "model" in entry
        assert "path" in entry
        assert "method" in entry
        assert "status_code" in entry
        assert "is_streaming" in entry

    async def test_get_cache_entry_found(self, webui_client, storage_with_entries):
        """Test GET /api/cache/{id} returns entry details."""
        # Get the list first to get a valid ID
        entries = storage_with_entries.list_entries()
        entry_id = entries[0][0]

        resp = await webui_client.get(f"/api/cache/{entry_id}")
        assert resp.status == 200
        data = await resp.json()

        assert data["id"] == entry_id
        assert "request" in data
        assert "response" in data
        assert data["request"]["method"] == "POST"
        assert data["request"]["path"] == "/v1/chat/completions"
        assert data["response"]["status_code"] == 200

    async def test_get_cache_entry_not_found(self, webui_client):
        """Test GET /api/cache/{id} returns 404 for non-existent entry."""
        resp = await webui_client.get("/api/cache/nonexistent")
        assert resp.status == 404
        data = await resp.json()
        assert "error" in data

    async def test_get_stats(self, webui_client):
        """Test GET /api/stats returns cache statistics."""
        resp = await webui_client.get("/api/stats")
        assert resp.status == 200
        data = await resp.json()

        assert "total_entries" in data
        assert "total_size_bytes" in data
        assert "streaming_responses" in data
        assert "entries_by_model" in data

        assert data["total_entries"] == 2
        assert data["streaming_responses"] == 1
        assert "gpt-4" in data["entries_by_model"]
        assert "gpt-3.5-turbo" in data["entries_by_model"]

    async def test_get_config(self, webui_client):
        """Test GET /api/config returns current configuration."""
        resp = await webui_client.get("/api/config")
        assert resp.status == 200
        data = await resp.json()

        assert "mode" in data
        assert "upstream_url" in data
        assert "host" in data
        assert "port" in data
        assert "cache_dir" in data

        assert data["mode"] == "record-and-replay"
        assert data["upstream_url"] == "https://api.openai.com"
        assert data["host"] == "127.0.0.1"
        assert data["port"] == 8080


class TestWebUIServer:
    """Tests for WebUI server lifecycle."""

    async def test_server_start_stop(self, webui_server):
        """Test that WebUI server can start and stop cleanly."""
        await webui_server.start()
        assert webui_server._runner is not None
        assert webui_server._site is not None

        await webui_server.stop()
        assert webui_server._runner is None
        assert webui_server._site is None

    async def test_static_file_serving(self, webui_client):
        """Test that static files are served correctly."""
        # The index.html should be served at root
        resp = await webui_client.get("/")
        assert resp.status == 200
        # Should be HTML content
        text = await resp.text()
        assert "<!DOCTYPE html>" in text or "<html" in text


class TestWebUIEmptyCache:
    """Tests for WebUI with empty cache."""

    @pytest.fixture
    def empty_storage(self, temp_cache_dir):
        """Create an empty CacheStorage instance."""
        return CacheStorage(temp_cache_dir)

    @pytest.fixture
    def empty_webui_server(self, empty_storage, temp_cache_dir):
        """Create a WebUIServer with empty cache."""
        return WebUIServer(
            host="127.0.0.1",
            port=0,
            storage=empty_storage,
            mode=Mode.REPLAY_ONLY,
            cache_dir=temp_cache_dir,
            upstream_base_url=None,
            proxy_host="127.0.0.1",
            proxy_port=8080,
        )

    @pytest.fixture
    async def empty_webui_client(self, empty_webui_server):
        """Create an aiohttp test client for the empty WebUI server."""
        app = empty_webui_server._create_app()
        async with TestClient(TestServer(app)) as client:
            yield client

    async def test_get_cache_list_empty(self, empty_webui_client):
        """Test GET /api/cache with empty cache returns empty list."""
        resp = await empty_webui_client.get("/api/cache")
        assert resp.status == 200
        data = await resp.json()
        assert isinstance(data, list)
        assert len(data) == 0

    async def test_get_stats_empty(self, empty_webui_client):
        """Test GET /api/stats with empty cache."""
        resp = await empty_webui_client.get("/api/stats")
        assert resp.status == 200
        data = await resp.json()

        assert data["total_entries"] == 0
        assert data["total_size_bytes"] >= 0
        assert data["streaming_responses"] == 0
        assert data["entries_by_model"] == {}
