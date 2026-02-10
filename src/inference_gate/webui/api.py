"""
API endpoint handlers for the WebUI Dashboard.

Provides JSON API endpoints for:
- Listing cached entries
- Getting entry details
- Cache statistics
- Current configuration
"""

import logging
import os
from pathlib import Path

from aiohttp import web

from inference_gate.modes import Mode
from inference_gate.recording.storage import CacheStorage


class WebUIAPI:
    """
    API handler class for WebUI endpoints.

    Provides methods to fetch cache data, statistics, and configuration.
    """

    def __init__(self, storage: CacheStorage, mode: Mode, cache_dir: str, upstream_base_url: str | None, host: str, port: int) -> None:
        """
        Initialize the WebUI API handler.

        `storage` is the CacheStorage instance for accessing cached data.
        `mode` is the current operation mode (RECORD_AND_REPLAY or REPLAY_ONLY).
        `cache_dir` is the directory where cache files are stored.
        `upstream_base_url` is the upstream API URL (None in REPLAY_ONLY mode).
        `host` and `port` are the proxy server configuration.
        """
        self.log = logging.getLogger("WebUIAPI")
        self.storage = storage
        self.mode = mode
        self.cache_dir = cache_dir
        self.upstream_base_url = upstream_base_url
        self.host = host
        self.port = port

    async def get_cache_list(self, request: web.Request) -> web.Response:
        """
        Handle GET /api/cache - list all cached entries.

        Returns a JSON list of all cached entries with summary information.
        """
        try:
            entries = self.storage.list_entries()
            result = []
            for cache_key, entry in entries:
                result.append({
                    "id": cache_key,
                    "model": entry.model,
                    "path": entry.request.path,
                    "method": entry.request.method,
                    "status_code": entry.response.status_code,
                    "is_streaming": entry.response.is_streaming,
                    "temperature": entry.temperature,
                    "prompt_hash": entry.prompt_hash,
                })
            self.log.debug("Returning %d cache entries", len(result))
            return web.json_response(result)
        except Exception as e:
            self.log.error("Error listing cache entries: %s", e)
            return web.json_response({"error": str(e)}, status=500)

    async def get_cache_entry(self, request: web.Request) -> web.Response:
        """
        Handle GET /api/cache/{entry_id} - get detailed entry information.

        Returns full request and response details for a cached entry.
        """
        entry_id = request.match_info.get("entry_id")
        if not entry_id:
            return web.json_response({"error": "Missing entry_id"}, status=400)

        try:
            # Find the entry by cache key
            entries = self.storage.list_entries()
            for cache_key, entry in entries:
                if cache_key == entry_id:
                    result = {
                        "id": cache_key,
                        "model": entry.model,
                        "temperature": entry.temperature,
                        "prompt_hash": entry.prompt_hash,
                        "request": {
                            "method": entry.request.method,
                            "path": entry.request.path,
                            "headers": entry.request.headers,
                            "body": entry.request.body,
                            "query_params": entry.request.query_params,
                        },
                        "response": {
                            "status_code": entry.response.status_code,
                            "headers": entry.response.headers,
                            "body": entry.response.body,
                            "chunks": entry.response.chunks,
                            "is_streaming": entry.response.is_streaming,
                        }
                    }
                    self.log.debug("Returning cache entry: %s", entry_id)
                    return web.json_response(result)

            # Entry not found
            return web.json_response({"error": "Entry not found"}, status=404)
        except Exception as e:
            self.log.error("Error getting cache entry %s: %s", entry_id, e)
            return web.json_response({"error": str(e)}, status=500)

    async def get_stats(self, request: web.Request) -> web.Response:
        """
        Handle GET /api/stats - get cache statistics.

        Returns statistics about the cache: total entries, size, entries by model, etc.
        """
        try:
            entries = self.storage.list_entries()
            total_entries = len(entries)

            # Calculate statistics
            models: dict[str, int] = {}
            streaming_count = 0
            total_size = 0

            for cache_key, entry in entries:
                # Count by model
                if entry.model:
                    models[entry.model] = models.get(entry.model, 0) + 1

                # Count streaming responses
                if entry.response.is_streaming:
                    streaming_count += 1

            # Calculate total cache directory size
            cache_path = Path(self.cache_dir)
            if cache_path.exists():
                total_size = sum(f.stat().st_size for f in cache_path.glob("*.json"))

            result = {
                "total_entries": total_entries,
                "total_size_bytes": total_size,
                "streaming_responses": streaming_count,
                "entries_by_model": models,
            }

            self.log.debug("Returning cache stats: %d entries", total_entries)
            return web.json_response(result)
        except Exception as e:
            self.log.error("Error getting cache stats: %s", e)
            return web.json_response({"error": str(e)}, status=500)

    async def get_config(self, request: web.Request) -> web.Response:
        """
        Handle GET /api/config - get current configuration.

        Returns current InferenceGate configuration.
        """
        try:
            result = {
                "mode": self.mode.value,
                "upstream_url": self.upstream_base_url,
                "host": self.host,
                "port": self.port,
                "cache_dir": self.cache_dir,
            }
            self.log.debug("Returning config")
            return web.json_response(result)
        except Exception as e:
            self.log.error("Error getting config: %s", e)
            return web.json_response({"error": str(e)}, status=500)
