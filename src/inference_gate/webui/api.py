"""
API endpoint handlers for the WebUI Dashboard.

Provides JSON API endpoints for:
- Listing cached entries (from TSV index)
- Getting entry details (from tape files)
- Cache statistics
- Current configuration
"""

import logging
from pathlib import Path

from aiohttp import web

from inference_gate.modes import Mode
from inference_gate.recording.storage import CacheStorage


class WebUIAPI:
    """
    API handler class for WebUI endpoints.

    Provides methods to fetch cache data, statistics, and configuration.
    Uses the TapeIndex for fast listing and tape files for entry details.
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

        Returns a JSON list of all cached entries from the TSV index with summary information.
        """
        try:
            rows = list(self.storage.index.by_content_hash.values())
            result = []
            for row in rows:
                result.append({
                    "id": row.content_hash,
                    "model": row.model,
                    "is_greedy": row.is_greedy,
                    "temperature": row.temperature,
                    "replies": row.replies,
                    "max_replies": row.max_replies,
                    "has_logprobs": row.has_logprobs,
                    "has_tool_use": row.has_tool_use,
                    "slug": row.slug,
                    "recorded": row.recorded,
                    "first_user_message": row.first_user_message,
                    "tokens_in": row.tokens_in,
                    "tokens_out": row.tokens_out,
                })
            self.log.debug("Returning %d cache entries", len(result))
            return web.json_response(result)
        except Exception as e:
            self.log.error("Error listing cache entries: %s", e, exc_info=True)
            return web.json_response({"error": "Failed to list cache entries"}, status=500)

    async def get_cache_entry(self, request: web.Request) -> web.Response:
        """
        Handle GET /api/cache/{entry_id} - get detailed entry information.

        Loads the tape file for the given content_hash and returns full details.
        """
        entry_id = request.match_info.get("entry_id")
        if not entry_id:
            return web.json_response({"error": "Missing entry_id"}, status=400)

        try:
            # Look up in index first
            index_row = self.storage.index.by_content_hash.get(entry_id)
            if index_row is None:
                return web.json_response({"error": "Entry not found"}, status=404)

            # Load the tape file for full details
            tape_data = self.storage.load_tape(entry_id)
            if tape_data is None:
                return web.json_response({"error": "Tape file not found"}, status=404)

            metadata, sections = tape_data
            result = {
                "id": entry_id,
                "model": metadata.model,
                "endpoint": metadata.endpoint,
                "sampling": metadata.sampling.model_dump(),
                "content_hash": metadata.content_hash,
                "prompt_model_hash": metadata.prompt_model_hash,
                "prompt_hash": metadata.prompt_hash,
                "recorded": metadata.recorded.isoformat() if metadata.recorded else None,
                "replies": metadata.replies,
                "max_replies": metadata.max_replies,
                "sections": [{"kind": s.kind.value, "header": s.header, "body": s.body} for s in sections],
                "index": {
                    "is_greedy": index_row.is_greedy,
                    "temperature": index_row.temperature,
                    "tokens_in": index_row.tokens_in,
                    "tokens_out": index_row.tokens_out,
                    "has_logprobs": index_row.has_logprobs,
                    "has_tool_use": index_row.has_tool_use,
                    "first_user_message": index_row.first_user_message,
                },
            }
            self.log.debug("Returning cache entry: %s", entry_id)
            return web.json_response(result)
        except Exception as e:
            self.log.error("Error getting cache entry %s: %s", entry_id, e, exc_info=True)
            return web.json_response({"error": "Failed to get cache entry"}, status=500)

    async def get_stats(self, request: web.Request) -> web.Response:
        """
        Handle GET /api/stats - get cache statistics.

        Returns statistics about the cache: total entries, size, entries by model, etc.
        """
        try:
            rows = list(self.storage.index.by_content_hash.values())
            total_entries = len(rows)

            # Calculate statistics
            models: dict[str, int] = {}
            greedy_count = 0
            total_replies = 0

            for row in rows:
                # Count by model
                if row.model:
                    models[row.model] = models.get(row.model, 0) + 1

                # Count greedy responses
                if row.is_greedy:
                    greedy_count += 1

                total_replies += row.replies

            # Calculate total cache directory size
            cache_path = Path(self.cache_dir)
            total_size = 0
            if cache_path.exists():
                for f in cache_path.rglob("*"):
                    if f.is_file():
                        total_size += f.stat().st_size

            result = {
                "total_entries": total_entries,
                "total_replies": total_replies,
                "total_size_bytes": total_size,
                "greedy_responses": greedy_count,
                "non_greedy_responses": total_entries - greedy_count,
                "entries_by_model": models,
            }

            self.log.debug("Returning cache stats: %d entries", total_entries)
            return web.json_response(result)
        except Exception as e:
            self.log.error("Error getting cache stats: %s", e, exc_info=True)
            return web.json_response({"error": "Failed to get cache statistics"}, status=500)

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
            self.log.error("Error getting config: %s", e, exc_info=True)
            return web.json_response({"error": "Failed to get configuration"}, status=500)
