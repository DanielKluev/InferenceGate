"""Storage layer for caching API calls."""

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class CachedRequest(BaseModel):
    """Cached request data."""

    method: str
    path: str
    headers: dict[str, str]
    body: dict[str, Any] | None = None
    query_params: dict[str, str] | None = None


class CachedResponse(BaseModel):
    """Cached response data."""

    status_code: int
    headers: dict[str, str]
    body: dict[str, Any] | None = None
    chunks: list[str] | None = None  # For streaming responses
    is_streaming: bool = False


class CacheEntry(BaseModel):
    """A single cache entry with request/response pair."""

    request: CachedRequest
    response: CachedResponse
    model: str | None = None
    temperature: float | None = None
    prompt_hash: str | None = None


class CacheStorage:
    """File-based storage for API call caching."""

    def __init__(self, cache_dir: str | Path = ".inference_cache") -> None:
        """Initialize cache storage.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_cache_key(self, request: CachedRequest) -> str:
        """Compute a unique cache key for a request.

        Args:
            request: The request to compute key for

        Returns:
            A unique hash string for this request
        """
        # Include relevant request data for caching
        key_data = {
            "method": request.method,
            "path": request.path,
            "body": request.body,
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]

    def _get_cache_file(self, cache_key: str) -> Path:
        """Get the cache file path for a given key."""
        return self.cache_dir / f"{cache_key}.json"

    def get(self, request: CachedRequest) -> CacheEntry | None:
        """Look up a cached response for a request.

        Args:
            request: The request to look up

        Returns:
            CacheEntry if found, None otherwise
        """
        cache_key = self._compute_cache_key(request)
        cache_file = self._get_cache_file(cache_key)

        if not cache_file.exists():
            return None

        with open(cache_file, encoding="utf-8") as f:
            data = json.load(f)
        return CacheEntry.model_validate(data)

    def put(self, entry: CacheEntry) -> str:
        """Store a cache entry.

        Args:
            entry: The cache entry to store

        Returns:
            The cache key used
        """
        cache_key = self._compute_cache_key(entry.request)
        cache_file = self._get_cache_file(cache_key)

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(entry.model_dump(), f, indent=2)

        return cache_key

    def exists(self, request: CachedRequest) -> bool:
        """Check if a request is cached.

        Args:
            request: The request to check

        Returns:
            True if cached, False otherwise
        """
        cache_key = self._compute_cache_key(request)
        return self._get_cache_file(cache_key).exists()

    def clear(self) -> int:
        """Clear all cached entries.

        Returns:
            Number of entries cleared
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count

    def list_entries(self) -> list[tuple[str, CacheEntry]]:
        """List all cached entries.

        Returns:
            List of (cache_key, entry) tuples
        """
        entries = []
        for cache_file in self.cache_dir.glob("*.json"):
            cache_key = cache_file.stem
            with open(cache_file, encoding="utf-8") as f:
                data = json.load(f)
            entries.append((cache_key, CacheEntry.model_validate(data)))
        return entries

    @staticmethod
    def compute_prompt_hash(messages: list[dict[str, Any]]) -> str:
        """Compute a hash for prompt messages.

        Args:
            messages: List of message dicts

        Returns:
            Hash string for the prompt
        """
        prompt_str = json.dumps(messages, sort_keys=True)
        return hashlib.sha256(prompt_str.encode()).hexdigest()[:16]
