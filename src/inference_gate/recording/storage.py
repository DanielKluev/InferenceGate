"""Storage layer for caching API calls."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError


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
    original_client_streaming: bool | None = None


class CacheStorage:
    """File-based storage for API call caching."""

    def __init__(self, cache_dir: str | Path = ".inference_cache") -> None:
        """Initialize cache storage.

        Args:
            cache_dir: Directory to store cache files
        """
        self.log = logging.getLogger("CacheStorage")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Lazily-built index mapping prompt_hash -> cache_key for fast fuzzy lookups.
        # Populated on the first call to get_by_prompt_hash() and kept in sync by put().
        self._prompt_hash_index: dict[str, str] | None = None

    # Fields excluded from cache key computation so that streaming and
    # non-streaming requests for the same prompt share the same entry.
    _CACHE_KEY_EXCLUDED_FIELDS = {"stream", "stream_options"}

    # Headers stripped from stored cassettes to prevent leaking credentials.
    # Applied as a defense-in-depth measure in put() before writing to disk.
    _SANITIZED_HEADERS = {"authorization", "x-api-key", "proxy-authorization"}

    def _compute_cache_key(self, request: CachedRequest) -> str:
        """Compute a unique cache key for a request.

        The `stream` and `stream_options` fields are excluded from the request
        body before hashing so that streaming and non-streaming requests for the
        same prompt share the same cache entry (the router may inject these
        fields when forcing streaming on the upstream request).

        Args:
            request: The request to compute key for

        Returns:
            A unique hash string for this request
        """
        # Build a body copy without streaming-related fields so that
        # stream=True and stream=False requests resolve to the same key.
        body_for_key = request.body
        if body_for_key is not None and isinstance(body_for_key, dict):
            excluded = self._CACHE_KEY_EXCLUDED_FIELDS
            if excluded & body_for_key.keys():
                body_for_key = {k: v for k, v in body_for_key.items() if k not in excluded}

        # Include relevant request data for caching
        key_data = {
            "method": request.method,
            "path": request.path,
            "body": body_for_key,
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

    def _build_prompt_hash_index(self) -> dict[str, str]:
        """Build an in-memory index mapping prompt_hash to cache_key.

        Scans all cassette files once and returns a ``{prompt_hash: cache_key}``
        dict. Files are processed in sorted order so that the *first* match for
        a given prompt_hash is deterministic across filesystems.

        Returns:
            Mapping of prompt_hash to cache_key
        """
        index: dict[str, str] = {}
        for cache_file in sorted(self.cache_dir.glob("*.json")):
            try:
                with open(cache_file, encoding="utf-8") as f:
                    data = json.load(f)
                entry = CacheEntry.model_validate(data)
                if entry.prompt_hash and entry.prompt_hash not in index:
                    index[entry.prompt_hash] = cache_file.stem
            except (json.JSONDecodeError, ValidationError):
                self.log.debug("Skipping unreadable cache file %s during index build", cache_file)
                continue
        return index

    def get_by_prompt_hash(self, prompt_hash: str) -> CacheEntry | None:
        """Look up a cached entry by prompt hash, ignoring the model.

        Uses a lazily-built in-memory index (``prompt_hash`` → ``cache_key``)
        to avoid scanning the entire cache directory on every call. The index
        is built once on the first lookup and kept in sync when new entries
        are stored via ``put()``.

        When multiple cassettes share the same prompt hash, the one whose
        filename sorts first is returned (deterministic across runs).

        This is used for fuzzy model matching: when an exact cache key miss
        occurs, the caller can compute the prompt hash from the request
        messages and search for any cached entry with the same prompt
        regardless of which model was used to record it.

        Args:
            prompt_hash: The prompt hash to search for

        Returns:
            CacheEntry if a matching entry is found, None otherwise
        """
        for attempt in range(2):
            if self._prompt_hash_index is None:
                self._prompt_hash_index = self._build_prompt_hash_index()

            cache_key = self._prompt_hash_index.get(prompt_hash)
            if cache_key is None:
                # No entry for this prompt_hash in the current index; nothing to return
                return None

            cache_file = self._get_cache_file(cache_key)
            if not cache_file.exists():
                # File was removed after the index was built; rebuild to pick up alternatives
                self.log.debug("Cache file %s for prompt_hash %s missing during fuzzy lookup (attempt %d); rebuilding index", cache_file,
                               prompt_hash, attempt + 1)
                self._prompt_hash_index = None
                # Retry once with a rebuilt index to find an alternative cassette, if any
                continue

            try:
                with open(cache_file, encoding="utf-8") as f:
                    data = json.load(f)
                entry = CacheEntry.model_validate(data)
                self.log.debug("Fuzzy match found in %s (model=%s)", cache_key, entry.model)
                return entry
            except (json.JSONDecodeError, ValidationError):
                self.log.debug("Skipping unreadable cache file %s during fuzzy lookup (attempt %d); rebuilding index", cache_file,
                               attempt + 1)
                # Invalidate and rebuild to pick up alternative entries for this prompt_hash
                self._prompt_hash_index = None
                # Retry once with a rebuilt index to find an alternative cassette, if any
                continue

        # After retrying once with a rebuilt index, no valid entry was found
        return None

    def put(self, entry: CacheEntry) -> str:
        """Store a cache entry.

        Sensitive headers (Authorization, X-Api-Key, etc.) are stripped from
        the stored request headers to prevent credential leakage in committed
        cassettes.

        Args:
            entry: The cache entry to store

        Returns:
            The cache key used
        """
        cache_key = self._compute_cache_key(entry.request)
        cache_file = self._get_cache_file(cache_key)

        data = entry.model_dump()
        # Strip sensitive headers from the stored request to prevent credential leakage
        if data.get("request", {}).get("headers"):
            data["request"]["headers"] = {k: v for k, v in data["request"]["headers"].items() if k.lower() not in self._SANITIZED_HEADERS}

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Keep the prompt_hash index in sync if it has been built
        if self._prompt_hash_index is not None and entry.prompt_hash:
            current_key = self._prompt_hash_index.get(entry.prompt_hash)
            if current_key is None or cache_key < current_key:
                # Always keep the lexicographically smallest cache key for a given prompt_hash
                self._prompt_hash_index[entry.prompt_hash] = cache_key

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
        # Invalidate the prompt_hash index since all entries are gone
        self._prompt_hash_index = None
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
