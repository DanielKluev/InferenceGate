"""
Router component that determines how to handle incoming requests.

Decides whether to replay an inference from local storage or forward
the request to the real AI model endpoint based on the current operating mode.

Key classes: `Router`
"""

import logging
from typing import Any

from inference_gate.modes import Mode
from inference_gate.outflow.client import OutflowClient
from inference_gate.recording.storage import CachedRequest, CachedResponse, CacheEntry, CacheStorage


class Router:
    """
    Routes incoming requests to either cached storage or upstream API.

    In `RECORD_AND_REPLAY` mode, checks cache first and returns cached response
    if available. On cache miss, forwards to upstream via OutflowClient, records
    the response, and returns it. Requests are forced to use streaming when sent
    upstream (unless the model is in `non_streaming_models`) so that a single
    streaming cassette can serve both streaming and non-streaming clients.

    In `REPLAY_ONLY` mode, only returns cached responses. Returns an error
    response on cache miss.
    """

    def __init__(self, mode: Mode, storage: CacheStorage, outflow: OutflowClient | None = None,
                 non_streaming_models: list[str] | None = None, fuzzy_model_matching: bool = False) -> None:
        """
        Initialize the router.

        `mode` determines the routing behavior.
        `storage` is the cache storage for recorded inferences.
        `outflow` is the client for forwarding to upstream (required for RECORD_AND_REPLAY mode).
        `non_streaming_models` is a list of model names that do not support streaming
        and should not be forced to stream.
        `fuzzy_model_matching` enables fallback to cache entries with the same prompt
        but a different model when the exact cache key is not found.
        """
        self.log = logging.getLogger("Router")
        self.mode = mode
        self.storage = storage
        self.outflow = outflow
        self.non_streaming_models = non_streaming_models or []
        self.fuzzy_model_matching = fuzzy_model_matching

        if mode == Mode.RECORD_AND_REPLAY and outflow is None:
            raise ValueError("OutflowClient is required for RECORD_AND_REPLAY mode")

    async def route_request(self, method: str, path: str, headers: dict[str, str], body: dict[str, Any] | None = None,
                            query_params: dict[str, str] | None = None) -> CachedResponse:
        """
        Route an incoming request based on the current mode.

        Builds a `CachedRequest`, checks cache, and either replays or forwards
        to upstream depending on the mode. When ``fuzzy_model_matching`` is
        enabled, a cache miss triggers a secondary lookup by prompt hash so
        that a cached response recorded with a different model can be reused.

        Returns a `CachedResponse` with the response data.
        """
        # Build cache request for lookup
        cached_request = self._build_cached_request(method, path, headers, body, query_params)

        # Check cache for existing response (exact match)
        cached_entry = self.storage.get(cached_request)

        # On exact miss, attempt fuzzy model matching if enabled
        if cached_entry is None and self.fuzzy_model_matching:
            cached_entry = self._try_fuzzy_match(body)

        if self.mode == Mode.REPLAY_ONLY:
            return self._handle_replay_only(cached_entry, cached_request)

        # RECORD_AND_REPLAY mode
        return await self._handle_record_and_replay(cached_entry, cached_request)

    def _build_cached_request(self, method: str, path: str, headers: dict[str, str], body: dict[str, Any] | None,
                              query_params: dict[str, str] | None) -> CachedRequest:
        """
        Build a `CachedRequest` from incoming request data.

        Filters headers to only include relevant ones for cache key computation.
        """
        # Filter headers to only include relevant ones for caching
        relevant_headers: dict[str, str] = {}
        for key, value in headers.items():
            key_lower = key.lower()
            if key_lower in ("content-type", "accept"):
                relevant_headers[key] = value

        return CachedRequest(method=method, path=path, headers=relevant_headers, body=body, query_params=query_params)

    def _extract_metadata(self, body: dict[str, Any] | None) -> tuple[str | None, float | None, str | None]:
        """
        Extract model, temperature, and prompt hash from request body.

        Returns a tuple of (model, temperature, prompt_hash).
        """
        if not body:
            return None, None, None

        model = body.get("model")
        temperature = body.get("temperature")
        messages = body.get("messages")

        prompt_hash = None
        if messages:
            prompt_hash = CacheStorage.compute_prompt_hash(messages)

        return model, temperature, prompt_hash

    def _try_fuzzy_match(self, body: dict[str, Any] | None) -> CacheEntry | None:
        """
        Attempt a fuzzy model match by prompt hash.

        Computes the prompt hash from the request body's messages and
        searches the cache for any entry with the same prompt hash,
        regardless of which model was used to record it.

        Returns a `CacheEntry` if a fuzzy match is found, None otherwise.
        """
        if not body:
            return None
        messages = body.get("messages")
        if not messages:
            return None
        prompt_hash = CacheStorage.compute_prompt_hash(messages)
        entry = self.storage.get_by_prompt_hash(prompt_hash)
        if entry is not None:
            self.log.info("Fuzzy model match found (prompt_hash=%s, cached_model=%s)", prompt_hash, entry.model)
        return entry

    def _handle_replay_only(self, cached_entry: CacheEntry | None, cached_request: CachedRequest) -> CachedResponse:
        """
        Handle request in REPLAY_ONLY mode.

        Returns cached response if found, otherwise returns 503 error response.
        """
        if cached_entry is not None:
            self.log.info("Replaying cached response for %s %s", cached_request.method, cached_request.path)
            return cached_entry.response

        self.log.warning("Cache miss in replay-only mode for %s %s", cached_request.method, cached_request.path)
        return CachedResponse(
            status_code=503,
            headers={"Content-Type": "application/json"},
            body={
                "error": {
                    "message": f"No cached response for {cached_request.method} {cached_request.path}. "
                               "Running in replay-only mode - cannot proxy to upstream.",
                    "type": "cache_miss",
                    "code": "no_cached_response"
                }
            },
            is_streaming=False,
        )

    async def _handle_record_and_replay(self, cached_entry: CacheEntry | None, cached_request: CachedRequest) -> CachedResponse:
        """
        Handle request in RECORD_AND_REPLAY mode.

        Returns cached response if found. On cache miss, forwards to upstream
        via OutflowClient (forcing streaming unless the model is in `non_streaming_models`),
        stores the response, and returns it.
        """
        if cached_entry is not None:
            self.log.info("Cache hit - replaying response for %s %s", cached_request.method, cached_request.path)
            return cached_entry.response

        # Cache miss - forward to upstream
        assert self.outflow is not None
        self.log.info("Cache miss - forwarding to upstream for %s %s", cached_request.method, cached_request.path)

        # Remember the client's original streaming preference
        original_client_streaming = False
        if cached_request.body and isinstance(cached_request.body, dict):
            original_client_streaming = cached_request.body.get("stream", False)

        # Force streaming on the upstream request unless the model is excluded
        model_name = cached_request.body.get("model") if cached_request.body else None
        should_force_streaming = model_name not in self.non_streaming_models if model_name else True

        if should_force_streaming and cached_request.body is not None and isinstance(cached_request.body, dict):
            cached_request.body["stream"] = True
            # Also enable stream_options.include_usage so usage data is preserved
            if "stream_options" not in cached_request.body:
                cached_request.body["stream_options"] = {"include_usage": True}
            elif isinstance(cached_request.body["stream_options"], dict):
                cached_request.body["stream_options"]["include_usage"] = True
            self.log.debug("Forced streaming=True for upstream request (client wanted streaming=%s)", original_client_streaming)

        response = await self.outflow.forward_request(cached_request)

        # Record the response
        model, temperature, prompt_hash = self._extract_metadata(cached_request.body)
        entry = CacheEntry(
            request=cached_request,
            response=response,
            model=model,
            temperature=temperature,
            prompt_hash=prompt_hash,
            original_client_streaming=original_client_streaming,
        )
        cache_key = self.storage.put(entry)
        self.log.info("Recorded response with cache key %s", cache_key)

        return response
