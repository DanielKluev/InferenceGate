"""
Router component that determines how to handle incoming requests.

Decides whether to replay an inference from local storage or forward
the request to the real AI model endpoint based on the current operating mode.
Supports multi-reply cassettes for non-greedy sampling and tiered fuzzy
matching (exact → sampling fuzzy → model fuzzy).

Key classes: `Router`, `ReplayCounter`
"""

import logging
import random
from typing import Any

from inference_gate.modes import Mode
from inference_gate.outflow.client import OutflowClient
from inference_gate.outflow.model_router import OutflowRouter
from inference_gate.recording.hashing import compute_content_hash, compute_prompt_hash, compute_prompt_model_hash, is_greedy
from inference_gate.recording.models import IndexRow
from inference_gate.recording.storage import CachedRequest, CachedResponse, CacheEntry, CacheStorage

# Paths where the proxy may force ``stream=True`` on upstream requests.
# These *must* have a matching reassembly function in ``recording.reassembly``
# so that a single streaming cassette can serve both streaming and
# non-streaming clients.  All other paths (``/tokenize``, ``/detokenize``,
# ``/v1/models``, ``/v1/completions``, etc.) are forwarded as-is.
_FORCE_STREAMING_PATH_PREFIXES: tuple[str, ...] = (
    "/v1/chat/completions",
    "/v1/responses",
    "/responses",
)


class ReplayCounter:
    """
    Tracks which reply to serve next for multi-reply cassettes.

    Supports round-robin (default), random, and first strategies.
    """

    def __init__(self) -> None:
        # content_hash → next reply index (0-based internally, 1-based in tapes)
        self._counters: dict[str, int] = {}

    def next_reply(self, content_hash: str, total_replies: int, strategy: str = "round-robin") -> int:
        """
        Get the next reply number (1-based) to serve for a cassette.

        `strategy` is one of: "round-robin", "random", "first".
        """
        if total_replies <= 0:
            return 1

        if strategy == "first":
            return 1

        if strategy == "random":
            return random.randint(1, total_replies)

        # round-robin (default)
        idx = self._counters.get(content_hash, 0)
        reply_num = (idx % total_replies) + 1
        self._counters[content_hash] = idx + 1
        return reply_num


class Router:
    """
    Routes incoming requests to either cached storage or upstream API.

    In `RECORD_AND_REPLAY` mode, checks cache first and returns cached response
    if available. On cache miss, forwards to upstream via OutflowClient, records
    the response, and returns it.

    For non-greedy requests (temperature > 0 or unspecified), the router collects
    multiple replies up to `max_non_greedy_replies` before switching to replay
    cycling. For greedy requests (temperature == 0), a single reply is stored
    and replayed immediately on cache hit.

    Supports tiered fuzzy matching:
    1. Exact match (content_hash)
    2. Sampling fuzzy (prompt_model_hash) — when `fuzzy_sampling` is not "off"
    3. Model fuzzy (prompt_hash) — when `fuzzy_model` is True

    In `REPLAY_ONLY` mode, only returns cached responses.
    """

    def __init__(self, mode: Mode, storage: CacheStorage, outflow: OutflowClient | OutflowRouter | None = None,
                 non_streaming_models: list[str] | None = None, fuzzy_model: bool = False, fuzzy_sampling: str = "off",
                 max_non_greedy_replies: int = 5) -> None:
        """
        Initialize the router.

        `mode` determines the routing behavior.
        `storage` is the cache storage for recorded inferences.
        `outflow` is the client for forwarding to upstream (required for RECORD_AND_REPLAY mode).
            Accepts either a single ``OutflowClient`` or an ``OutflowRouter`` for
            model-based multi-upstream routing.
        `non_streaming_models` is a list of model names that do not support streaming.
        `fuzzy_model` enables fallback to cache entries with the same prompt but a different model.
        `fuzzy_sampling` controls sampling parameter fuzzy matching: "off", "soft", or "aggressive".
        `max_non_greedy_replies` is the max replies to collect per non-greedy cassette.
        """
        self.log = logging.getLogger("Router")
        self.mode = mode
        self.storage = storage
        self.outflow = outflow
        self.non_streaming_models = non_streaming_models or []
        self.fuzzy_model = fuzzy_model
        self.fuzzy_sampling = fuzzy_sampling
        self.max_non_greedy_replies = max_non_greedy_replies
        self.replay_counter = ReplayCounter()

        if mode == Mode.RECORD_AND_REPLAY and outflow is None:
            raise ValueError("OutflowClient is required for RECORD_AND_REPLAY mode")

    async def route_request(self, method: str, path: str, headers: dict[str, str], body: dict[str, Any] | None = None,
                            query_params: dict[str, str] | None = None) -> CachedResponse:
        """
        Route an incoming request based on the current mode.

        Builds a `CachedRequest`, performs tiered cache lookup, and either
        replays or forwards to upstream depending on the mode.

        Returns a `CachedResponse` with the response data.
        """
        # Build cache request for lookup
        cached_request = self._build_cached_request(method, path, headers, body, query_params)

        # Extract replay strategy from header (default: round-robin)
        strategy = "round-robin"
        strategy_header = headers.get("X-Gate-Reply-Strategy", headers.get("x-gate-reply-strategy", ""))
        if strategy_header in ("round-robin", "random", "first"):
            strategy = strategy_header

        # Tiered cache lookup
        index_row = self._tiered_lookup(method, path, body)

        if self.mode == Mode.REPLAY_ONLY:
            return self._handle_replay_only(index_row, cached_request, strategy)

        # RECORD_AND_REPLAY mode
        return await self._handle_record_and_replay(index_row, cached_request, strategy)

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

    def _tiered_lookup(self, method: str, path: str, body: dict[str, Any] | None) -> IndexRow | None:
        """
        Perform tiered cache lookup: exact → sampling fuzzy → model fuzzy.

        Returns the matching `IndexRow` or None.
        """
        # Tier 1: Exact match (content_hash)
        content_hash = compute_content_hash(method, path, body)
        row = self.storage.index.by_content_hash.get(content_hash)
        if row is not None:
            self.log.debug("Exact match: content_hash=%s", content_hash)
            return row

        # Tier 2: Sampling fuzzy (prompt_model_hash) — if enabled
        if self.fuzzy_sampling != "off":
            pm_hash = compute_prompt_model_hash(method, path, body)
            candidates = self.storage.index.by_prompt_model_hash.get(pm_hash, [])
            row = self._filter_sampling_candidates(candidates, body)
            if row is not None:
                self.log.info("Sampling fuzzy match: prompt_model_hash=%s, cached_model=%s", pm_hash, row.model)
                return row

        # Tier 3: Model fuzzy (prompt_hash) — if enabled
        if self.fuzzy_model:
            p_hash = compute_prompt_hash(body)
            candidates = self.storage.index.by_prompt_hash.get(p_hash, [])
            # Apply sampling filter here too if fuzzy_sampling is active
            if self.fuzzy_sampling != "off":
                row = self._filter_sampling_candidates(candidates, body)
            elif candidates:
                row = self._pick_best_candidate(candidates)
            else:
                row = None
            if row is not None:
                self.log.info("Model fuzzy match: prompt_hash=%s, cached_model=%s", p_hash, row.model)
                return row

        return None

    def _filter_sampling_candidates(self, candidates: list[IndexRow], body: dict[str, Any] | None) -> IndexRow | None:
        """
        Filter and pick from sampling-fuzzy candidates based on the fuzzy_sampling level.

        `soft`: only non-greedy cassettes matched against non-greedy request (non-greedy ↔ non-greedy).
        `aggressive`: any cassette matches (greedy ↔ non-greedy allowed).
        """
        if not candidates:
            return None

        if self.fuzzy_sampling == "soft":
            # Soft: only match non-greedy with non-greedy
            request_is_greedy = is_greedy(body)
            filtered = [r for r in candidates if r.is_greedy == request_is_greedy]
            return self._pick_best_candidate(filtered)

        if self.fuzzy_sampling == "aggressive":
            # Aggressive: any match is fine
            return self._pick_best_candidate(candidates)

        return None

    @staticmethod
    def _pick_best_candidate(candidates: list[IndexRow]) -> IndexRow | None:
        """
        Pick the best candidate from a list of index rows.

        Prefers the one with the most replies (most data), then most recent.
        """
        if not candidates:
            return None
        # Sort by replies desc, then recorded desc (most data first, newest first)
        return max(candidates, key=lambda r: (r.replies, r.recorded))

    def _handle_replay_only(self, index_row: IndexRow | None, cached_request: CachedRequest, strategy: str) -> CachedResponse:
        """
        Handle request in REPLAY_ONLY mode.

        Returns cached response if found (cycling through replies for multi-reply cassettes),
        otherwise returns 503 error response.
        """
        if index_row is not None:
            self.log.info("Replaying cached response for %s %s", cached_request.method, cached_request.path)
            return self._serve_reply(index_row, strategy)

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

    async def _handle_record_and_replay(self, index_row: IndexRow | None, cached_request: CachedRequest, strategy: str) -> CachedResponse:
        """
        Handle request in RECORD_AND_REPLAY mode.

        For greedy requests: cache hit → replay immediately; cache miss → forward and store.
        For non-greedy requests: cache hit with replies < max → forward and append;
        cache hit with replies >= max → cycle through stored replies.
        """
        request_is_greedy = is_greedy(cached_request.body)

        if index_row is not None:
            # Determine max replies for this cassette
            max_replies = 1 if request_is_greedy else self.max_non_greedy_replies

            if not request_is_greedy and index_row.replies < max_replies:
                # Non-greedy cassette not full yet — record another reply
                self.log.info("Non-greedy cassette %s has %d/%d replies — recording another", index_row.content_hash, index_row.replies,
                              max_replies)
                return await self._forward_and_append(cached_request, index_row)

            # Cassette full or greedy — replay
            self.log.info("Cache hit - replaying response for %s %s (replies=%d)", cached_request.method, cached_request.path,
                          index_row.replies)
            return self._serve_reply(index_row, strategy)

        # Cache miss - forward to upstream and create new cassette
        assert self.outflow is not None
        self.log.info("Cache miss - forwarding to upstream for %s %s", cached_request.method, cached_request.path)
        return await self._forward_and_store(cached_request, request_is_greedy)

    async def _forward_and_store(self, cached_request: CachedRequest, request_is_greedy: bool) -> CachedResponse:
        """
        Forward a request to upstream, store the response as a new cassette, and return the response.
        """
        assert self.outflow is not None

        # Remember the client's original streaming preference
        original_client_streaming = False
        if cached_request.body and isinstance(cached_request.body, dict):
            original_client_streaming = cached_request.body.get("stream", False)

        # Force streaming on the upstream request unless the model or path is excluded
        self._maybe_force_streaming(cached_request, original_client_streaming)

        response = await self.outflow.forward_request(cached_request)

        # Determine max replies for this cassette
        max_replies = 1 if request_is_greedy else self.max_non_greedy_replies

        # Record the response
        model = cached_request.body.get("model") if cached_request.body else None
        temperature = cached_request.body.get("temperature") if cached_request.body else None
        p_hash = compute_prompt_hash(cached_request.body)

        entry = CacheEntry(
            request=cached_request,
            response=response,
            model=model,
            temperature=temperature,
            prompt_hash=p_hash,
            original_client_streaming=original_client_streaming,
        )
        content_hash = self.storage.put(entry, max_replies=max_replies)
        self.log.info("Recorded response with content hash %s", content_hash)

        return response

    async def _forward_and_append(self, cached_request: CachedRequest, index_row: IndexRow) -> CachedResponse:
        """
        Forward a request to upstream and append the response to an existing cassette.
        """
        assert self.outflow is not None

        # Remember the client's original streaming preference
        original_client_streaming = False
        if cached_request.body and isinstance(cached_request.body, dict):
            original_client_streaming = cached_request.body.get("stream", False)

        # Force streaming on upstream
        self._maybe_force_streaming(cached_request, original_client_streaming)

        response = await self.outflow.forward_request(cached_request)

        # Store as an appended reply to existing cassette
        model = cached_request.body.get("model") if cached_request.body else None
        temperature = cached_request.body.get("temperature") if cached_request.body else None
        p_hash = compute_prompt_hash(cached_request.body)

        entry = CacheEntry(
            request=cached_request,
            response=response,
            model=model,
            temperature=temperature,
            prompt_hash=p_hash,
            original_client_streaming=original_client_streaming,
        )
        self.storage.put(entry, max_replies=index_row.max_replies)
        self.log.info("Appended reply to cassette %s", index_row.content_hash)

        return response

    def _serve_reply(self, index_row: IndexRow, strategy: str) -> CachedResponse:
        """
        Serve a reply from a cassette, cycling through replies for multi-reply cassettes.
        """
        reply_num = self.replay_counter.next_reply(index_row.content_hash, index_row.replies, strategy)

        # Get the (response_hash, status_code) tuples for this cassette so that
        # recorded error statuses (4xx/5xx) replay with their original HTTP code.
        reply_meta = self.storage.get_reply_metadata(index_row.content_hash)
        if not reply_meta:
            self.log.error("No response hashes found for cassette %s", index_row.content_hash)
            return CachedResponse(
                status_code=500,
                headers={"Content-Type": "application/json"},
                body={"error": {
                    "message": "Internal error: cassette has no replies",
                    "type": "internal_error"
                }},
                is_streaming=False,
            )

        # Clamp reply_num to available replies
        reply_idx = min(reply_num, len(reply_meta)) - 1
        response_hash, status_code = reply_meta[reply_idx]

        # Load response (try streaming first, fall back to non-streaming)
        cached_response = self.storage.load_response(response_hash, streaming=True, status_code=status_code)
        if cached_response is None:
            cached_response = self.storage.load_response(response_hash, streaming=False, status_code=status_code)

        if cached_response is None:
            self.log.error("Response %s not found for cassette %s reply %d", response_hash, index_row.content_hash, reply_num)
            return CachedResponse(
                status_code=500,
                headers={"Content-Type": "application/json"},
                body={"error": {
                    "message": "Internal error: response file missing",
                    "type": "internal_error"
                }},
                is_streaming=False,
            )

        self.log.debug("Serving reply %d/%d from cassette %s", reply_num, index_row.replies, index_row.content_hash)
        return cached_response

    def _maybe_force_streaming(self, cached_request: CachedRequest, original_client_streaming: bool) -> None:
        """
        Force streaming on the upstream request unless the model or path is excluded.
        """
        model_name = cached_request.body.get("model") if cached_request.body else None
        path_supports_streaming = cached_request.path.startswith(_FORCE_STREAMING_PATH_PREFIXES)
        model_supports_streaming = model_name not in self.non_streaming_models if model_name else True
        should_force_streaming = path_supports_streaming and model_supports_streaming

        if should_force_streaming and cached_request.body is not None and isinstance(cached_request.body, dict):
            cached_request.body["stream"] = True
            # Also enable stream_options.include_usage so usage data is preserved
            if "stream_options" not in cached_request.body:
                cached_request.body["stream_options"] = {"include_usage": True}
            elif isinstance(cached_request.body["stream_options"], dict):
                cached_request.body["stream_options"]["include_usage"] = True
            self.log.debug("Forced streaming=True for upstream request (client wanted streaming=%s)", original_client_streaming)
