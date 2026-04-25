"""
Model-based upstream routing for multi-endpoint recording.

When InferenceGate is configured with ``model_routes``, the ``OutflowRouter``
inspects each request's ``model`` field and dispatches to the correct
upstream endpoint.  This allows a single InferenceGate proxy to record
cassettes for models served by different backends (e.g. vLLM on one GPU
server, llama.cpp on another).

Key classes: ``UpstreamConfig``, ``OutflowRouter``
"""

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import Any

from inference_gate.outflow.client import OutflowClient
from inference_gate.recording.storage import CachedRequest, CachedResponse


@dataclass
class UpstreamConfig:
    """
    Connection details for a single upstream API endpoint.

    Attributes:
        url: Base URL of the upstream API (e.g. ``"http://127.0.0.1:8125"``).
        api_key: Optional Bearer token for authentication.
        timeout: Request timeout in seconds.
        proxy: Optional HTTP proxy URL for routing upstream requests.
    """

    url: str
    api_key: str | None = None
    timeout: float = 120.0
    proxy: str | None = None


class OutflowRouter:
    """
    Routes outgoing requests to different upstream endpoints based on model name.

    Implements the same interface as ``OutflowClient`` (``start``, ``stop``,
    ``forward_request``) so it can be used as a drop-in replacement in the
    ``Router`` component.

    Resolution order for a request with ``model="Gemma4:E4B-it-Q4_K_M"``:
    1. Exact model name match in ``model_routes``.
    2. ``fnmatch``-style glob match (e.g. key ``"Gemma4:*"``).
    3. Fall back to ``default_upstream``.

    Each unique upstream URL gets its own ``OutflowClient`` (deduplicated).
    """

    def __init__(self, default_upstream: UpstreamConfig, model_routes: dict[str, UpstreamConfig] | None = None) -> None:
        """
        Initialize the outflow router.

        ``default_upstream`` is the fallback used when a request's model does
        not match any explicit route.
        ``model_routes`` maps model names (or glob patterns) to their upstream
        configs.  Exact-name entries are tried first, then glob patterns, then
        the default.
        """
        self.log = logging.getLogger("OutflowRouter")
        self._default_upstream = default_upstream
        self._model_routes: dict[str, UpstreamConfig] = model_routes or {}

        # Split routes into exact names and glob patterns for efficient lookup
        self._exact_routes: dict[str, UpstreamConfig] = {}
        self._glob_routes: list[tuple[str, UpstreamConfig]] = []
        for pattern, cfg in self._model_routes.items():
            if any(ch in pattern for ch in ("*", "?", "[", "]")):
                self._glob_routes.append((pattern, cfg))
            else:
                self._exact_routes[pattern] = cfg

        # Deduplicated OutflowClient pool, keyed by upstream URL
        self._clients: dict[str, OutflowClient] = {}
        self._build_client_pool()

    def _build_client_pool(self) -> None:
        """
        Pre-create one ``OutflowClient`` per unique upstream URL.

        Deduplicates: if two model routes point to the same URL, they share
        a single client.
        """
        all_configs = [self._default_upstream] + list(self._model_routes.values())
        for cfg in all_configs:
            url_key = cfg.url.rstrip("/")
            if url_key not in self._clients:
                self._clients[url_key] = OutflowClient(upstream_base_url=cfg.url, api_key=cfg.api_key, timeout=cfg.timeout, proxy=cfg.proxy)

    def _resolve_client(self, model_name: str | None) -> OutflowClient:
        """
        Resolve which ``OutflowClient`` should handle a request for ``model_name``.

        Resolution order: exact match → glob match → default.
        """
        if model_name:
            # 1. Exact match
            cfg = self._exact_routes.get(model_name)
            if cfg is not None:
                self.log.debug("Exact route match for model '%s'", model_name)
                return self._clients[cfg.url.rstrip("/")]

            # 2. Glob match (first match wins)
            for pattern, cfg in self._glob_routes:
                if fnmatch.fnmatch(model_name, pattern):
                    self.log.debug("Glob route match '%s' for model '%s'", pattern, model_name)
                    return self._clients[cfg.url.rstrip("/")]

        # 3. Default
        self.log.debug("Using default upstream for model '%s'", model_name)
        return self._clients[self._default_upstream.url.rstrip("/")]

    async def start(self) -> None:
        """
        Start all pooled ``OutflowClient`` instances.
        """
        for url_key, client in self._clients.items():
            await client.start()
        self.log.info("OutflowRouter started with %d upstream(s): %s", len(self._clients), list(self._clients.keys()))

    async def stop(self) -> None:
        """
        Stop all pooled ``OutflowClient`` instances.
        """
        for client in self._clients.values():
            await client.stop()
        self.log.info("OutflowRouter stopped")

    async def forward_request(self, request: CachedRequest) -> CachedResponse:
        """
        Forward a request to the appropriate upstream based on the model name in the request body.

        Extracts ``model`` from ``request.body`` and resolves the target
        client.  For requests without a model field (e.g. ``/v1/models``),
        the default upstream is used.
        """
        model_name: str | None = None
        if request.body and isinstance(request.body, dict):
            model_name = request.body.get("model")

        client = self._resolve_client(model_name)
        return await client.forward_request(request)
