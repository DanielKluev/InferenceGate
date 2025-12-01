"""FastAPI application for the OpenAI API proxy."""

import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from inference_replay.modes import Mode
from inference_replay.storage import (
    CachedRequest,
    CachedResponse,
    CacheEntry,
    CacheStorage,
)

logger = logging.getLogger(__name__)


class ProxyConfig:
    """Configuration for the proxy server."""

    def __init__(
        self,
        mode: Mode = Mode.DEVELOPMENT,
        cache_dir: str = ".inference_cache",
        upstream_base_url: str = "https://api.openai.com",
        api_key: str | None = None,
    ) -> None:
        self.mode = mode
        self.cache_dir = cache_dir
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.api_key = api_key


# Global config that will be set at startup
_config: ProxyConfig | None = None
_storage: CacheStorage | None = None
_http_client: httpx.AsyncClient | None = None


def get_config() -> ProxyConfig:
    """Get the current proxy configuration."""
    if _config is None:
        raise RuntimeError("Proxy not initialized. Call init_proxy() first.")
    return _config


def get_storage() -> CacheStorage:
    """Get the cache storage instance."""
    if _storage is None:
        raise RuntimeError("Proxy not initialized. Call init_proxy() first.")
    return _storage


def get_client() -> httpx.AsyncClient:
    """Get the HTTP client instance."""
    if _http_client is None:
        raise RuntimeError("Proxy not initialized. Call init_proxy() first.")
    return _http_client


def init_proxy(config: ProxyConfig) -> None:
    """Initialize the proxy with configuration."""
    global _config, _storage, _http_client
    _config = config
    _storage = CacheStorage(config.cache_dir)
    _http_client = httpx.AsyncClient(timeout=120.0)


async def close_proxy() -> None:
    """Close the proxy and release resources."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan."""
    yield
    await close_proxy()


def create_app(config: ProxyConfig | None = None) -> FastAPI:
    """Create the FastAPI application.

    Args:
        config: Optional proxy configuration

    Returns:
        Configured FastAPI application
    """
    if config is not None:
        init_proxy(config)

    app = FastAPI(
        title="InferenceReplay Proxy",
        description="OpenAI API proxy for recording and replaying inference calls",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "mode": get_config().mode.value}

    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        response_model=None,
    )
    async def proxy_openai(request: Request, path: str) -> StreamingResponse | JSONResponse:
        """Proxy requests to OpenAI API."""
        return await handle_proxy_request(request, f"/v1/{path}")

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        response_model=None,
    )
    async def proxy_fallback(request: Request, path: str) -> StreamingResponse | JSONResponse:
        """Fallback proxy for other paths."""
        return await handle_proxy_request(request, f"/{path}")

    return app


def _build_cached_request(
    request: Request,
    body: dict[str, Any] | None,
    path: str,
) -> CachedRequest:
    """Build a CachedRequest from a FastAPI request."""
    # Filter headers to only include relevant ones for caching
    relevant_headers = {}
    for key, value in request.headers.items():
        key_lower = key.lower()
        if key_lower in ("content-type", "accept"):
            relevant_headers[key] = value

    return CachedRequest(
        method=request.method,
        path=path,
        headers=relevant_headers,
        body=body,
        query_params=dict(request.query_params) if request.query_params else None,
    )


def _extract_metadata(body: dict[str, Any] | None) -> tuple[str | None, float | None, str | None]:
    """Extract model, temperature, and prompt hash from request body."""
    if not body:
        return None, None, None

    model = body.get("model")
    temperature = body.get("temperature")
    messages = body.get("messages")

    prompt_hash = None
    if messages:
        prompt_hash = CacheStorage.compute_prompt_hash(messages)

    return model, temperature, prompt_hash


async def _proxy_to_upstream(
    request: Request,
    path: str,
    body: dict[str, Any] | None,
    cached_request: CachedRequest,
) -> StreamingResponse | JSONResponse:
    """Proxy request to upstream and optionally cache response."""
    config = get_config()
    storage = get_storage()
    client = get_client()

    # Build upstream URL
    upstream_url = f"{config.upstream_base_url}{path}"

    # Build headers for upstream
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    # Use configured API key or pass through from request
    if config.api_key:
        headers["authorization"] = f"Bearer {config.api_key}"

    # Check if streaming is requested
    is_streaming = body.get("stream", False) if body else False

    if is_streaming:
        return await _handle_streaming_request(
            client, upstream_url, headers, body, cached_request, storage
        )
    else:
        return await _handle_non_streaming_request(
            client, upstream_url, headers, body, cached_request, storage
        )


async def _handle_streaming_request(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any] | None,
    cached_request: CachedRequest,
    storage: CacheStorage,
) -> StreamingResponse:
    """Handle a streaming request to upstream."""
    model, temperature, prompt_hash = _extract_metadata(body)

    async def stream_and_cache() -> AsyncGenerator[bytes, None]:
        chunks: list[str] = []
        async with client.stream(
            method=cached_request.method,
            url=url,
            headers=headers,
            json=body,
        ) as response:
            async for chunk in response.aiter_bytes():
                chunk_str = chunk.decode("utf-8")
                chunks.append(chunk_str)
                yield chunk

            # Cache the response after streaming completes
            cached_response = CachedResponse(
                status_code=response.status_code,
                headers=dict(response.headers),
                chunks=chunks,
                is_streaming=True,
            )
            entry = CacheEntry(
                request=cached_request,
                response=cached_response,
                model=model,
                temperature=temperature,
                prompt_hash=prompt_hash,
            )
            storage.put(entry)
            logger.info("Cached streaming response for %s", cached_request.path)

    return StreamingResponse(
        stream_and_cache(),
        media_type="text/event-stream",
    )


async def _handle_non_streaming_request(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any] | None,
    cached_request: CachedRequest,
    storage: CacheStorage,
) -> JSONResponse:
    """Handle a non-streaming request to upstream."""
    model, temperature, prompt_hash = _extract_metadata(body)

    if cached_request.method == "GET":
        response = await client.get(url, headers=headers)
    elif cached_request.method == "POST":
        response = await client.post(url, headers=headers, json=body)
    elif cached_request.method == "PUT":
        response = await client.put(url, headers=headers, json=body)
    elif cached_request.method == "DELETE":
        response = await client.delete(url, headers=headers)
    elif cached_request.method == "PATCH":
        response = await client.patch(url, headers=headers, json=body)
    else:
        raise HTTPException(status_code=405, detail=f"Method {cached_request.method} not allowed")

    # Parse response body
    try:
        response_body = response.json()
    except json.JSONDecodeError:
        response_body = None

    # Cache the response
    cached_response = CachedResponse(
        status_code=response.status_code,
        headers=dict(response.headers),
        body=response_body,
        is_streaming=False,
    )
    entry = CacheEntry(
        request=cached_request,
        response=cached_response,
        model=model,
        temperature=temperature,
        prompt_hash=prompt_hash,
    )
    storage.put(entry)
    logger.info("Cached response for %s", cached_request.path)

    return JSONResponse(
        content=response_body,
        status_code=response.status_code,
    )


async def _replay_from_cache(entry: CacheEntry) -> StreamingResponse | JSONResponse:
    """Replay a cached response."""
    response = entry.response

    if response.is_streaming and response.chunks:

        async def replay_chunks() -> AsyncGenerator[bytes, None]:
            for chunk in response.chunks:
                yield chunk.encode("utf-8")

        return StreamingResponse(
            replay_chunks(),
            media_type="text/event-stream",
        )
    else:
        return JSONResponse(
            content=response.body,
            status_code=response.status_code,
        )


async def handle_proxy_request(request: Request, path: str) -> StreamingResponse | JSONResponse:
    """Handle a proxy request based on current mode."""
    config = get_config()
    storage = get_storage()

    # Parse request body if present
    body: dict[str, Any] | None = None
    if request.method in ("POST", "PUT", "PATCH"):
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            body = None

    # Build cached request for lookup
    cached_request = _build_cached_request(request, body, path)

    # Check cache
    cached_entry = storage.get(cached_request)

    if config.mode == Mode.REPLAY:
        # Replay mode: only return cached responses
        if cached_entry is None:
            logger.warning("Cache miss in replay mode for %s", path)
            raise HTTPException(
                status_code=503,
                detail=f"No cached response for {request.method} {path}. "
                "Running in replay mode - cannot proxy to upstream.",
            )
        logger.info("Replaying cached response for %s", path)
        return await _replay_from_cache(cached_entry)

    elif config.mode == Mode.DEVELOPMENT:
        # Development mode: use cache if available, otherwise proxy
        if cached_entry is not None:
            logger.info("Cache hit - replaying response for %s", path)
            return await _replay_from_cache(cached_entry)
        logger.info("Cache miss - proxying to upstream for %s", path)
        return await _proxy_to_upstream(request, path, body, cached_request)

    else:  # Mode.RECORD
        # Record mode: always proxy and cache
        logger.info("Recording request to %s", path)
        return await _proxy_to_upstream(request, path, body, cached_request)
