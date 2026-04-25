"""
`admin` is providing the ``/gate/*`` namespace for InferenceGate's own control
endpoints.  This namespace is reserved: requests under ``/gate/`` are never
routed upstream and never recorded as cassettes.  It is the canonical way for
clients (Glue, pytest-xdist controllers, operators) to introspect and reconfigure
a running Gate without restarting it.

Endpoints:

- ``GET  /gate/health``        — liveness + current mode/fuzzy settings.
- ``GET  /gate/config``        — full redacted config snapshot.
- ``POST /gate/config``        — apply runtime overrides to mode/fuzzy/limits.
- ``POST /gate/index/reload``  — re-scan the cassette directory and rebuild index.
- ``GET  /gate/stats``         — counters: cassettes, replies, hits/misses by tier.

Sensitive values (api_key, Authorization header, URL userinfo) are masked via
:func:`redact` before being included in any response.

Key functions: :func:`register_admin_routes`, :func:`redact`.
"""

import logging
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from aiohttp import web

from inference_gate.modes import Mode
from inference_gate.outflow.client import OutflowClient
from inference_gate.outflow.model_router import OutflowRouter, UpstreamConfig
from inference_gate.router.router import Router

log = logging.getLogger("InflowServer.admin")

REDACTION_PLACEHOLDER = "***"

_FUZZY_SAMPLING_VALUES = ("off", "soft", "aggressive")


def _redact_url(url: str) -> str:
    """
    Return ``url`` with any embedded ``user:password@`` userinfo replaced by ``***@``.
    Non-URL strings are returned unchanged.
    """
    if not url or "://" not in url:
        return url
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if not parts.username and not parts.password:
        return url
    netloc = parts.hostname or ""
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    netloc = f"{REDACTION_PLACEHOLDER}@{netloc}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _redact_upstream(cfg: UpstreamConfig) -> dict[str, Any]:
    """
    Render an :class:`UpstreamConfig` as a JSON-safe dict with secrets removed.
    """
    return {
        "url": _redact_url(cfg.url),
        "api_key": REDACTION_PLACEHOLDER if cfg.api_key else None,
        "timeout": cfg.timeout,
        "proxy": _redact_url(cfg.proxy) if cfg.proxy else None,
    }


def _redact_outflow(outflow: OutflowClient | OutflowRouter | None) -> dict[str, Any] | None:
    """
    Render the router's outflow component (single client or model-based router) as a
    JSON-safe dict with secrets redacted.
    """
    if outflow is None:
        return None
    if isinstance(outflow, OutflowRouter):
        return {
            "type": "router",
            "default": {
                "url": _redact_url(outflow._default_upstream.url),
                "api_key": REDACTION_PLACEHOLDER if outflow._default_upstream.api_key else None,
                "timeout": outflow._default_upstream.timeout,
                "proxy": _redact_url(outflow._default_upstream.proxy) if outflow._default_upstream.proxy else None,
            },
            "routes": {pattern: _redact_upstream(cfg) for pattern, cfg in outflow._model_routes.items()},
        }
    # Single OutflowClient.  ``timeout`` is an ``aiohttp.ClientTimeout`` so we surface
    # only the human-meaningful total seconds.
    timeout_seconds: float | None = None
    timeout = getattr(outflow, "timeout", None)
    if timeout is not None:
        timeout_seconds = getattr(timeout, "total", None)
    return {
        "type": "client",
        "url": _redact_url(outflow.upstream_base_url),
        "api_key": REDACTION_PLACEHOLDER if outflow.api_key else None,
        "timeout": timeout_seconds,
        "proxy": _redact_url(outflow.proxy) if getattr(outflow, "proxy", None) else None,
    }


_HEADER_REDACT_PATTERN = re.compile(r"^(authorization|proxy-authorization|x-api-key)$", re.IGNORECASE)


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """
    Return a copy of ``headers`` with auth-bearing fields masked.

    Useful for any future endpoint that echoes back request metadata.
    """
    out: dict[str, str] = {}
    for name, value in headers.items():
        if _HEADER_REDACT_PATTERN.match(name):
            out[name] = REDACTION_PLACEHOLDER
        else:
            out[name] = value
    return out


def redact(value: Any) -> Any:
    """
    Best-effort recursive redaction for arbitrary JSON-serialisable structures.

    Strings whose key suggests an API key, password, or token are masked; all
    other scalars pass through.  This is the public helper for endpoints that
    need to dump partially-known payloads.
    """
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, sub in value.items():
            if isinstance(key, str) and re.search(r"(api[_-]?key|password|secret|token|authorization)", key, re.IGNORECASE):
                out[key] = REDACTION_PLACEHOLDER if sub else sub
            else:
                out[key] = redact(sub)
        return out
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def _config_snapshot(router: Router) -> dict[str, Any]:
    """
    Build the JSON payload for ``GET /gate/config`` from the live :class:`Router`.
    """
    storage = router.storage
    return {
        "mode": router.mode.value,
        "fuzzy_model": router.fuzzy_model,
        "fuzzy_sampling": router.fuzzy_sampling,
        "max_non_greedy_replies": router.max_non_greedy_replies,
        "non_streaming_models": list(router.non_streaming_models),
        "cache_dir": str(storage.cache_dir),
        "outflow": _redact_outflow(router.outflow),
        "stats": {
            "cassettes": len(storage.index.by_content_hash),
        },
    }


async def _handle_health(request: web.Request) -> web.Response:
    """
    Return a small liveness payload with the current mode and fuzzy settings.

    Mirrors ``GET /health`` (kept as alias for backward compatibility) but
    located in the reserved ``/gate/`` namespace so future probes can rely on
    the namespace contract.
    """
    router: Router = request.app["router"]
    return web.json_response({
        "status": "healthy",
        "mode": router.mode.value,
        "fuzzy_model": router.fuzzy_model,
        "fuzzy_sampling": router.fuzzy_sampling,
    })


async def _handle_get_config(request: web.Request) -> web.Response:
    """
    Return a full, redacted snapshot of the running Gate's config.
    """
    router: Router = request.app["router"]
    return web.json_response(_config_snapshot(router))


async def _handle_post_config(request: web.Request) -> web.Response:
    """
    Apply a partial runtime override of router config.

    Accepted JSON body keys: ``mode``, ``fuzzy_model``, ``fuzzy_sampling``,
    ``max_non_greedy_replies``.  Unknown keys cause a 400 response so callers
    can rely on strict validation.  Returns the new redacted snapshot.
    """
    router: Router = request.app["router"]
    try:
        body = await request.json()
    except (ValueError, Exception):  # pylint: disable=broad-except
        return web.json_response(status=400, data={"error": {"message": "Invalid JSON body", "type": "invalid_request"}})
    if not isinstance(body, dict):
        return web.json_response(status=400, data={"error": {"message": "Body must be a JSON object", "type": "invalid_request"}})

    accepted = {"mode", "fuzzy_model", "fuzzy_sampling", "max_non_greedy_replies"}
    unknown = set(body) - accepted
    if unknown:
        return web.json_response(status=400, data={"error": {
            "message": f"Unknown config keys: {sorted(unknown)}. Accepted: {sorted(accepted)}",
            "type": "invalid_request",
        }})

    if "mode" in body:
        try:
            router.mode = Mode(body["mode"])
        except ValueError:
            return web.json_response(status=400, data={"error": {
                "message": f"Invalid mode: {body['mode']!r}. Expected one of {[m.value for m in Mode]}",
                "type": "invalid_request",
            }})
    if "fuzzy_model" in body:
        if not isinstance(body["fuzzy_model"], bool):
            return web.json_response(status=400, data={"error": {
                "message": "fuzzy_model must be a boolean",
                "type": "invalid_request",
            }})
        router.fuzzy_model = body["fuzzy_model"]
    if "fuzzy_sampling" in body:
        if body["fuzzy_sampling"] not in _FUZZY_SAMPLING_VALUES:
            return web.json_response(status=400, data={"error": {
                "message": f"fuzzy_sampling must be one of {list(_FUZZY_SAMPLING_VALUES)}",
                "type": "invalid_request",
            }})
        router.fuzzy_sampling = body["fuzzy_sampling"]
    if "max_non_greedy_replies" in body:
        value = body["max_non_greedy_replies"]
        if not isinstance(value, int) or value < 1:
            return web.json_response(status=400, data={"error": {
                "message": "max_non_greedy_replies must be a positive integer",
                "type": "invalid_request",
            }})
        router.max_non_greedy_replies = value

    log.info("Runtime config updated via /gate/config: %s", sorted(body))
    return web.json_response(_config_snapshot(router))


async def _handle_index_reload(request: web.Request) -> web.Response:
    """
    Force a rebuild of the on-disk cassette index.

    Useful when external tools have added cassettes to the cache directory
    while the server was running.
    """
    router: Router = request.app["router"]
    router.storage.index.rebuild(router.storage.requests_dir)
    return web.json_response({
        "status": "reloaded",
        "cassettes": len(router.storage.index.by_content_hash),
    })


async def _handle_stats(request: web.Request) -> web.Response:
    """
    Return per-tier counters describing the current cache state.
    """
    router: Router = request.app["router"]
    storage = router.storage
    return web.json_response({
        "cassettes": len(storage.index.by_content_hash),
        "by_prompt_model_hash": len(storage.index.by_prompt_model_hash),
        "by_prompt_hash": len(storage.index.by_prompt_hash),
        "mode": router.mode.value,
    })


async def _handle_gate_not_found(request: web.Request) -> web.Response:
    """
    Catch-all for unknown ``/gate/*`` paths so they do not fall through to the
    proxy handler.  Returns 404 with a structured JSON body.
    """
    return web.json_response(status=404, data={
        "error": {
            "message": f"Unknown admin endpoint: {request.path}",
            "type": "not_found",
        }
    })


def register_admin_routes(app: web.Application, router: Router) -> None:
    """
    Register the ``/gate/*`` admin namespace on ``app`` and stash ``router``
    under ``app['router']`` for the handlers.

    Routes are added before any catch-all proxy route so they take precedence.
    """
    app["router"] = router
    app.router.add_route("GET", "/gate/health", _handle_health)
    app.router.add_route("GET", "/gate/config", _handle_get_config)
    app.router.add_route("POST", "/gate/config", _handle_post_config)
    app.router.add_route("POST", "/gate/index/reload", _handle_index_reload)
    app.router.add_route("GET", "/gate/stats", _handle_stats)
    app.router.add_route("*", "/gate/{path:.*}", _handle_gate_not_found)
