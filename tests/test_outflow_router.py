"""
Tests for the ``OutflowRouter`` model-based multi-upstream routing.

Validates exact-name matching, glob-pattern matching, and default
fallback behaviour, as well as client pool deduplication and the
``start``/``stop``/``forward_request`` interface.
"""

import pytest
from unittest.mock import AsyncMock, patch

from inference_gate.outflow.model_router import OutflowRouter, UpstreamConfig
from inference_gate.recording.storage import CachedRequest, CachedResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(model: str | None = None) -> CachedRequest:
    """
    Build a minimal ``CachedRequest`` with an optional model field.
    """
    body = {"messages": [{"role": "user", "content": "hi"}], "stream": False}
    if model is not None:
        body["model"] = model
    return CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body=body)


def _make_response() -> CachedResponse:
    """
    Build a minimal ``CachedResponse`` stub.
    """
    return CachedResponse(status_code=200, headers={}, body={"choices": []}, is_streaming=False)


# ---------------------------------------------------------------------------
# Route resolution tests
# ---------------------------------------------------------------------------


class TestRouteResolution:
    """
    Verify that ``_resolve_client`` dispatches to the correct upstream
    for exact names, glob patterns, and the default fallback.
    """

    def test_exact_match(self):
        """
        An exact model name match returns the route-specific client.
        """
        default = UpstreamConfig(url="http://default:8000")
        routes = {"Gemma4:E4B-it-Q4_K_M": UpstreamConfig(url="http://llamacpp:8125")}
        router = OutflowRouter(default_upstream=default, model_routes=routes)

        client = router._resolve_client("Gemma4:E4B-it-Q4_K_M")
        assert client.upstream_base_url == "http://llamacpp:8125"

    def test_glob_match(self):
        """
        A glob pattern like ``"Gemma4:*"`` matches model names that start with ``"Gemma4:"``.
        """
        default = UpstreamConfig(url="http://default:8000")
        routes = {"Gemma4:*": UpstreamConfig(url="http://llamacpp:8125")}
        router = OutflowRouter(default_upstream=default, model_routes=routes)

        client = router._resolve_client("Gemma4:E4B-it-Q4_K_M")
        assert client.upstream_base_url == "http://llamacpp:8125"

    def test_exact_takes_priority_over_glob(self):
        """
        When both an exact match and a glob match exist, exact wins.
        """
        default = UpstreamConfig(url="http://default:8000")
        routes = {
            "Gemma4:E4B-it-Q4_K_M": UpstreamConfig(url="http://exact:8125"),
            "Gemma4:*": UpstreamConfig(url="http://glob:8126"),
        }
        router = OutflowRouter(default_upstream=default, model_routes=routes)

        client = router._resolve_client("Gemma4:E4B-it-Q4_K_M")
        assert client.upstream_base_url == "http://exact:8125"

    def test_default_fallback(self):
        """
        A model that doesn't match any route falls back to the default upstream.
        """
        default = UpstreamConfig(url="http://default:8000")
        routes = {"Gemma4:*": UpstreamConfig(url="http://llamacpp:8125")}
        router = OutflowRouter(default_upstream=default, model_routes=routes)

        client = router._resolve_client("some-other-model")
        assert client.upstream_base_url == "http://default:8000"

    def test_none_model_uses_default(self):
        """
        Requests without a model field (e.g. ``/v1/models``) use the default.
        """
        default = UpstreamConfig(url="http://default:8000")
        routes = {"Gemma4:*": UpstreamConfig(url="http://llamacpp:8125")}
        router = OutflowRouter(default_upstream=default, model_routes=routes)

        client = router._resolve_client(None)
        assert client.upstream_base_url == "http://default:8000"

    def test_question_mark_glob(self):
        """
        ``fnmatch`` ``?`` matches a single character.
        """
        default = UpstreamConfig(url="http://default:8000")
        routes = {"model-?": UpstreamConfig(url="http://matched:8125")}
        router = OutflowRouter(default_upstream=default, model_routes=routes)

        assert router._resolve_client("model-A").upstream_base_url == "http://matched:8125"
        assert router._resolve_client("model-AB").upstream_base_url == "http://default:8000"


# ---------------------------------------------------------------------------
# Client pool deduplication
# ---------------------------------------------------------------------------


class TestClientPoolDeduplication:
    """
    Verify that duplicate upstream URLs share a single ``OutflowClient``.
    """

    def test_same_url_deduplicates(self):
        """
        Two routes with the same upstream URL should share one ``OutflowClient``.
        """
        default = UpstreamConfig(url="http://shared:8000")
        routes = {
            "model-a": UpstreamConfig(url="http://shared:8000"),
            "model-b": UpstreamConfig(url="http://shared:8000"),
        }
        router = OutflowRouter(default_upstream=default, model_routes=routes)
        assert len(router._clients) == 1

    def test_different_urls_separate(self):
        """
        Routes with distinct URLs each get their own ``OutflowClient``.
        """
        default = UpstreamConfig(url="http://default:8000")
        routes = {
            "model-a": UpstreamConfig(url="http://host-a:8001"),
            "model-b": UpstreamConfig(url="http://host-b:8002"),
        }
        router = OutflowRouter(default_upstream=default, model_routes=routes)
        assert len(router._clients) == 3  # default + host-a + host-b

    def test_trailing_slash_normalization(self):
        """
        URLs differing only by trailing slash should deduplicate.
        """
        default = UpstreamConfig(url="http://host:8000/")
        routes = {"model-a": UpstreamConfig(url="http://host:8000")}
        router = OutflowRouter(default_upstream=default, model_routes=routes)
        assert len(router._clients) == 1


# ---------------------------------------------------------------------------
# Start / stop lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    """
    Verify ``start`` and ``stop`` are forwarded to all pooled clients.
    """

    @pytest.mark.asyncio
    async def test_start_calls_all_clients(self):
        """
        ``start()`` invokes ``start()`` on every pooled ``OutflowClient``.
        """
        default = UpstreamConfig(url="http://a:8000")
        routes = {"model-b": UpstreamConfig(url="http://b:8001")}
        router = OutflowRouter(default_upstream=default, model_routes=routes)

        for client in router._clients.values():
            client.start = AsyncMock()
        await router.start()
        for client in router._clients.values():
            client.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_calls_all_clients(self):
        """
        ``stop()`` invokes ``stop()`` on every pooled ``OutflowClient``.
        """
        default = UpstreamConfig(url="http://a:8000")
        routes = {"model-b": UpstreamConfig(url="http://b:8001")}
        router = OutflowRouter(default_upstream=default, model_routes=routes)

        for client in router._clients.values():
            client.stop = AsyncMock()
        await router.stop()
        for client in router._clients.values():
            client.stop.assert_awaited_once()


# ---------------------------------------------------------------------------
# forward_request dispatch
# ---------------------------------------------------------------------------


class TestForwardRequest:
    """
    Verify ``forward_request`` extracts the model and routes correctly.
    """

    @pytest.mark.asyncio
    async def test_dispatches_by_model(self):
        """
        ``forward_request`` routes to the correct client based on the model field.
        """
        default = UpstreamConfig(url="http://default:8000")
        routes = {"special-model": UpstreamConfig(url="http://special:8001")}
        router = OutflowRouter(default_upstream=default, model_routes=routes)

        expected_response = _make_response()
        for client in router._clients.values():
            client.start = AsyncMock()
            client.forward_request = AsyncMock(return_value=expected_response)

        await router.start()
        request = _make_request(model="special-model")
        response = await router.forward_request(request)

        assert response == expected_response
        # Only the special client should have been called
        special_client = router._clients["http://special:8001"]
        special_client.forward_request.assert_awaited_once_with(request)

    @pytest.mark.asyncio
    async def test_no_model_uses_default(self):
        """
        Requests without a model field are forwarded to the default upstream.
        """
        default = UpstreamConfig(url="http://default:8000")
        routes = {"some-model": UpstreamConfig(url="http://other:8001")}
        router = OutflowRouter(default_upstream=default, model_routes=routes)

        expected_response = _make_response()
        for client in router._clients.values():
            client.start = AsyncMock()
            client.forward_request = AsyncMock(return_value=expected_response)

        await router.start()
        request = _make_request(model=None)
        await router.forward_request(request)

        default_client = router._clients["http://default:8000"]
        default_client.forward_request.assert_awaited_once_with(request)
