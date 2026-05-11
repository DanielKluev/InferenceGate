"""
Tests for the ``OutflowRouter`` model-based multi-upstream routing.

Validates exact-name matching, glob-pattern matching with specificity and
declaration-order tie-breaking, the offline sentinel and unrouted-model
sentinel responses, client pool deduplication, and the
``start``/``stop``/``forward_request`` lifecycle.
"""

from unittest.mock import AsyncMock

import pytest

from inference_gate.outflow.model_router import EndpointConfig, ModelRoute, OutflowRouter
from inference_gate.recording.storage import CachedRequest, CachedResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(model: str | None = None) -> CachedRequest:
    """
    Build a minimal ``CachedRequest`` with an optional model field.
    """
    body: dict = {"messages": [{"role": "user", "content": "hi"}], "stream": False}
    if model is not None:
        body["model"] = model
    return CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body=body)


def _make_response() -> CachedResponse:
    """
    Build a minimal ``CachedResponse`` stub.
    """
    return CachedResponse(status_code=200, headers={}, body={"choices": []}, is_streaming=False)


def _build(endpoints: dict[str, EndpointConfig], routes: list[tuple[str, str | None]]) -> OutflowRouter:
    """
    Convenience factory: build an ``OutflowRouter`` from a list of
    ``(pattern, endpoint_name)`` tuples (declaration order = list order).
    """
    return OutflowRouter(endpoints=endpoints, routes=[ModelRoute(pattern=p, endpoint_name=ep) for p, ep in routes])


# ---------------------------------------------------------------------------
# Route resolution tests
# ---------------------------------------------------------------------------


class TestRouteResolution:
    """
    Verify that ``_resolve_route`` dispatches to the correct upstream for
    exact names, glob patterns with specificity ranking, and the offline /
    unrouted sentinels.
    """

    def test_exact_match(self):
        """
        An exact model name match returns the route-specific endpoint.
        """
        endpoints = {
            "default": EndpointConfig(url="http://default:8000"),
            "llamacpp": EndpointConfig(url="http://llamacpp:8125"),
        }
        router = _build(endpoints, [("Gemma4:E4B-it-Q4_K_M", "llamacpp"), ("*", "default")])

        route = router._resolve_route("Gemma4:E4B-it-Q4_K_M")
        assert route is not None
        assert route.endpoint_name == "llamacpp"

    def test_glob_match(self):
        """
        A glob pattern like ``"Gemma4:*"`` matches model names that start with ``"Gemma4:"``.
        """
        endpoints = {
            "default": EndpointConfig(url="http://default:8000"),
            "llamacpp": EndpointConfig(url="http://llamacpp:8125"),
        }
        router = _build(endpoints, [("Gemma4:*", "llamacpp"), ("*", "default")])

        route = router._resolve_route("Gemma4:E4B-it-Q4_K_M")
        assert route is not None
        assert route.endpoint_name == "llamacpp"

    def test_exact_takes_priority_over_glob(self):
        """
        When both an exact match and a glob match exist, exact wins.
        """
        endpoints = {
            "exact": EndpointConfig(url="http://exact:8125"),
            "glob": EndpointConfig(url="http://glob:8126"),
        }
        router = _build(endpoints, [("Gemma4:E4B-it-Q4_K_M", "exact"), ("Gemma4:*", "glob")])

        route = router._resolve_route("Gemma4:E4B-it-Q4_K_M")
        assert route is not None
        assert route.endpoint_name == "exact"

    def test_more_specific_glob_wins(self):
        """
        Among multiple glob matches, the one with more non-wildcard chars wins.
        """
        endpoints = {
            "broad": EndpointConfig(url="http://broad:8000"),
            "narrow": EndpointConfig(url="http://narrow:8001"),
            "default": EndpointConfig(url="http://default:8002"),
        }
        # ``Gemma4:E4B-*`` has more literal characters than ``Gemma4:*`` and
        # thus wins for ``Gemma4:E4B-it-Q4_K_M``.
        router = _build(endpoints, [("Gemma4:*", "broad"), ("Gemma4:E4B-*", "narrow"), ("*", "default")])

        route = router._resolve_route("Gemma4:E4B-it-Q4_K_M")
        assert route is not None
        assert route.endpoint_name == "narrow"

    def test_declaration_order_breaks_specificity_tie(self):
        """
        Two equally-specific globs are tie-broken by declaration order
        (earlier wins).
        """
        endpoints = {
            "a": EndpointConfig(url="http://a:8000"),
            "b": EndpointConfig(url="http://b:8001"),
        }
        router = _build(endpoints, [("model-?", "a"), ("model-*", "b")])
        # ``model-?`` (specificity 6) > ``model-*`` (specificity 6) — both
        # have 6 literal chars so declaration order wins.  ``model-A`` matches
        # both.
        route = router._resolve_route("model-A")
        assert route is not None
        assert route.endpoint_name == "a"

    def test_unrouted_model_returns_none(self):
        """
        When no exact or glob pattern matches, ``_resolve_route`` returns ``None``.
        """
        endpoints = {"llamacpp": EndpointConfig(url="http://llamacpp:8125")}
        router = _build(endpoints, [("Gemma4:*", "llamacpp")])

        assert router._resolve_route("totally-unknown-model") is None

    def test_offline_sentinel_route(self):
        """
        A route with ``endpoint_name=None`` is the offline sentinel.
        """
        endpoints: dict[str, EndpointConfig] = {}
        router = _build(endpoints, [("offline-model", None)])

        route = router._resolve_route("offline-model")
        assert route is not None
        assert route.endpoint_name is None

    def test_question_mark_glob(self):
        """
        ``fnmatch`` ``?`` matches a single character.
        """
        endpoints = {"matched": EndpointConfig(url="http://matched:8125")}
        router = _build(endpoints, [("model-?", "matched")])

        assert router._resolve_route("model-A") is not None
        assert router._resolve_route("model-AB") is None


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    """
    Verify eager validation of route → endpoint references.
    """

    def test_unknown_endpoint_reference_raises(self):
        """
        A route referencing an undefined endpoint name fails fast.
        """
        with pytest.raises(ValueError, match="references unknown endpoint"):
            _build({"a": EndpointConfig(url="http://a:8000")}, [("*", "missing")])


# ---------------------------------------------------------------------------
# Client pool deduplication
# ---------------------------------------------------------------------------


class TestClientPoolDeduplication:
    """
    Verify that endpoints with identical ``(url, api_key, proxy)`` triples
    share a single ``OutflowClient``.
    """

    def test_same_dedup_key_shares_client(self):
        """
        Two endpoints with the same url/api_key/proxy share one client.
        """
        endpoints = {
            "ep_a": EndpointConfig(url="http://shared:8000", api_key="K"),
            "ep_b": EndpointConfig(url="http://shared:8000", api_key="K"),
        }
        router = _build(endpoints, [("a", "ep_a"), ("b", "ep_b")])
        assert len(router._client_by_key) == 1
        assert router._client_by_endpoint["ep_a"] is router._client_by_endpoint["ep_b"]

    def test_different_api_keys_separate(self):
        """
        Two endpoints with the same URL but different api_keys get separate clients.
        """
        endpoints = {
            "ep_a": EndpointConfig(url="http://shared:8000", api_key="K1"),
            "ep_b": EndpointConfig(url="http://shared:8000", api_key="K2"),
        }
        router = _build(endpoints, [("a", "ep_a"), ("b", "ep_b")])
        assert len(router._client_by_key) == 2

    def test_trailing_slash_normalization(self):
        """
        URLs differing only by trailing slash should deduplicate.
        """
        endpoints = {
            "ep_a": EndpointConfig(url="http://host:8000/"),
            "ep_b": EndpointConfig(url="http://host:8000"),
        }
        router = _build(endpoints, [("a", "ep_a"), ("b", "ep_b")])
        assert len(router._client_by_key) == 1


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
        endpoints = {
            "a": EndpointConfig(url="http://a:8000"),
            "b": EndpointConfig(url="http://b:8001"),
        }
        router = _build(endpoints, [("alpha", "a"), ("beta", "b")])

        for client in router._client_by_key.values():
            client.start = AsyncMock()
        await router.start()
        for client in router._client_by_key.values():
            client.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_calls_all_clients(self):
        """
        ``stop()`` invokes ``stop()`` on every pooled ``OutflowClient``.
        """
        endpoints = {
            "a": EndpointConfig(url="http://a:8000"),
            "b": EndpointConfig(url="http://b:8001"),
        }
        router = _build(endpoints, [("alpha", "a"), ("beta", "b")])

        for client in router._client_by_key.values():
            client.stop = AsyncMock()
        await router.stop()
        for client in router._client_by_key.values():
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
        endpoints = {
            "default": EndpointConfig(url="http://default:8000"),
            "special": EndpointConfig(url="http://special:8001"),
        }
        router = _build(endpoints, [("special-model", "special"), ("*", "default")])

        expected_response = _make_response()
        for client in router._client_by_key.values():
            client.start = AsyncMock()
            client.forward_request = AsyncMock(return_value=expected_response)

        await router.start()
        request = _make_request(model="special-model")
        response = await router.forward_request(request)

        assert response is expected_response
        special_client = router._client_by_endpoint["special"]
        special_client.forward_request.assert_awaited_once_with(request)
        # Default client must NOT have been called.
        default_client = router._client_by_endpoint["default"]
        default_client.forward_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unrouted_model_returns_422(self):
        """
        A request whose model has no matching pattern yields HTTP 422
        ``unrouted_model``.
        """
        endpoints = {"only": EndpointConfig(url="http://only:8000")}
        router = _build(endpoints, [("known-model", "only")])

        response = await router.forward_request(_make_request(model="totally-unknown"))
        assert response.status_code == 422
        assert isinstance(response.body, dict)
        assert response.body["error"]["code"] == "unrouted_model"
        assert response.body["error"]["model"] == "totally-unknown"

    @pytest.mark.asyncio
    async def test_offline_sentinel_returns_503(self):
        """
        A matched route with ``endpoint_name=None`` yields HTTP 503
        ``model_offline``.
        """
        endpoints: dict[str, EndpointConfig] = {}
        router = _build(endpoints, [("offline-model", None)])

        response = await router.forward_request(_make_request(model="offline-model"))
        assert response.status_code == 503
        assert isinstance(response.body, dict)
        assert response.body["error"]["code"] == "model_offline"
        assert response.body["error"]["model"] == "offline-model"
        assert response.body["error"]["matched_pattern"] == "offline-model"

    @pytest.mark.asyncio
    async def test_no_model_falls_through_to_catch_all(self):
        """
        Requests without a model field fall through to the bare ``"*"`` catch-all.
        """
        endpoints = {"default": EndpointConfig(url="http://default:8000")}
        router = _build(endpoints, [("*", "default")])

        expected_response = _make_response()
        for client in router._client_by_key.values():
            client.start = AsyncMock()
            client.forward_request = AsyncMock(return_value=expected_response)

        await router.start()
        response = await router.forward_request(_make_request(model=None))
        assert response is expected_response
        default_client = router._client_by_endpoint["default"]
        default_client.forward_request.assert_awaited_once()
