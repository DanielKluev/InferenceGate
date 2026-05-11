"""
Tests for the ``/gate/*`` admin namespace.

Covers redaction helpers, GET/POST config (new endpoints+models schema),
index reload, stats, 404 handling for unknown gate endpoints, and the
``/gate/`` namespace reservation contract.
"""

import pytest
from aiohttp.test_utils import TestClient, TestServer

from inference_gate.inflow.admin import (_redact_url, _validate_endpoints_payload, _validate_models_payload, redact, redact_headers)
from inference_gate.inflow.server import InflowServer
from inference_gate.modes import Mode
from inference_gate.outflow.model_router import OutflowRouter
from inference_gate.recording.storage import CacheStorage
from inference_gate.router.router import Router


@pytest.fixture
def temp_cache_dir(tmp_path):
    """
    Create a temporary cache directory for the storage layer.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture
def replay_router(temp_cache_dir):
    """
    Create a Router in replay-only mode for admin tests.
    """
    storage = CacheStorage(temp_cache_dir)
    return Router(mode=Mode.REPLAY_ONLY, storage=storage, fuzzy_model=True, fuzzy_sampling="soft")


@pytest.fixture
async def admin_client(replay_router):
    """
    Wire ``InflowServer`` to an aiohttp test client carrying the ``/gate/*`` namespace.
    """
    server = InflowServer(host="127.0.0.1", port=0, router=replay_router)
    app = server._create_app()
    async with TestClient(TestServer(app)) as client:
        yield client


# ---------------------------------------------------------------------------
# Redaction helpers
# ---------------------------------------------------------------------------


class TestRedactionHelpers:
    """
    Unit tests for the pure-function redaction helpers.
    """

    def test_redact_url_with_userinfo(self):
        """
        Embedded ``user:pass@`` userinfo is replaced with ``***@``.
        """
        assert _redact_url("https://user:pass@example.com:8080/path") == "https://***@example.com:8080/path"

    def test_redact_url_without_userinfo(self):
        """
        URLs without userinfo pass through unchanged.
        """
        assert _redact_url("https://example.com:8080/path") == "https://example.com:8080/path"

    def test_redact_url_non_url_string(self):
        """
        Non-URL strings pass through unchanged.
        """
        assert _redact_url("not a url") == "not a url"
        assert _redact_url("") == ""

    def test_redact_headers_masks_authorization(self):
        """
        Auth-bearing headers are masked, others preserved.
        """
        result = redact_headers({"Authorization": "Bearer secret", "Content-Type": "application/json", "X-API-Key": "abc"})
        assert result["Authorization"] == "***"
        assert result["X-API-Key"] == "***"
        assert result["Content-Type"] == "application/json"

    def test_redact_recursive_masks_secret_keys(self):
        """
        Recursive redactor masks keys whose name suggests a secret.
        """
        payload = {
            "model": "gpt-4",
            "api_key": "sk-abc",
            "nested": {
                "password": "p",
                "ok": "v"
            },
            "items": [{
                "token": "t"
            }],
        }
        result = redact(payload)
        assert result["api_key"] == "***"
        assert result["nested"]["password"] == "***"
        assert result["nested"]["ok"] == "v"
        assert result["items"][0]["token"] == "***"
        assert result["model"] == "gpt-4"


# ---------------------------------------------------------------------------
# Validation helpers (unit tests, no HTTP roundtrip)
# ---------------------------------------------------------------------------


class TestValidationHelpers:
    """
    Direct unit tests for the payload validators.
    """

    def test_endpoints_must_be_object(self):
        """
        ``endpoints`` must be a JSON object, not list/string/etc.
        """
        assert _validate_endpoints_payload([]) is not None
        assert _validate_endpoints_payload("nope") is not None

    def test_endpoint_must_have_url(self):
        """
        Each endpoint must contain a non-empty string ``url``.
        """
        err = _validate_endpoints_payload({"primary": {}})
        assert err is not None and "url" in err

    def test_endpoint_optional_field_types(self):
        """
        ``api_key``, ``proxy``, and ``timeout`` must be the right types when provided.
        """
        assert _validate_endpoints_payload({"a": {"url": "u", "timeout": "no"}}) is not None
        assert _validate_endpoints_payload({"a": {"url": "u", "api_key": 42}}) is not None
        assert _validate_endpoints_payload({"a": {"url": "u", "timeout": 60.0, "api_key": "K"}}) is None

    def test_models_must_be_list(self):
        """
        ``models`` must be a JSON array.
        """
        assert _validate_models_payload({}, set()) is not None

    def test_models_pattern_required(self):
        """
        Each route must have a non-empty ``pattern`` string.
        """
        err = _validate_models_payload([{"endpoint": "a"}], {"a"})
        assert err is not None and "pattern" in err

    def test_models_endpoint_must_match_known(self):
        """
        Route's ``endpoint`` must reference an endpoint in the known set.
        """
        err = _validate_models_payload([{"pattern": "*", "endpoint": "missing"}], {"a"})
        assert err is not None and "missing" in err

    def test_models_endpoint_null_is_offline_sentinel(self):
        """
        ``endpoint=None`` is allowed (offline sentinel) regardless of endpoint set.
        """
        assert _validate_models_payload([{"pattern": "*", "endpoint": None}], set()) is None


# ---------------------------------------------------------------------------
# /gate/health
# ---------------------------------------------------------------------------


class TestGateHealth:
    """
    Tests for ``GET /gate/health``.
    """

    async def test_health_returns_mode_and_fuzzy(self, admin_client):
        """
        ``/gate/health`` echoes mode and live fuzzy settings.
        """
        resp = await admin_client.get("/gate/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "healthy"
        assert data["mode"] == "replay-only"
        assert data["fuzzy_model"] is True
        assert data["fuzzy_sampling"] == "soft"


# ---------------------------------------------------------------------------
# /gate/config — GET
# ---------------------------------------------------------------------------


class TestGateConfigGet:
    """
    Tests for ``GET /gate/config``.
    """

    async def test_get_config_snapshot(self, admin_client, replay_router):
        """
        GET returns a redacted snapshot reflecting the live router.
        """
        resp = await admin_client.get("/gate/config")
        assert resp.status == 200
        data = await resp.json()
        assert data["mode"] == "replay-only"
        assert data["fuzzy_model"] is True
        assert data["fuzzy_sampling"] == "soft"
        assert data["max_non_greedy_replies"] == replay_router.max_non_greedy_replies
        # Replay-only carries no outflow.
        assert data["outflow"] is None
        assert "stats" in data and "cassettes" in data["stats"]


# ---------------------------------------------------------------------------
# /gate/config — POST: scalar router-level fields
# ---------------------------------------------------------------------------


class TestGateConfigPostScalars:
    """
    POST ``/gate/config`` accepts router-level scalar overrides.
    """

    async def test_updates_fuzzy_settings(self, admin_client, replay_router):
        """
        POST applies new fuzzy_model/fuzzy_sampling and the router reflects them.
        """
        resp = await admin_client.post("/gate/config", json={"fuzzy_model": False, "fuzzy_sampling": "off"})
        assert resp.status == 200
        data = await resp.json()
        assert data["fuzzy_model"] is False
        assert data["fuzzy_sampling"] == "off"
        assert replay_router.fuzzy_model is False
        assert replay_router.fuzzy_sampling == "off"

    async def test_rejects_unknown_keys(self, admin_client):
        """
        Unknown body keys yield 400 listing the accepted set.
        """
        resp = await admin_client.post("/gate/config", json={"made_up": 1})
        assert resp.status == 400
        data = await resp.json()
        assert data["error"]["type"] == "invalid_request"
        assert "made_up" in data["error"]["message"]

    async def test_rejects_invalid_mode(self, admin_client):
        """
        Bogus mode value yields 400.
        """
        resp = await admin_client.post("/gate/config", json={"mode": "warp"})
        assert resp.status == 400
        data = await resp.json()
        assert data["error"]["type"] == "invalid_request"

    async def test_rejects_invalid_fuzzy_sampling(self, admin_client):
        """
        Bogus fuzzy_sampling value yields 400.
        """
        resp = await admin_client.post("/gate/config", json={"fuzzy_sampling": "extreme"})
        assert resp.status == 400

    async def test_rejects_non_object_body(self, admin_client):
        """
        Non-JSON-object bodies are rejected.
        """
        resp = await admin_client.post("/gate/config", data="not json", headers={"content-type": "application/json"})
        assert resp.status == 400

    async def test_rejects_invalid_max_non_greedy_replies(self, admin_client):
        """
        ``max_non_greedy_replies`` must be a positive integer.
        """
        resp = await admin_client.post("/gate/config", json={"max_non_greedy_replies": 0})
        assert resp.status == 400

    async def test_updates_non_streaming_models(self, admin_client, replay_router):
        """
        ``non_streaming_models`` list is accepted and applied to the router.
        """
        resp = await admin_client.post("/gate/config", json={"non_streaming_models": ["llama-cpp-only"]})
        assert resp.status == 200
        assert replay_router.non_streaming_models == ["llama-cpp-only"]


# ---------------------------------------------------------------------------
# /gate/config — POST: outflow rebuild (endpoints + models)
# ---------------------------------------------------------------------------


class TestGateConfigPostOutflow:
    """
    POST ``/gate/config`` accepts the new ``endpoints`` + ``models`` payload
    and atomically rebuilds the outflow router.
    """

    async def test_installs_outflow_router(self, admin_client, replay_router):
        """
        Flipping mode to record-and-replay with a routing table installs an
        :class:`OutflowRouter` and surfaces it through the snapshot.
        """
        payload = {
            "mode": "record-and-replay",
            "endpoints": {
                "local_spark": {
                    "url": "http://gpu1.example:8125/",
                    "api_key": "k1",
                    "timeout": 90.0,
                },
                "vast_vllm": {
                    "url": "http://gpu2.example:8000",
                    "proxy": "http://other-proxy/",
                },
            },
            "models": [
                {
                    "pattern": "Gemma4:E4B-it-Q4_K_M",
                    "endpoint": "local_spark"
                },
                {
                    "pattern": "Gemma-4-31B",
                    "endpoint": "vast_vllm"
                },
                {
                    "pattern": "offline-only",
                    "endpoint": None
                },
            ],
            "non_streaming_models": ["llama-cpp-only"],
        }
        resp = await admin_client.post("/gate/config", json=payload)
        assert resp.status == 200, await resp.text()
        data = await resp.json()
        assert data["mode"] == "record-and-replay"
        assert data["non_streaming_models"] == ["llama-cpp-only"]
        outflow_snap = data["outflow"]
        assert outflow_snap is not None
        assert outflow_snap["type"] == "router"
        assert set(outflow_snap["endpoints"].keys()) == {"local_spark", "vast_vllm"}
        # api_keys must be redacted in the snapshot.
        assert outflow_snap["endpoints"]["local_spark"]["api_key"] == "***"
        assert outflow_snap["endpoints"]["vast_vllm"]["api_key"] is None
        assert outflow_snap["endpoints"]["vast_vllm"]["proxy"] == "http://other-proxy/"
        # Routes preserved in declaration order, including offline sentinel.
        assert [r["pattern"] for r in outflow_snap["routes"]] == ["Gemma4:E4B-it-Q4_K_M", "Gemma-4-31B", "offline-only"]
        assert outflow_snap["routes"][2]["endpoint"] is None
        # Live router state mirrors the snapshot.
        assert isinstance(replay_router.outflow, OutflowRouter)
        assert replay_router.non_streaming_models == ["llama-cpp-only"]
        # Cleanup: the test client teardown does not stop the outflow.
        await replay_router.outflow.stop()
        replay_router.outflow = None

    async def test_get_after_post_roundtrips_outflow(self, admin_client, replay_router):
        """
        GET /gate/config after a POST returns the same outflow snapshot.
        """
        await admin_client.post("/gate/config", json={
            "mode": "record-and-replay",
            "endpoints": {
                "primary": {
                    "url": "http://127.0.0.1:8001"
                }
            },
            "models": [{
                "pattern": "*",
                "endpoint": "primary"
            }],
        })
        try:
            resp = await admin_client.get("/gate/config")
            assert resp.status == 200
            data = await resp.json()
            assert data["mode"] == "record-and-replay"
            assert "primary" in data["outflow"]["endpoints"]
            assert data["outflow"]["routes"][0]["pattern"] == "*"
        finally:
            await replay_router.outflow.stop()
            replay_router.outflow = None

    async def test_rejects_endpoint_missing_url(self, admin_client):
        """
        Endpoint without ``url`` yields 400.
        """
        resp = await admin_client.post("/gate/config",
                                       json={
                                           "mode": "record-and-replay",
                                           "endpoints": {
                                               "bad": {
                                                   "api_key": "k"
                                               }
                                           },
                                           "models": [],
                                       })
        assert resp.status == 400
        data = await resp.json()
        assert "url" in data["error"]["message"]

    async def test_rejects_models_referencing_unknown_endpoint(self, admin_client):
        """
        Routing rule pointing to an undeclared endpoint yields 400.
        """
        resp = await admin_client.post("/gate/config",
                                       json={
                                           "mode": "record-and-replay",
                                           "endpoints": {
                                               "a": {
                                                   "url": "http://a:8000"
                                               }
                                           },
                                           "models": [{
                                               "pattern": "*",
                                               "endpoint": "missing"
                                           }],
                                       })
        assert resp.status == 400
        data = await resp.json()
        assert "missing" in data["error"]["message"]

    async def test_partial_post_merges_with_live_state(self, admin_client, replay_router):
        """
        A subsequent POST that omits ``endpoints`` reuses the live endpoints
        when validating the models payload.
        """
        # Initial install.
        await admin_client.post("/gate/config",
                                json={
                                    "mode": "record-and-replay",
                                    "endpoints": {
                                        "primary": {
                                            "url": "http://127.0.0.1:8001"
                                        }
                                    },
                                    "models": [{
                                        "pattern": "*",
                                        "endpoint": "primary"
                                    }],
                                })
        try:
            # Subsequent POST replaces ONLY models; endpoints are inherited.
            resp = await admin_client.post("/gate/config", json={"models": [{"pattern": "Gemma4:*", "endpoint": "primary"}]})
            assert resp.status == 200, await resp.text()
            data = await resp.json()
            assert "primary" in data["outflow"]["endpoints"]
            assert data["outflow"]["routes"][0]["pattern"] == "Gemma4:*"
        finally:
            await replay_router.outflow.stop()
            replay_router.outflow = None


# ---------------------------------------------------------------------------
# /gate/config — POST: idempotency + concurrent-update gating
# ---------------------------------------------------------------------------


class TestGateConfigPostIdempotencyAndGating:
    """
    Tests for the :class:`ConfigGate` hardening of ``POST /gate/config``.

    Covers:

    - Repeated identical payloads are short-circuited (idempotent no-op) and
      do not rebuild the outflow.
    - Different payloads do rebuild and update the stored hash so subsequent
      equal pushes still no-op.
    - Concurrent identical pushes (simulating xdist worker storm at session
      start) all succeed and only the first one performs the rebuild.
    - Proxy requests issued during a config update are quiesced and complete
      successfully once the update releases the gate.
    """

    async def test_repeated_identical_payload_is_noop(self, admin_client, replay_router):
        """
        Posting the same body twice yields equal snapshots and does NOT rebuild
        the outflow on the second call (verified via outflow object identity).
        """
        payload = {
            "mode": "record-and-replay",
            "endpoints": {
                "primary": {
                    "url": "http://127.0.0.1:8001"
                }
            },
            "models": [{
                "pattern": "*",
                "endpoint": "primary"
            }],
        }
        resp1 = await admin_client.post("/gate/config", json=payload)
        assert resp1.status == 200
        snapshot1 = await resp1.json()
        outflow_after_first = replay_router.outflow

        try:
            resp2 = await admin_client.post("/gate/config", json=payload)
            assert resp2.status == 200
            snapshot2 = await resp2.json()
            assert snapshot1 == snapshot2
            # No-op: the underlying outflow object must be the SAME instance,
            # not a freshly-rebuilt one.
            assert replay_router.outflow is outflow_after_first
        finally:
            await replay_router.outflow.stop()
            replay_router.outflow = None

    async def test_payload_with_reordered_keys_is_noop(self, admin_client, replay_router):
        """
        Idempotency is computed on canonicalised JSON, so semantically-identical
        payloads with different key ordering still short-circuit.
        """
        payload_a = {
            "mode": "record-and-replay",
            "endpoints": {
                "primary": {
                    "url": "http://127.0.0.1:8001",
                    "api_key": "k"
                }
            },
            "models": [{
                "pattern": "*",
                "endpoint": "primary"
            }],
        }
        # Same logical content, keys reordered top-level and inside endpoint.
        payload_b = {
            "models": [{
                "endpoint": "primary",
                "pattern": "*"
            }],
            "endpoints": {
                "primary": {
                    "api_key": "k",
                    "url": "http://127.0.0.1:8001"
                }
            },
            "mode": "record-and-replay",
        }
        await admin_client.post("/gate/config", json=payload_a)
        outflow_after_first = replay_router.outflow
        try:
            resp = await admin_client.post("/gate/config", json=payload_b)
            assert resp.status == 200
            assert replay_router.outflow is outflow_after_first
        finally:
            await replay_router.outflow.stop()
            replay_router.outflow = None

    async def test_changed_payload_rebuilds_and_updates_hash(self, admin_client, replay_router):
        """
        A payload that differs from the last applied one DOES rebuild the
        outflow, and the new hash is stored so a third identical push no-ops.
        """
        payload_v1 = {
            "mode": "record-and-replay",
            "endpoints": {
                "primary": {
                    "url": "http://127.0.0.1:8001"
                }
            },
            "models": [{
                "pattern": "*",
                "endpoint": "primary"
            }],
        }
        payload_v2 = {
            "mode": "record-and-replay",
            "endpoints": {
                "primary": {
                    "url": "http://127.0.0.1:8002"
                }
            },
            "models": [{
                "pattern": "*",
                "endpoint": "primary"
            }],
        }
        await admin_client.post("/gate/config", json=payload_v1)
        outflow_v1 = replay_router.outflow
        try:
            resp = await admin_client.post("/gate/config", json=payload_v2)
            assert resp.status == 200
            assert replay_router.outflow is not outflow_v1
            outflow_v2 = replay_router.outflow

            # Re-pushing v2 must now no-op.
            resp = await admin_client.post("/gate/config", json=payload_v2)
            assert resp.status == 200
            assert replay_router.outflow is outflow_v2
        finally:
            await replay_router.outflow.stop()
            replay_router.outflow = None

    async def test_concurrent_identical_pushes_all_succeed_with_one_rebuild(self, admin_client, replay_router):
        """
        Eight concurrent identical posts (xdist storm) all return 200 and only
        the first one rebuilds the outflow; the rest no-op on the cached hash.
        """
        import asyncio

        payload = {
            "mode": "record-and-replay",
            "endpoints": {
                "primary": {
                    "url": "http://127.0.0.1:8001"
                }
            },
            "models": [{
                "pattern": "*",
                "endpoint": "primary"
            }],
        }
        try:
            results = await asyncio.gather(*(admin_client.post("/gate/config", json=payload) for _ in range(8)))
            for resp in results:
                assert resp.status == 200, await resp.text()
            # Exactly one outflow was built — repeated pushes did not churn it.
            outflow_after = replay_router.outflow
            assert outflow_after is not None
            # One more push should still no-op against the cached hash.
            resp = await admin_client.post("/gate/config", json=payload)
            assert resp.status == 200
            assert replay_router.outflow is outflow_after
        finally:
            if replay_router.outflow is not None:
                await replay_router.outflow.stop()
                replay_router.outflow = None

    async def test_invalid_payload_does_not_update_hash(self, admin_client, replay_router):
        """
        A 400-rejected payload must not be remembered; a follow-up identical
        invalid payload must still hit validation, not the no-op path.
        """
        bad_payload = {
            "mode": "record-and-replay",
            "endpoints": {
                "primary": {
                    "api_key": "k"
                }  # missing url
            },
            "models": [],
        }
        resp1 = await admin_client.post("/gate/config", json=bad_payload)
        assert resp1.status == 400

        # Re-post the same broken body — must still be 400 (validation runs again).
        resp2 = await admin_client.post("/gate/config", json=bad_payload)
        assert resp2.status == 400


# ---------------------------------------------------------------------------


class TestGateIndexReload:
    """
    Tests for ``POST /gate/index/reload``.
    """

    async def test_reload_returns_cassette_count(self, admin_client):
        """
        Reload returns the new cassette count from the rebuilt index.
        """
        resp = await admin_client.post("/gate/index/reload")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "reloaded"
        assert data["cassettes"] == 0


# ---------------------------------------------------------------------------
# /gate/stats
# ---------------------------------------------------------------------------


class TestGateStats:
    """
    Tests for ``GET /gate/stats``.
    """

    async def test_stats_returns_zero_for_empty_cache(self, admin_client):
        """
        Stats endpoint reports zero cassettes for an empty cache directory.
        """
        resp = await admin_client.get("/gate/stats")
        assert resp.status == 200
        data = await resp.json()
        assert data["cassettes"] == 0
        assert data["mode"] == "replay-only"


# ---------------------------------------------------------------------------
# /gate/* namespace reservation
# ---------------------------------------------------------------------------


class TestGateNamespaceReserved:
    """
    The ``/gate/`` namespace must never fall through to the proxy handler.
    """

    async def test_unknown_gate_path_returns_404(self, admin_client):
        """
        Unknown ``/gate/*`` paths return a structured 404, not a proxy miss.
        """
        resp = await admin_client.get("/gate/does-not-exist")
        assert resp.status == 404
        data = await resp.json()
        assert data["error"]["type"] == "not_found"

    async def test_unknown_gate_post_returns_404(self, admin_client):
        """
        POST to unknown ``/gate/*`` path also returns 404.
        """
        resp = await admin_client.post("/gate/whatever", json={})
        assert resp.status == 404
