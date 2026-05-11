"""
Tests for the :class:`InferenceGate` orchestrator class.
"""

import pytest

from inference_gate.inference_gate import InferenceGate
from inference_gate.modes import Mode
from inference_gate.outflow.model_router import EndpointConfig, ModelRoute


class TestInferenceGateInit:
    """
    Tests for :class:`InferenceGate` initialization.
    """

    def test_default_config(self):
        """
        Default-constructed gate exposes sensible defaults and an empty
        endpoints / models routing table.
        """
        gate = InferenceGate()
        assert gate.host == "127.0.0.1"
        assert gate.port == 8080
        assert gate.mode == Mode.RECORD_AND_REPLAY
        assert gate.cache_dir == ".inference_cache"
        assert gate.endpoints == {}
        assert gate.models == []

    def test_custom_config(self):
        """
        Custom-constructed gate honours the new endpoints + models schema.
        """
        endpoints = {"primary": EndpointConfig(url="https://custom.api.com", api_key="sk-test")}
        models = [ModelRoute(pattern="*", endpoint_name="primary")]
        gate = InferenceGate(
            host="0.0.0.0",
            port=9090,
            mode=Mode.REPLAY_ONLY,
            cache_dir="/tmp/cache",
            endpoints=endpoints,
            models=models,
        )
        assert gate.host == "0.0.0.0"
        assert gate.port == 9090
        assert gate.mode == Mode.REPLAY_ONLY
        assert gate.cache_dir == "/tmp/cache"
        assert gate.endpoints == endpoints
        assert gate.models == models


class TestInferenceGateLifecycle:
    """Tests for InferenceGate start/stop lifecycle."""

    async def test_start_creates_components(self, tmp_path):
        """Test that start() creates all required components."""
        cache_dir = str(tmp_path / "cache")
        gate = InferenceGate(mode=Mode.REPLAY_ONLY, cache_dir=cache_dir)
        await gate.start()

        assert gate._storage is not None
        assert gate._router is not None
        assert gate._server is not None
        # In REPLAY_ONLY mode, outflow should be None
        assert gate._outflow is None

        await gate.stop()

    async def test_start_record_mode_creates_outflow(self, tmp_path):
        """
        Starting the gate in RECORD_AND_REPLAY mode constructs an
        :class:`OutflowRouter` from the endpoints + models routing table.
        """
        cache_dir = str(tmp_path / "cache")
        endpoints = {"primary": EndpointConfig(url="https://api.openai.com")}
        models = [ModelRoute(pattern="*", endpoint_name="primary")]
        gate = InferenceGate(mode=Mode.RECORD_AND_REPLAY, cache_dir=cache_dir, endpoints=endpoints, models=models)
        await gate.start()

        assert gate._outflow is not None
        assert gate._router is not None

        await gate.stop()

    async def test_stop_cleans_up(self, tmp_path):
        """Test that stop() properly shuts down all components."""
        cache_dir = str(tmp_path / "cache")
        gate = InferenceGate(mode=Mode.REPLAY_ONLY, cache_dir=cache_dir)
        await gate.start()
        await gate.stop()
        # After stop, we should be able to call stop again without error
        await gate.stop()

    async def test_storage_property(self, tmp_path):
        """Test that storage property returns the CacheStorage instance after start."""
        cache_dir = str(tmp_path / "cache")
        gate = InferenceGate(mode=Mode.REPLAY_ONLY, cache_dir=cache_dir)

        # Before start, storage is None
        assert gate.storage is None

        await gate.start()
        assert gate.storage is not None

        await gate.stop()
