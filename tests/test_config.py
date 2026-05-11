"""
Tests for the configuration management module.

Covers ``Config`` defaults / custom values, ``ConfigManager`` load/save
roundtrips, the new ``endpoints`` + ``models`` schema, and the security
invariant that per-endpoint api_keys are never persisted to disk.
"""

from pathlib import Path

import pytest
import yaml

from inference_gate.config import Config, ConfigManager, EndpointDef, ModelRouteDef


class TestConfig:
    """
    Tests for the ``Config`` Pydantic model.
    """

    def test_default_values(self) -> None:
        """
        ``Config`` exposes sensible defaults for every field.
        """
        config = Config()

        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.cache_dir == ".inference_cache"
        assert config.verbose is False
        assert config.test_model == "gpt-4o-mini"
        assert "OK." in config.test_prompt
        assert config.fuzzy_model is False
        assert config.fuzzy_sampling == "off"
        assert config.max_non_greedy_replies == 5
        assert config.record_timeout == 600.0
        assert config.endpoints == {}
        assert config.models == []
        assert config.non_streaming_models == []

    def test_custom_values(self) -> None:
        """
        ``Config`` accepts custom server, fuzzy, and routing-table values.
        """
        config = Config(host="0.0.0.0", port=9000, cache_dir="./custom_cache", verbose=True, test_model="gpt-4",
                        test_prompt="Custom prompt", record_timeout=120.0,
                        endpoints={"primary": EndpointDef(url="http://127.0.0.1:8001", api_key="K",
                                                          timeout=30.0)}, models=[ModelRouteDef(pattern="*", endpoint="primary")])

        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.cache_dir == "./custom_cache"
        assert config.verbose is True
        assert config.test_model == "gpt-4"
        assert config.test_prompt == "Custom prompt"
        assert config.record_timeout == 120.0
        assert config.endpoints["primary"].url == "http://127.0.0.1:8001"
        assert config.endpoints["primary"].api_key == "K"
        assert config.endpoints["primary"].timeout == 30.0
        assert config.models[0].pattern == "*"
        assert config.models[0].endpoint == "primary"

    def test_offline_sentinel_route(self) -> None:
        """
        A ``ModelRouteDef`` may have ``endpoint=None`` to mark an offline sentinel.
        """
        route = ModelRouteDef(pattern="offline-model", endpoint=None)
        assert route.endpoint is None


class TestConfigManager:
    """
    Tests for :class:`ConfigManager` load / save / default-path behaviour.
    """

    def test_default_config_path(self) -> None:
        """
        Without an explicit path, ConfigManager uses ``~/.InferenceGate/config.yaml``.
        """
        manager = ConfigManager()
        path = manager.get_config_path()

        assert ".InferenceGate" in str(path)
        assert path.name == "config.yaml"

    def test_custom_config_path(self, tmp_path: Path) -> None:
        """
        An explicit path is honored verbatim.
        """
        custom_path = tmp_path / "custom" / "config.yaml"
        manager = ConfigManager(custom_path)

        assert manager.get_config_path() == custom_path

    def test_load_nonexistent_returns_defaults(self, tmp_path: Path) -> None:
        """
        Loading from a nonexistent file yields a default :class:`Config`.
        """
        config_path = tmp_path / "nonexistent" / "config.yaml"
        manager = ConfigManager(config_path)

        config = manager.load()

        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.endpoints == {}
        assert config.models == []

    def test_load_endpoints_and_models_from_file(self, tmp_path: Path) -> None:
        """
        ``endpoints`` and ``models`` round-trip through YAML correctly.
        """
        config_path = tmp_path / "config.yaml"
        config_data = {
            "host":
                "0.0.0.0",
            "port":
                9000,
            "cache_dir":
                "./my_cache",
            "endpoints": {
                "local_spark": {
                    "url": "http://127.0.0.1:8001",
                    "timeout": 90.0
                },
                "vast_vllm": {
                    "url": "http://10.0.0.1:8000",
                    "proxy": "http://127.0.0.1:8888"
                },
            },
            "models": [
                {
                    "pattern": "Gemma4:*",
                    "endpoint": "local_spark"
                },
                {
                    "pattern": "Qwen3:*",
                    "endpoint": "vast_vllm"
                },
                {
                    "pattern": "offline-only",
                    "endpoint": None
                },
            ],
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        manager = ConfigManager(config_path)
        config = manager.load()

        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.cache_dir == "./my_cache"
        assert set(config.endpoints) == {"local_spark", "vast_vllm"}
        assert config.endpoints["local_spark"].timeout == 90.0
        assert config.endpoints["vast_vllm"].proxy == "http://127.0.0.1:8888"
        assert [r.pattern for r in config.models] == ["Gemma4:*", "Qwen3:*", "offline-only"]
        assert config.models[2].endpoint is None

    def test_load_legacy_upstream_schema_migrates_to_routes(self, tmp_path: Path) -> None:
        """
        Legacy ``upstream`` / ``model_routes`` YAML is translated to ``endpoints`` / ``models``.
        """
        config_path = tmp_path / "config.yaml"
        config_data = {
            "upstream": "http://default.example:8000/",
            "api_key": "default-key",
            "proxy": "http://proxy.example:8888/",
            "upstream_timeout": 3600.0,
            "model_routes": {
                "Gemma4:*": {
                    "upstream": "http://llamacpp.example:8125/",
                    "api_key": "route-key",
                    "timeout": 90.0,
                }
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = ConfigManager(config_path).load()

        assert config.record_timeout == 3600.0
        assert set(config.endpoints) == {"legacy_default", "legacy_route_0"}
        assert config.endpoints["legacy_default"].url == "http://default.example:8000/"
        assert config.endpoints["legacy_default"].api_key == "default-key"
        assert config.endpoints["legacy_default"].proxy == "http://proxy.example:8888/"
        assert config.endpoints["legacy_route_0"].url == "http://llamacpp.example:8125/"
        assert config.endpoints["legacy_route_0"].api_key == "route-key"
        assert config.endpoints["legacy_route_0"].timeout == 90.0
        assert [(route.pattern, route.endpoint) for route in config.models] == [("Gemma4:*", "legacy_route_0"), ("*", "legacy_default")]

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        """
        ``save`` creates parent directories if needed.
        """
        config_path = tmp_path / "deep" / "nested" / "config.yaml"
        manager = ConfigManager(config_path)

        config = Config(port=9000)
        manager.save(config)

        assert config_path.exists()

    def test_save_strips_endpoint_api_keys(self, tmp_path: Path) -> None:
        """
        Per-endpoint ``api_key`` fields are NEVER persisted to disk.
        """
        config_path = tmp_path / "config.yaml"
        manager = ConfigManager(config_path)

        config = Config(endpoints={"primary": EndpointDef(url="http://127.0.0.1:8001", api_key="secret-key-123")},
                        models=[ModelRouteDef(pattern="*", endpoint="primary")])
        manager.save(config)

        with open(config_path, "r") as f:
            saved_data = yaml.safe_load(f)

        # Endpoint must be present, but its api_key must be stripped.
        assert "primary" in saved_data["endpoints"]
        assert "api_key" not in saved_data["endpoints"]["primary"]
        assert saved_data["endpoints"]["primary"]["url"] == "http://127.0.0.1:8001"

    def test_save_load_roundtrip_preserves_routing_table(self, tmp_path: Path) -> None:
        """
        Save → load preserves the routing table (modulo stripped api_keys).
        """
        config_path = tmp_path / "config.yaml"
        manager = ConfigManager(config_path)

        config = Config(
            endpoints={
                "a": EndpointDef(url="http://a:8000"),
                "b": EndpointDef(url="http://b:8001", proxy="http://proxy:9000"),
            }, models=[
                ModelRouteDef(pattern="model-a-*", endpoint="a"),
                ModelRouteDef(pattern="model-b", endpoint="b"),
                ModelRouteDef(pattern="dead-model", endpoint=None),
            ])
        manager.save(config)
        loaded = manager.load()

        assert set(loaded.endpoints) == {"a", "b"}
        assert loaded.endpoints["b"].proxy == "http://proxy:9000"
        assert [(r.pattern, r.endpoint) for r in loaded.models] == [
            ("model-a-*", "a"),
            ("model-b", "b"),
            ("dead-model", None),
        ]

    def test_exists(self, tmp_path: Path) -> None:
        """
        ``exists()`` accurately reports config-file presence.
        """
        config_path = tmp_path / "config.yaml"
        manager = ConfigManager(config_path)

        assert manager.exists() is False

        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.touch()

        assert manager.exists() is True

    def test_create_default(self, tmp_path: Path) -> None:
        """
        ``create_default`` writes a default :class:`Config` to disk.
        """
        config_path = tmp_path / "config.yaml"
        manager = ConfigManager(config_path)

        config = manager.create_default()

        assert config_path.exists()
        assert config.host == "127.0.0.1"
        assert config.port == 8080

        with open(config_path, "r") as f:
            saved_data = yaml.safe_load(f)

        assert saved_data["host"] == "127.0.0.1"
        assert saved_data["port"] == 8080

    def test_load_does_not_pull_openai_api_key_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        The legacy top-level ``api_key`` field is gone; ``OPENAI_API_KEY`` no
        longer leaks into the config.  Per-endpoint api_keys must be supplied
        via the runtime ``/gate/config`` POST or env-substituted YAML.
        """
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"port": 8080}, f)

        monkeypatch.setenv("OPENAI_API_KEY", "should-not-leak")

        manager = ConfigManager(config_path)
        config = manager.load()

        # No top-level api_key field exists on Config any more.
        assert not hasattr(config, "api_key") or getattr(config, "api_key", None) != "should-not-leak"
        # Endpoints map starts empty and remains so.
        assert config.endpoints == {}
