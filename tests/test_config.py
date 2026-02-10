"""
Tests for the configuration management module.

Tests config loading, saving, and default handling for InferenceGate.
"""

import os
from pathlib import Path

import pytest
import yaml

from inference_gate.config import Config, ConfigManager


class TestConfig:
    """
    Tests for the Config model.
    """

    def test_default_values(self) -> None:
        """
        Test that Config has correct default values.
        """
        config = Config()

        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.upstream == "https://api.openai.com"
        assert config.api_key is None
        assert config.cache_dir == ".inference_cache"
        assert config.verbose is False
        assert config.test_model == "gpt-4o-mini"
        assert "OK." in config.test_prompt

    def test_custom_values(self) -> None:
        """
        Test that Config accepts custom values.
        """
        config = Config(host="0.0.0.0", port=9000, upstream="https://custom.api.com", api_key="test-key", cache_dir="./custom_cache",
                        verbose=True, test_model="gpt-4", test_prompt="Custom prompt")

        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.upstream == "https://custom.api.com"
        assert config.api_key == "test-key"
        assert config.cache_dir == "./custom_cache"
        assert config.verbose is True
        assert config.test_model == "gpt-4"
        assert config.test_prompt == "Custom prompt"


class TestConfigManager:
    """
    Tests for the ConfigManager class.
    """

    def test_default_config_path(self) -> None:
        """
        Test that default config path is in user directory.
        """
        manager = ConfigManager()
        path = manager.get_config_path()

        # Should be in user's home directory under .InferenceGate
        assert ".InferenceGate" in str(path)
        assert path.name == "config.yaml"

    def test_custom_config_path(self, tmp_path: Path) -> None:
        """
        Test that custom config path is used when specified.
        """
        custom_path = tmp_path / "custom" / "config.yaml"
        manager = ConfigManager(custom_path)

        assert manager.get_config_path() == custom_path

    def test_load_nonexistent_returns_defaults(self, tmp_path: Path) -> None:
        """
        Test that loading from nonexistent file returns default config.
        """
        config_path = tmp_path / "nonexistent" / "config.yaml"
        manager = ConfigManager(config_path)

        config = manager.load()

        assert config.host == "127.0.0.1"
        assert config.port == 8080

    def test_load_from_file(self, tmp_path: Path) -> None:
        """
        Test that configuration is loaded from YAML file.
        """
        config_path = tmp_path / "config.yaml"
        config_data = {"host": "0.0.0.0", "port": 9000, "cache_dir": "./my_cache"}

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        manager = ConfigManager(config_path)
        config = manager.load()

        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.cache_dir == "./my_cache"
        # Other values should be defaults
        assert config.upstream == "https://api.openai.com"
        assert config.verbose is False

    def test_load_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Test that OPENAI_API_KEY environment variable overrides config.
        """
        config_path = tmp_path / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Create config file without api_key
        with open(config_path, "w") as f:
            yaml.dump({"port": 8080}, f)

        # Set environment variable
        monkeypatch.setenv("OPENAI_API_KEY", "env-api-key-123")

        manager = ConfigManager(config_path)
        config = manager.load()

        # API key should come from environment
        assert config.api_key == "env-api-key-123"

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        """
        Test that save creates parent directories if needed.
        """
        config_path = tmp_path / "deep" / "nested" / "config.yaml"
        manager = ConfigManager(config_path)

        config = Config(port=9000)
        manager.save(config)

        assert config_path.exists()

    def test_save_does_not_store_api_key(self, tmp_path: Path) -> None:
        """
        Test that api_key is not stored in config file for security.
        """
        config_path = tmp_path / "config.yaml"
        manager = ConfigManager(config_path)

        config = Config(api_key="secret-key-123")
        manager.save(config)

        # Read file contents directly
        with open(config_path, "r") as f:
            saved_data = yaml.safe_load(f)

        assert "api_key" not in saved_data

    def test_exists(self, tmp_path: Path) -> None:
        """
        Test that exists() accurately reports file existence.
        """
        config_path = tmp_path / "config.yaml"
        manager = ConfigManager(config_path)

        assert manager.exists() is False

        # Create the file
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.touch()

        assert manager.exists() is True

    def test_create_default(self, tmp_path: Path) -> None:
        """
        Test that create_default creates a config file with default values.
        """
        config_path = tmp_path / "config.yaml"
        manager = ConfigManager(config_path)

        config = manager.create_default()

        assert config_path.exists()
        assert config.host == "127.0.0.1"
        assert config.port == 8080

        # Verify file contents
        with open(config_path, "r") as f:
            saved_data = yaml.safe_load(f)

        assert saved_data["host"] == "127.0.0.1"
        assert saved_data["port"] == 8080
