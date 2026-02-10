"""
Configuration management for InferenceGate.

Handles loading and saving configuration from YAML files.
Default configuration file is stored at `$USERDIR/.InferenceGate/config.yaml`.

Key classes: `Config`, `ConfigManager`
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class Config(BaseModel):
    """
    Configuration model for InferenceGate.

    Stores all configurable options including server settings,
    upstream API configuration, and storage paths.
    """

    # Server settings
    host: str = Field(default="127.0.0.1", description="Host to bind the server to")
    port: int = Field(default=8080, description="Port to run the server on")

    # Upstream API settings
    upstream: str = Field(default="https://api.openai.com", description="Upstream OpenAI API base URL")
    api_key: str | None = Field(default=None, description="OpenAI API key")

    # Storage settings
    cache_dir: str = Field(default=".inference_cache", description="Directory to store cached responses")

    # Logging settings
    verbose: bool = Field(default=False, description="Enable verbose logging")

    # Test command settings
    test_prompt: str = Field(
        default='This is a test prompt. Reply with **ONLY** "OK." to confirm that everything is ok. DO NOT output anything else.',
        description="Default prompt for the test command")
    test_model: str = Field(default="gpt-4o-mini", description="Default model for the test command")


class ConfigManager:
    """
    Manages loading and saving InferenceGate configuration.

    Configuration is loaded from YAML files. The default location is
    `$USERDIR/.InferenceGate/config.yaml`, but a custom path can be specified.
    """

    DEFAULT_CONFIG_DIR = ".InferenceGate"
    DEFAULT_CONFIG_FILE = "config.yaml"

    def __init__(self, config_path: str | Path | None = None) -> None:
        """
        Initialize the configuration manager.

        `config_path` specifies a custom configuration file path.
        If None, the default path `$USERDIR/.InferenceGate/config.yaml` is used.
        """
        self.log = logging.getLogger("ConfigManager")
        if config_path is not None:
            self.config_path = Path(config_path)
        else:
            self.config_path = self._get_default_config_path()

    def _get_default_config_path(self) -> Path:
        """
        Get the default configuration file path.

        Returns `$USERDIR/.InferenceGate/config.yaml`.
        """
        # Use USERPROFILE on Windows, HOME on Unix
        user_dir = os.environ.get("USERPROFILE") or os.environ.get("HOME") or Path.home()
        return Path(user_dir) / self.DEFAULT_CONFIG_DIR / self.DEFAULT_CONFIG_FILE

    def load(self) -> Config:
        """
        Load configuration from the YAML file.

        If the file doesn't exist, returns default configuration.
        Environment variables override file values:
        - OPENAI_API_KEY overrides api_key
        """
        config_dict: dict[str, Any] = {}

        if self.config_path.exists():
            self.log.debug("Loading configuration from %s", self.config_path)
            with open(self.config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded is not None:
                    config_dict = loaded
        else:
            self.log.debug("Configuration file not found at %s, creating with defaults", self.config_path)

        # Environment variable overrides
        env_api_key = os.environ.get("OPENAI_API_KEY")
        if env_api_key and "api_key" not in config_dict:
            config_dict["api_key"] = env_api_key

        config = Config(**config_dict)

        # If the config file didn't exist, save the current values so the user has a file to edit
        if not self.config_path.exists():
            self.save(config)

        return config

    def save(self, config: Config) -> None:
        """
        Save configuration to the YAML file.

        Creates the directory structure if it doesn't exist.
        """
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict, excluding None values and defaults we don't want to persist
        config_dict = config.model_dump(exclude_none=True, exclude_defaults=False)

        # Don't save API key to file for security
        config_dict.pop("api_key", None)

        self.log.debug("Saving configuration to %s", self.config_path)
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def get_config_path(self) -> Path:
        """
        Get the current configuration file path.
        """
        return self.config_path

    def exists(self) -> bool:
        """
        Check if the configuration file exists.
        """
        return self.config_path.exists()

    def create_default(self) -> Config:
        """
        Create and save a default configuration file.

        Returns the created configuration.
        """
        config = Config()
        self.save(config)
        return config
