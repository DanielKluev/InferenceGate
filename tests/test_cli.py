"""Tests for InferenceGate CLI module."""

import pytest
from click.testing import CliRunner

from inference_gate.cli import main


@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return str(cache_dir)


class TestMainCommand:
    """Tests for main CLI commands."""

    def test_version(self, runner):
        """Test that --version flag prints version and exits cleanly."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self, runner):
        """Test that --help shows available commands including start and replay."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "InferenceGate" in result.output
        assert "start" in result.output
        assert "replay" in result.output

    def test_start_help(self, runner):
        """Test that start --help shows record-and-replay mode description."""
        result = runner.invoke(main, ["start", "--help"])
        assert result.exit_code == 0
        assert "record-and-replay" in result.output

    def test_replay_help(self, runner):
        """Test that replay --help shows replay-only mode description."""
        result = runner.invoke(main, ["replay", "--help"])
        assert result.exit_code == 0
        assert "replay-only" in result.output


class TestCacheCommands:
    """Tests for cache management commands."""

    def test_cache_list_empty(self, runner, temp_cache_dir):
        """Test that listing empty cache shows 'No cached entries found'."""
        result = runner.invoke(main, ["cache", "list", "--cache-dir", temp_cache_dir])
        assert result.exit_code == 0
        assert "No cached entries found" in result.output

    def test_cache_info_empty(self, runner, temp_cache_dir):
        """Test that cache info with empty cache shows 'Total entries: 0'."""
        result = runner.invoke(main, ["cache", "info", "--cache-dir", temp_cache_dir])
        assert result.exit_code == 0
        assert "Total entries: 0" in result.output

    def test_cache_clear_empty(self, runner, temp_cache_dir):
        """Test that clearing empty cache shows 'No cached entries to clear'."""
        result = runner.invoke(main, ["cache", "clear", "--cache-dir", temp_cache_dir, "-y"])
        assert result.exit_code == 0
        assert "No cached entries to clear" in result.output


class TestTestGateCommand:
    """
    Tests for the test-gate command.

    Note: These tests only verify the command structure and error handling.
    Actual connectivity tests require a running InferenceGate instance.
    """

    def test_test_gate_help(self, runner):
        """
        Test that test-gate --help shows correct options (host, port, model, prompt).
        """
        result = runner.invoke(main, ["test-gate", "--help"])
        assert result.exit_code == 0
        assert "host" in result.output
        assert "port" in result.output
        assert "model" in result.output
        assert "prompt" in result.output
        # Should NOT have api-key or upstream — those belong to test-upstream
        assert "api-key" not in result.output
        assert "upstream" not in result.output

    def test_test_gate_connection_refused(self, runner):
        """
        Test that test-gate reports connection error when no instance is running.
        """
        # Use a port that shouldn't have anything running
        result = runner.invoke(main, ["test-gate", "--port", "19999"])
        assert result.exit_code == 1
        assert "Connection refused" in result.output


class TestTestUpstreamCommand:
    """
    Tests for the test-upstream command.

    Note: These tests only verify the command structure and error handling.
    Actual API connectivity tests require mocking the HTTP client.
    """

    def test_test_upstream_help(self, runner):
        """
        Test that test-upstream --help shows correct options (upstream, api-key, model, prompt).
        """
        result = runner.invoke(main, ["test-upstream", "--help"])
        assert result.exit_code == 0
        assert "upstream" in result.output
        assert "api-key" in result.output
        assert "model" in result.output
        assert "prompt" in result.output

    def test_test_upstream_no_api_key_error(self, runner):
        """
        Test that test-upstream command fails with helpful error when no API key provided.
        """
        result = runner.invoke(main, ["test-upstream"])
        assert result.exit_code == 1
        assert "No API key provided" in result.output


class TestConfigCommands:
    """
    Tests for configuration management commands.
    """

    def test_config_help(self, runner):
        """
        Test that config --help shows available subcommands.
        """
        result = runner.invoke(main, ["config", "--help"])
        assert result.exit_code == 0
        assert "show" in result.output
        assert "init" in result.output
        assert "path" in result.output

    def test_config_show(self, runner):
        """
        Test that config show displays current configuration.
        """
        result = runner.invoke(main, ["config", "show"])
        assert result.exit_code == 0
        assert "host:" in result.output
        assert "port:" in result.output
        assert "upstream:" in result.output

    def test_config_path(self, runner):
        """
        Test that config path shows the configuration file path.
        """
        result = runner.invoke(main, ["config", "path"])
        assert result.exit_code == 0
        assert ".InferenceGate" in result.output
        assert "config.yaml" in result.output

    def test_config_init(self, runner, tmp_path):
        """
        Test that config init creates a configuration file.
        """
        config_path = tmp_path / "test_config.yaml"
        result = runner.invoke(main, ["--config", str(config_path), "config", "init"])
        assert result.exit_code == 0
        assert "Created default configuration file" in result.output
        assert config_path.exists()

    def test_config_init_no_overwrite(self, runner, tmp_path):
        """
        Test that config init doesn't overwrite existing file without --force.
        """
        config_path = tmp_path / "test_config.yaml"
        config_path.touch()

        result = runner.invoke(main, ["--config", str(config_path), "config", "init"])
        assert result.exit_code == 0
        assert "already exists" in result.output

    def test_config_init_force(self, runner, tmp_path):
        """
        Test that config init --force overwrites existing file.
        """
        config_path = tmp_path / "test_config.yaml"
        config_path.touch()

        result = runner.invoke(main, ["--config", str(config_path), "config", "init", "--force"])
        assert result.exit_code == 0
        assert "Created default configuration file" in result.output

    def test_custom_config_path(self, runner, tmp_path):
        """
        Test that --config option uses custom config path.
        """
        config_path = tmp_path / "custom_config.yaml"
        result = runner.invoke(main, ["--config", str(config_path), "config", "path"])
        assert result.exit_code == 0
        assert str(config_path) in result.output
