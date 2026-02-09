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
