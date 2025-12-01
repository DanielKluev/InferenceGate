"""Tests for InferenceReplay CLI module."""

import pytest
from click.testing import CliRunner

from inference_replay.cli import main


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
        """Test version flag."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self, runner):
        """Test help output."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "InferenceReplay" in result.output
        assert "record" in result.output
        assert "dev" in result.output
        assert "replay" in result.output


class TestCacheCommands:
    """Tests for cache management commands."""

    def test_cache_list_empty(self, runner, temp_cache_dir):
        """Test listing empty cache."""
        result = runner.invoke(main, ["cache", "list", "--cache-dir", temp_cache_dir])
        assert result.exit_code == 0
        assert "No cached entries found" in result.output

    def test_cache_info_empty(self, runner, temp_cache_dir):
        """Test cache info with empty cache."""
        result = runner.invoke(main, ["cache", "info", "--cache-dir", temp_cache_dir])
        assert result.exit_code == 0
        assert "Total entries: 0" in result.output

    def test_cache_clear_empty(self, runner, temp_cache_dir):
        """Test clearing empty cache."""
        result = runner.invoke(main, ["cache", "clear", "--cache-dir", temp_cache_dir, "-y"])
        assert result.exit_code == 0
        assert "No cached entries to clear" in result.output
