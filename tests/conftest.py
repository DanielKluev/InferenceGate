"""Pytest configuration and fixtures for InferenceGate tests."""

import pytest

from inference_gate.recording.storage import CacheStorage


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture
def storage(temp_cache_dir):
    """Create a CacheStorage instance with a temporary directory."""
    return CacheStorage(temp_cache_dir)
