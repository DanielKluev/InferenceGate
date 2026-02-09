"""
Recording component - handles storage and retrieval of captured inferences.

Key classes: `CacheStorage`, `CachedRequest`, `CachedResponse`, `CacheEntry`
"""

from inference_gate.recording.storage import CachedRequest, CachedResponse, CacheEntry, CacheStorage

__all__ = ["CachedRequest", "CachedResponse", "CacheEntry", "CacheStorage"]
