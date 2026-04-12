"""
Recording component — handles storage and retrieval of captured inferences.

Uses the v2 tape format: human-readable `.tape` cassette files with YAML
frontmatter + MIME body, content-addressed response/asset storage, and a
TSV index for multi-tier fuzzy lookup.

Key classes: `CacheStorage`, `CachedRequest`, `CachedResponse`, `CacheEntry`,
    `TapeMetadata`, `TapeIndex`, `IndexRow`
Key functions: `reassemble_streaming_response`, `compute_content_hash`,
    `compute_prompt_model_hash`, `compute_prompt_hash`
"""

from inference_gate.recording.hashing import compute_content_hash, compute_prompt_hash, compute_prompt_model_hash
from inference_gate.recording.models import IndexRow, TapeMetadata
from inference_gate.recording.reassembly import reassemble_streaming_response
from inference_gate.recording.storage import CachedRequest, CachedResponse, CacheEntry, CacheStorage
from inference_gate.recording.tape_index import TapeIndex

__all__ = [
    "CachedRequest", "CachedResponse", "CacheEntry", "CacheStorage",
    "TapeMetadata", "TapeIndex", "IndexRow",
    "reassemble_streaming_response",
    "compute_content_hash", "compute_prompt_model_hash", "compute_prompt_hash",
]
