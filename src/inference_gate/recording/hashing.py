"""
Content hashing for the v2 cassette tape storage format.

Provides three tiers of hash computation for request matching:
- `compute_content_hash`: exact match (all request params including sampling)
- `compute_prompt_model_hash`: sampling-fuzzy (excludes sampling params)
- `compute_prompt_hash`: model-fuzzy (messages only)

All hashes use SHA-256 with recursive key sorting and compact JSON serialization.
Content hash and prompt_model_hash are truncated to 12 hex characters (48 bits).
Prompt hash is truncated to 16 hex characters (64 bits) for backward compatibility.

Key functions: `compute_content_hash`, `compute_prompt_model_hash`,
    `compute_prompt_hash`, `compute_response_hash`, `compute_asset_hash`,
    `is_greedy`, `generate_slug`
"""

import hashlib
import json
import re
from typing import Any

from inference_gate.recording.models import ALWAYS_EXCLUDED_FIELDS, SAMPLING_PARAM_NAMES

# Truncation lengths (hex characters)
_CONTENT_HASH_LEN = 12
_PROMPT_MODEL_HASH_LEN = 12
_PROMPT_HASH_LEN = 16
_RESPONSE_HASH_LEN = 12
_ASSET_HASH_LEN = 12


def _sort_recursive(obj: Any) -> Any:
    """
    Recursively sort all dict keys in a JSON-like object.

    Lists are preserved in order (element order is semantically meaningful
    for messages and tool_calls). Dict keys are sorted alphabetically.
    """
    if isinstance(obj, dict):
        return {k: _sort_recursive(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_sort_recursive(item) for item in obj]
    return obj


def _hash_data(data: Any, truncate: int) -> str:
    """
    Serialize `data` to compact sorted JSON, SHA-256 hash, and truncate.

    Returns a lowercase hex string of length `truncate`.
    """
    sorted_data = _sort_recursive(data)
    encoded = json.dumps(sorted_data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:truncate]


def compute_content_hash(method: str, path: str, body: dict[str, Any] | None) -> str:
    """
    Compute the exact-match content hash for a request.

    Includes method, path, and all body fields except transport-level ones
    (`stream`, `stream_options`).  This is the cassette filename key.

    Returns a 12-character hex string.
    """
    filtered_body = _filter_body(body, exclude=ALWAYS_EXCLUDED_FIELDS)
    key_data = {"method": method, "path": path, "body": filtered_body}
    return _hash_data(key_data, _CONTENT_HASH_LEN)


def compute_prompt_model_hash(method: str, path: str, body: dict[str, Any] | None) -> str:
    """
    Compute the sampling-fuzzy hash for a request.

    Includes method, path, model, messages, and non-sampling params
    (max_tokens, tools, tool_choice, stop, logprobs settings).
    Excludes all sampling params (temperature, top_p, top_k, etc.) and
    transport fields.

    Returns a 12-character hex string.
    """
    filtered_body = _filter_body(body, exclude=ALWAYS_EXCLUDED_FIELDS | SAMPLING_PARAM_NAMES)
    key_data = {"method": method, "path": path, "body": filtered_body}
    return _hash_data(key_data, _PROMPT_MODEL_HASH_LEN)


def compute_prompt_hash(body: dict[str, Any] | None) -> str:
    """
    Compute the model-fuzzy hash from messages only.

    For Chat Completions API: hashes the ``messages`` field.
    For Responses API: hashes the ``input`` field.
    For raw Completions API (``/v1/completions``, ``/completion``): hashes
    the ``prompt`` field as a raw string.  Raw prompts are hashed directly
    (not wrapped in a messages structure) so they never cross-match with
    messages-based requests at tier 3.

    Returns a 16-character hex string.
    """
    if not body:
        return _hash_data(None, _PROMPT_HASH_LEN)

    # Chat Completions API uses "messages", Responses API uses "input"
    messages = body.get("messages") or body.get("input")
    if messages is not None:
        return _hash_data(messages, _PROMPT_HASH_LEN)

    # Raw Completions API uses "prompt" (pre-formatted string from chat template)
    prompt = body.get("prompt")
    if prompt is not None:
        return _hash_data(prompt, _PROMPT_HASH_LEN)

    return _hash_data(None, _PROMPT_HASH_LEN)


def compute_response_hash(response_body: dict[str, Any]) -> str:
    """
    Compute a content hash for a response body, for deduplication.

    Strips server-assigned `id` and `usage` fields before hashing so that
    identical completions with different request IDs or token counts
    collapse to the same hash.

    Returns a 12-character hex string.
    """
    # Shallow copy to avoid mutating the caller's dict
    normalized = {k: v for k, v in response_body.items() if k not in ("id", "usage")}
    return _hash_data(normalized, _RESPONSE_HASH_LEN)


def compute_asset_hash(raw_bytes: bytes) -> str:
    """
    Compute a content hash for raw binary asset data.

    No normalization — hashes the exact bytes.

    Returns a 12-character hex string.
    """
    return hashlib.sha256(raw_bytes).hexdigest()[:_ASSET_HASH_LEN]


def is_greedy(body: dict[str, Any] | None) -> bool:
    """
    Determine if a request uses greedy (deterministic) sampling.

    Returns True only when `temperature` is explicitly set to 0.0.
    Unspecified temperature returns False (conservative — treated as non-greedy).
    """
    if not body:
        return False
    temp = body.get("temperature")
    if temp is not None and float(temp) == 0.0:
        return True
    return False


def generate_slug(body: dict[str, Any] | None, max_length: int = 40) -> str:
    """
    Generate a human-readable filename slug from the first user message.

    Takes the first 40 characters of the first user message content,
    lowercased, non-alphanumeric characters replaced with hyphens,
    consecutive hyphens collapsed, leading/trailing hyphens stripped.

    Falls back to the raw ``prompt`` field for Completions API requests.

    Returns an empty string if no user message is found.
    """
    first_user_text = _extract_prompt_text(body)
    if not first_user_text:
        return ""

    slug = first_user_text[:max_length].lower()
    slug = re.sub(r"[^a-z0-9]", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    return slug


def extract_first_user_message(body: dict[str, Any] | None, max_length: int = 120) -> str:
    """
    Extract the first user message text for the index TSV.

    Returns up to ``max_length`` characters with newlines and tabs replaced
    by spaces.  Falls back to the raw ``prompt`` field for Completions API
    requests.
    """
    first_user_text = _extract_prompt_text(body)
    # Sanitize for TSV: replace newlines and tabs with spaces
    text = first_user_text[:max_length]
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return text


def _extract_prompt_text(body: dict[str, Any] | None) -> str:
    """
    Extract the first user-facing text from a request body.

    Checks ``messages`` (Chat Completions), ``input`` (Responses API),
    and ``prompt`` (raw Completions) in that order.

    Returns the raw text string or empty string if nothing is found.
    """
    if not body:
        return ""

    # Chat Completions API / Responses API
    messages = body.get("messages") or body.get("input")
    if messages and isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            return part.get("text", "")
                return ""
            if isinstance(msg, str):
                # Responses API simple string input
                return msg

    # Raw Completions API (pre-formatted prompt string)
    prompt = body.get("prompt")
    if prompt and isinstance(prompt, str):
        return prompt

    return ""


def _filter_body(body: dict[str, Any] | None, exclude: frozenset[str]) -> dict[str, Any] | None:
    """
    Return a copy of `body` with `exclude` keys removed.

    Returns None if body is None.
    """
    if body is None:
        return None
    return {k: v for k, v in body.items() if k not in exclude}
