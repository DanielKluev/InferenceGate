"""
`headers` defines the strict registry and parser for the `X-InferenceGate-*`
HTTP header contract.

The header contract is the single source of truth for per-request control of
InferenceGate behaviour. Three suffix categories are recognised:

- ``Require-*`` — match-time filters; never forwarded upstream; never part of
  any cache hash; used to constrain cassette lookup (engine, fuzzy mode, etc.).
- ``Metadata-*`` — recorded into tape frontmatter when a new cassette is being
  written; ignored on REPLAY; never part of any cache hash.
- ``Control-*`` — runtime behaviour overrides for a single request (mode,
  reply-strategy).

Unknown ``X-InferenceGate-*`` headers cause a 400 response. All headers in the
``X-InferenceGate-*`` namespace are stripped from the request dict before any
upstream forward or cache-key computation.

Key classes/functions: `HeaderCategory`, `ParsedHeaders`, `parse_headers`,
`strip_inferencegate_headers`, `HEADER_REGISTRY`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

log = logging.getLogger("inference_gate.headers")

# Common namespace prefix for every header in the contract.
INFERENCEGATE_HEADER_PREFIX = "x-inferencegate-"

# Legacy alias kept for backward compatibility (one minor version).
LEGACY_REPLY_STRATEGY_HEADER = "x-gate-reply-strategy"


class HeaderCategory(str, Enum):
    """
    Suffix categories of the `X-InferenceGate-*` header contract.
    """

    REQUIRE = "Require"
    METADATA = "Metadata"
    CONTROL = "Control"


@dataclass(frozen=True)
class HeaderSpec:
    """
    Specification of a single registered header.

    `canonical` is the canonical case-correct header name as documented.
    `category` is the suffix category.
    `valid_values` is an optional set of allowed lowercase string values; when
    None the header accepts any non-empty string.
    """

    canonical: str
    category: HeaderCategory
    valid_values: frozenset[str] | None = None
    description: str = ""


# Strict allowlist. Adding a header MUST go through this registry — both the
# parser and Gate's docs read from here.
HEADER_REGISTRY: dict[str, HeaderSpec] = {
    # Require-*
    "x-inferencegate-require-engine":
        HeaderSpec("X-InferenceGate-Require-Engine", HeaderCategory.REQUIRE, None,
                   "Match cassettes whose metadata.engine equals this value."),
    "x-inferencegate-require-engine-version":
        HeaderSpec("X-InferenceGate-Require-Engine-Version", HeaderCategory.REQUIRE, None,
                   "Match cassettes whose metadata.engine_version equals this value."),
    "x-inferencegate-require-fuzzy-model":
        HeaderSpec("X-InferenceGate-Require-Fuzzy-Model", HeaderCategory.REQUIRE, frozenset({"on", "off"}),
                   "Per-request override of session fuzzy_model setting."),
    "x-inferencegate-require-fuzzy-sampling":
        HeaderSpec("X-InferenceGate-Require-Fuzzy-Sampling", HeaderCategory.REQUIRE, frozenset({"off", "soft", "aggressive"}),
                   "Per-request override of session fuzzy_sampling setting."),
    "x-inferencegate-require-exact":
        HeaderSpec("X-InferenceGate-Require-Exact", HeaderCategory.REQUIRE, frozenset({"true", "false"}),
                   "Shortcut: when true, forces fuzzy_model=off and fuzzy_sampling=off."),
    # Metadata-*
    "x-inferencegate-metadata-engine":
        HeaderSpec("X-InferenceGate-Metadata-Engine", HeaderCategory.METADATA, None, "Stored as metadata.engine in tape frontmatter."),
    "x-inferencegate-metadata-engine-version":
        HeaderSpec("X-InferenceGate-Metadata-Engine-Version", HeaderCategory.METADATA, None,
                   "Stored as metadata.engine_version in tape frontmatter."),
    "x-inferencegate-metadata-test-nodeid":
        HeaderSpec("X-InferenceGate-Metadata-Test-NodeID", HeaderCategory.METADATA, None, "Pytest node id of the recording test."),
    "x-inferencegate-metadata-worker-id":
        HeaderSpec("X-InferenceGate-Metadata-Worker-ID", HeaderCategory.METADATA, None, "pytest-xdist worker id (or 'main')."),
    "x-inferencegate-metadata-recorded-by":
        HeaderSpec("X-InferenceGate-Metadata-Recorded-By", HeaderCategory.METADATA, None, "Free-form provenance string."),
    # Control-*
    "x-inferencegate-control-mode":
        HeaderSpec("X-InferenceGate-Control-Mode", HeaderCategory.CONTROL, frozenset({"replay", "record", "passthrough"}),
                   "Per-request mode override."),
    "x-inferencegate-control-reply-strategy":
        HeaderSpec("X-InferenceGate-Control-Reply-Strategy", HeaderCategory.CONTROL, frozenset({"round-robin", "random", "first"}),
                   "Per-request multi-reply selection strategy."),
}


# Convenience: lowercase → metadata field-name mapping for Metadata-* headers.
METADATA_HEADER_TO_FIELD: dict[str, str] = {
    "x-inferencegate-metadata-engine": "engine",
    "x-inferencegate-metadata-engine-version": "engine_version",
    "x-inferencegate-metadata-test-nodeid": "test_node_id",
    "x-inferencegate-metadata-worker-id": "worker_id",
    "x-inferencegate-metadata-recorded-by": "recorded_by",
}


@dataclass
class ParsedHeaders:
    """
    Parsed `X-InferenceGate-*` headers grouped by category.

    `require` carries lookup constraints; `metadata` carries fields to record
    into tape frontmatter; `control` carries per-request overrides.
    """

    require: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    control: dict[str, str] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """
        Return True when no recognised headers were parsed.
        """
        return not (self.require or self.metadata or self.control)


class HeaderValidationError(ValueError):
    """
    Raised when an unknown `X-InferenceGate-*` header or an invalid value is
    encountered. Carries an HTTP-friendly diagnostic message.
    """

    def __init__(self, message: str, *, offending_header: str | None = None, offending_value: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.offending_header = offending_header
        self.offending_value = offending_value


def parse_headers(headers: dict[str, str]) -> ParsedHeaders:
    """
    Extract and validate `X-InferenceGate-*` headers from a request header dict.

    Header lookup is case-insensitive. Unknown headers in the namespace raise
    `HeaderValidationError`. Values are validated against `HeaderSpec.valid_values`
    when present; invalid values also raise.

    The legacy `X-Gate-Reply-Strategy` header is accepted as an alias for
    `X-InferenceGate-Control-Reply-Strategy` and warned about in logs.

    Returns a `ParsedHeaders` object. Note: this function does NOT mutate or
    strip the input dict. Use `strip_inferencegate_headers` for that.
    """
    parsed = ParsedHeaders()

    for key, value in headers.items():
        key_lower = key.lower()

        # Legacy alias handling: X-Gate-Reply-Strategy → control reply-strategy.
        if key_lower == LEGACY_REPLY_STRATEGY_HEADER:
            spec = HEADER_REGISTRY["x-inferencegate-control-reply-strategy"]
            _validate_value(spec, value)
            parsed.control["reply_strategy"] = value.strip()
            log.debug("Accepted legacy header %s (alias for %s)", key, spec.canonical)
            continue

        if not key_lower.startswith(INFERENCEGATE_HEADER_PREFIX):
            continue

        spec = HEADER_REGISTRY.get(key_lower)
        if spec is None:
            allowed = ", ".join(sorted(s.canonical for s in HEADER_REGISTRY.values()))
            raise HeaderValidationError(
                f"Unknown InferenceGate header: {key!r}. Allowed: {allowed}",
                offending_header=key,
            )

        _validate_value(spec, value)
        normalised = value.strip()

        if spec.category == HeaderCategory.REQUIRE:
            # Strip the X-InferenceGate-Require- prefix to get the field name.
            field_name = key_lower[len("x-inferencegate-require-"):].replace("-", "_")
            parsed.require[field_name] = normalised
        elif spec.category == HeaderCategory.METADATA:
            field_name = METADATA_HEADER_TO_FIELD[key_lower]
            parsed.metadata[field_name] = normalised
        elif spec.category == HeaderCategory.CONTROL:
            field_name = key_lower[len("x-inferencegate-control-"):].replace("-", "_")
            parsed.control[field_name] = normalised

    # Apply Require-Exact shortcut: it forces both fuzzy fields off.
    exact_value = parsed.require.pop("exact", None)
    if exact_value is not None and exact_value.lower() == "true":
        parsed.require.setdefault("fuzzy_model", "off")
        parsed.require.setdefault("fuzzy_sampling", "off")

    return parsed


def _validate_value(spec: HeaderSpec, value: str) -> None:
    """
    Validate a header value against its `HeaderSpec`.

    Raises `HeaderValidationError` when the value is empty or fails an
    enumerated-value check.
    """
    stripped = value.strip()
    if not stripped:
        raise HeaderValidationError(
            f"Empty value for header {spec.canonical}",
            offending_header=spec.canonical,
            offending_value=value,
        )
    if spec.valid_values is not None and stripped.lower() not in spec.valid_values:
        allowed = ", ".join(sorted(spec.valid_values))
        raise HeaderValidationError(
            f"Invalid value {stripped!r} for header {spec.canonical}. Allowed: {allowed}",
            offending_header=spec.canonical,
            offending_value=value,
        )


def strip_inferencegate_headers(headers: dict[str, str]) -> dict[str, str]:
    """
    Return a copy of `headers` with every `X-InferenceGate-*` header (and the
    legacy `X-Gate-Reply-Strategy` alias) removed.

    Used at the boundary between Gate and any downstream consumer (cache-hash
    computation, upstream forward) so that contract headers never leak through.
    """
    cleaned: dict[str, str] = {}
    for key, value in headers.items():
        key_lower = key.lower()
        if key_lower.startswith(INFERENCEGATE_HEADER_PREFIX):
            continue
        if key_lower == LEGACY_REPLY_STRATEGY_HEADER:
            continue
        cleaned[key] = value
    return cleaned


def required_engine_matches(parsed: ParsedHeaders, candidate_metadata: dict[str, Any] | None) -> bool:
    """
    Check whether a candidate cassette satisfies any `Require-Engine`/`Require-Engine-Version`
    constraints in `parsed`.

    `candidate_metadata` is the `TapeMetadata.metadata` dict for the candidate (or None when
    the cassette has no recorded metadata block — in which case any Require-* engine constraint
    is treated as a mismatch).

    Returns True when the candidate matches all engine-related Require-* constraints.
    """
    required_engine = parsed.require.get("engine")
    required_version = parsed.require.get("engine_version")
    if required_engine is None and required_version is None:
        return True

    candidate_metadata = candidate_metadata or {}
    if required_engine is not None and candidate_metadata.get("engine") != required_engine:
        return False
    if required_version is not None and candidate_metadata.get("engine_version") != required_version:
        return False
    return True
