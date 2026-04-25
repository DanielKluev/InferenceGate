"""
Data models for the v2 cassette tape storage format.

Defines Pydantic models for tape frontmatter metadata, MIME body sections,
index rows, and response references used across the tape parser, writer,
index, and storage modules.

Key classes: `TapeMetadata`, `TapeSection`, `IndexRow`, `ReplyInfo`
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SamplingParams(BaseModel):
    """
    Sampling parameters extracted from a request body.

    Only includes parameters that were actually present in the original request.
    `is_greedy` is always computed and present.
    """

    is_greedy: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None


# All sampling-related fields that should be excluded from prompt_model_hash.
SAMPLING_PARAM_NAMES: frozenset[str] = frozenset({
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
    "seed",
})

# Fields always excluded from ALL hash computations (non-deterministic or transport-level).
ALWAYS_EXCLUDED_FIELDS: frozenset[str] = frozenset({
    "stream",
    "stream_options",
})


def extract_sampling_params(body: dict[str, Any]) -> SamplingParams:
    """
    Extract sampling parameters from a request body dict.

    Returns a `SamplingParams` instance with only the fields that were
    present in the body.  `is_greedy` is computed based on the temperature
    field: True only when temperature is explicitly 0.0.
    """
    temp = body.get("temperature")
    is_greedy = temp is not None and float(temp) == 0.0

    kwargs: dict[str, Any] = {"is_greedy": is_greedy}
    for field_name in SAMPLING_PARAM_NAMES:
        value = body.get(field_name)
        if value is not None:
            kwargs[field_name] = value
    return SamplingParams(**kwargs)


class ReplyInfo(BaseModel):
    """
    Metadata for a single reply section within a tape file.

    References the response files in the `responses/` directory by content hash
    and carries summary statistics extracted at record time.
    """

    reply_number: int
    response_hash: str
    has_stream: bool = False
    stop_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    # HTTP status code from the upstream response. Defaults to 200 for backward compatibility
    # with v1 tapes (see migration in CacheStorage).  Non-200 statuses indicate recorded errors.
    status_code: int = 200
    # The reassembled plain-text body of the reply (for human readability in the tape)
    text: str = ""
    # Reasoning / chain-of-thought content, stored separately from `text` so that it can be
    # rendered in its own MIME sub-section for readability and inspected programmatically.
    reasoning: str = ""
    # Tool calls within this reply: list of (tool_name, tool_call_id, arguments_json)
    tool_calls: list[tuple[str, str, str]] = Field(default_factory=list)


class SectionKind(str, Enum):
    """
    Kinds of MIME sections in a tape file body.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT_PREFILL = "assistant_prefill"
    USER_ATTACHMENT = "user_attachment"
    TOOLS = "tools"
    REPLY = "reply"
    REPLY_REASONING = "reply_reasoning"
    REPLY_TOOL_CALL = "reply_tool_call"
    UNKNOWN = "unknown"


class TapeSection(BaseModel):
    """
    A single MIME section from a tape file body.

    Represents one boundary-delimited section: a conversation message,
    a tool definition block, a reply, or a reply tool call.
    """

    kind: SectionKind
    # Raw header line after the boundary marker (e.g. "user", "reply 1", "reply 1 tool_call get_weather toolu_01")
    header: str = ""
    # Key-value metadata lines (e.g. Response: hash, Stop-Reason: end_turn) for reply sections
    metadata: dict[str, str] = Field(default_factory=dict)
    # Body text (everything after the first blank line following metadata)
    body: str = ""
    # For reply sections: the reply number
    reply_number: int | None = None
    # For tool_call sections: tool name and tool call ID
    tool_name: str | None = None
    tool_call_id: str | None = None


class TapeMetadata(BaseModel):
    """
    YAML frontmatter of a tape file.

    Contains all metadata needed for indexing, lookup, and replay
    without parsing the MIME body sections.
    """

    tape_version: int = 2
    content_hash: str = ""
    prompt_model_hash: str = ""
    prompt_hash: str = ""
    model: str | None = None
    endpoint: str = ""
    sampling: SamplingParams = Field(default_factory=SamplingParams)
    max_tokens: int | None = None
    stop_sequences: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    tool_choice: str | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    recorded: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    replies: int = 0
    max_replies: int = 1
    # HTTP status of the primary (first) reply.  Per-reply status is stored in each
    # reply section's `Status:` metadata header and may differ from this value when
    # a multi-reply tape contains mixed outcomes.
    status_code: int = 200
    boundary: str = ""


class IndexRow(BaseModel):
    """
    One row of the `index.tsv` file, representing a single cassette.

    All three hash tiers are stored for multi-level lookup.
    """

    content_hash: str
    prompt_model_hash: str
    prompt_hash: str
    model: str = ""
    endpoint: str = ""
    is_greedy: bool = False
    temperature: str = ""  # stored as string to distinguish "0.0" from empty/absent
    tokens_in: str = ""
    tokens_out: str = ""
    replies: int = 0
    max_replies: int = 1
    has_logprobs: bool = False
    has_tool_use: bool = False
    # HTTP status of the primary reply.  Defaults to 200 for v1 cassettes and for
    # tapes that predate the status_code field.
    status_code: int = 200
    # True when any reply in the cassette has non-empty reasoning/CoT content.
    has_reasoning: bool = False
    slug: str = ""
    recorded: str = ""
    first_user_message: str = ""

    @classmethod
    def from_tape_metadata(cls, meta: TapeMetadata, slug: str, first_user_message: str, tokens_in: str = "", tokens_out: str = "",
                           has_reasoning: bool = False) -> IndexRow:
        """
        Create an IndexRow from a TapeMetadata and supplementary data.

        `has_reasoning` should be set by the caller based on whether any reply
        in the cassette carries reasoning content (determined after parsing the
        body sections).  Defaults to False.
        """
        temp_str = ""
        if meta.sampling.temperature is not None:
            temp_str = str(meta.sampling.temperature)

        return cls(
            content_hash=meta.content_hash,
            prompt_model_hash=meta.prompt_model_hash,
            prompt_hash=meta.prompt_hash,
            model=meta.model or "",
            endpoint=meta.endpoint or "",
            is_greedy=meta.sampling.is_greedy,
            temperature=temp_str,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            replies=meta.replies,
            max_replies=meta.max_replies,
            has_logprobs=meta.logprobs,
            has_tool_use=bool(meta.tools),
            status_code=meta.status_code,
            has_reasoning=has_reasoning,
            slug=slug,
            recorded=meta.recorded.isoformat() if meta.recorded else "",
            first_user_message=first_user_message,
        )
