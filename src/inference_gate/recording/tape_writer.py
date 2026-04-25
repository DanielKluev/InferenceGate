"""
Writer for the v2 `.tape` cassette file format.

Produces tape files with YAML frontmatter and MIME-style body sections
from `TapeMetadata` and `TapeSection` objects.

Key functions: `write_tape`, `write_tape_to_file`, `generate_boundary`,
    `append_reply_to_tape`
"""

import logging
import secrets
from pathlib import Path
from typing import Any

import yaml

from inference_gate.recording.models import ReplyInfo, SectionKind, TapeMetadata, TapeSection

log = logging.getLogger("tape_writer")


def write_tape(metadata: TapeMetadata, sections: list[TapeSection]) -> str:
    """
    Serialize a tape to its string representation.

    Produces YAML frontmatter followed by MIME-style body sections
    delimited by the boundary from `metadata.boundary`.

    Returns the complete tape file content as a string.
    """
    parts: list[str] = []

    # Write frontmatter
    parts.append("---\n")
    parts.append(_serialize_frontmatter(metadata))
    parts.append("---\n")

    # Write body sections
    boundary = metadata.boundary
    for section in sections:
        parts.append(f"--{boundary} {section.header}\n")
        # Write metadata key-value pairs
        for key, value in section.metadata.items():
            parts.append(f"{key}: {value}\n")
        # Blank line separating metadata from body
        if section.metadata or section.body:
            parts.append("\n")
        # Write body
        if section.body:
            parts.append(section.body)
            parts.append("\n")

    # Terminal boundary
    parts.append(f"--{boundary}--\n")

    return "".join(parts)


def write_tape_to_file(filepath: Path, metadata: TapeMetadata, sections: list[TapeSection]) -> None:
    """
    Write a tape to a file.

    Creates parent directories if they do not exist.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    content = write_tape(metadata, sections)
    filepath.write_text(content, encoding="utf-8")
    log.debug("Wrote tape to %s", filepath)


def generate_boundary(content_to_check: str = "", max_attempts: int = 100) -> str:
    """
    Generate a 6-character hex boundary string that does not appear in `content_to_check`.

    Regenerates if the boundary string is found within the content.

    Returns a 6-character hex string.
    """
    for _ in range(max_attempts):
        boundary = secrets.token_hex(3)  # 6 hex chars
        if boundary not in content_to_check:
            return boundary
    # Fallback: use a longer boundary that's even less likely to collide
    return secrets.token_hex(8)


def build_message_sections(body: dict[str, Any] | None, boundary: str) -> list[TapeSection]:
    """
    Build tape body sections from a request body dict.

    Converts `messages` (Chat Completions) or `input` (Responses API)
    into a list of `TapeSection` objects representing the conversation
    and any tool definitions.

    Returns a list of `TapeSection` objects.
    """
    if not body:
        return []

    sections: list[TapeSection] = []

    # Handle tools (must come early since they define capabilities)
    tools = body.get("tools")
    if tools:
        import json
        sections.append(
            TapeSection(
                kind=SectionKind.TOOLS,
                header="tools",
                body=json.dumps(tools, separators=(",", ":"), ensure_ascii=False),
            ))

    # Handle messages (Chat Completions API)
    messages = body.get("messages")
    if messages and isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                text = _extract_text_content(content)
                sections.append(TapeSection(kind=SectionKind.SYSTEM, header="system", body=text))
            elif role == "user":
                # Check for multimodal content (attachments)
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                sections.append(TapeSection(kind=SectionKind.USER, header="user", body=part.get("text", "")))
                            elif part.get("type") == "image_url":
                                # Image attachment — store reference
                                url_data = part.get("image_url", {})
                                url = url_data.get("url", "") if isinstance(url_data, dict) else str(url_data)
                                sections.append(
                                    TapeSection(
                                        kind=SectionKind.USER_ATTACHMENT,
                                        header="user attachment",
                                        metadata={
                                            "Type": "image",
                                            "URL": url[:200]
                                        },
                                    ))
                else:
                    text = _extract_text_content(content)
                    sections.append(TapeSection(kind=SectionKind.USER, header="user", body=text))
            elif role == "assistant":
                text = _extract_text_content(content)
                if text:
                    sections.append(TapeSection(kind=SectionKind.ASSISTANT_PREFILL, header="assistant prefill", body=text))

    # Handle Responses API input
    input_data = body.get("input")
    if input_data and isinstance(input_data, list) and not messages:
        for item in input_data:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                text = _extract_text_content(content)
                if role == "user":
                    sections.append(TapeSection(kind=SectionKind.USER, header="user", body=text))
                elif role == "system":
                    sections.append(TapeSection(kind=SectionKind.SYSTEM, header="system", body=text))
            elif isinstance(item, str):
                sections.append(TapeSection(kind=SectionKind.USER, header="user", body=item))

    # Handle raw Completions API (pre-formatted prompt string from chat template).
    # The entire pre-formatted string is stored as a single user section.
    if not messages and not input_data:
        prompt = body.get("prompt")
        if prompt and isinstance(prompt, str):
            sections.append(TapeSection(kind=SectionKind.USER, header="user", body=prompt))

    return sections


def build_reply_section(reply_info: ReplyInfo) -> list[TapeSection]:
    """
    Build tape sections for a single reply (reply header + optional tool_call sections).

    Always emits the primary ``reply N`` section with ``Status:`` metadata so that
    HTTP status code (including non-200 errors) is faithfully recorded in v2 tapes.
    When ``reply_info.reasoning`` is non-empty, emits a sibling ``reply N reasoning``
    sub-section immediately after the primary section.  Tool calls (if any) follow.

    Returns a list of `TapeSection` objects (1 reply section + 0..1 reasoning section + 0..N tool_call sections).
    """
    num = reply_info.reply_number
    metadata: dict[str, str] = {}

    metadata["Response"] = f"{reply_info.response_hash}.json"
    if reply_info.has_stream:
        metadata["Stream"] = f"{reply_info.response_hash}.chunks.ndjson"
    # Always emit Status so v2 tapes are unambiguous (including explicit 200).
    metadata["Status"] = str(reply_info.status_code)
    if reply_info.stop_reason:
        metadata["Stop-Reason"] = reply_info.stop_reason
    if reply_info.input_tokens is not None:
        metadata["Input-Tokens"] = str(reply_info.input_tokens)
    if reply_info.output_tokens is not None:
        metadata["Output-Tokens"] = str(reply_info.output_tokens)
    if reply_info.latency_ms is not None:
        metadata["Latency-Ms"] = str(reply_info.latency_ms)

    sections: list[TapeSection] = []

    # Main reply section
    sections.append(TapeSection(
        kind=SectionKind.REPLY,
        header=f"reply {num}",
        metadata=metadata,
        body=reply_info.text,
        reply_number=num,
    ))

    # Reasoning sub-section (only when non-empty).  Placed immediately after the
    # primary reply so that human readers see the chain-of-thought alongside the answer.
    if reply_info.reasoning:
        sections.append(
            TapeSection(
                kind=SectionKind.REPLY_REASONING,
                header=f"reply {num} reasoning",
                body=reply_info.reasoning,
                reply_number=num,
            ))

    # Tool call sections
    for tool_name, tool_call_id, arguments_json in reply_info.tool_calls:
        sections.append(
            TapeSection(
                kind=SectionKind.REPLY_TOOL_CALL,
                header=f"reply {num} tool_call {tool_name} {tool_call_id}",
                body=arguments_json,
                reply_number=num,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            ))

    return sections


def append_reply_to_tape(filepath: Path, reply_info: ReplyInfo, new_reply_count: int) -> None:
    """
    Append a new reply to an existing tape file.

    Reads the tape, inserts the new reply section(s) before the terminal
    boundary, updates the `replies` count in frontmatter, and rewrites
    the file.
    """
    from inference_gate.recording.tape_parser import parse_tape

    content = filepath.read_text(encoding="utf-8")
    metadata, existing_sections = parse_tape(content)

    # Build new reply sections
    new_sections = build_reply_section(reply_info)

    # Combine: existing sections + new reply sections
    all_sections = existing_sections + new_sections

    # Update frontmatter
    metadata.replies = new_reply_count

    # Rewrite the tape file
    write_tape_to_file(filepath, metadata, all_sections)
    log.debug("Appended reply %d to %s (total replies: %d)", reply_info.reply_number, filepath.name, new_reply_count)


def _serialize_frontmatter(metadata: TapeMetadata) -> str:
    """
    Serialize `TapeMetadata` to YAML frontmatter text.

    Handles the `sampling` nested dict specially to produce clean YAML.
    """
    # Convert to dict, excluding None values and defaults where possible
    data = metadata.model_dump(mode="json", exclude_none=False)

    # Convert sampling to a clean dict (exclude None values)
    sampling_data = data.pop("sampling", {})
    clean_sampling: dict[str, Any] = {}
    # is_greedy is always present
    clean_sampling["is_greedy"] = sampling_data.get("is_greedy", False)
    for key, value in sampling_data.items():
        if key != "is_greedy" and value is not None:
            clean_sampling[key] = value
    data["sampling"] = clean_sampling

    # Remove fields that are None
    data = {k: v for k, v in data.items() if v is not None}

    return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _extract_text_content(content: Any) -> str:
    """
    Extract plain text from a message content field.

    Handles string content directly, or extracts text from a list of
    content parts (multimodal messages).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts)
    return str(content) if content else ""
