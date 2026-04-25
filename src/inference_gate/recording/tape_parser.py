"""
Parser for the v2 `.tape` cassette file format.

Reads a tape file containing YAML frontmatter and MIME-style body sections,
producing a `TapeMetadata` object and a list of `TapeSection` objects.

The parser is lenient: unknown section types are preserved as `SectionKind.UNKNOWN`
and do not cause errors.  This allows forward-compatible format evolution.

Key functions: `parse_tape`, `parse_tape_frontmatter`
"""

import logging
from typing import Any

import yaml

from inference_gate.recording.models import SamplingParams, SectionKind, TapeMetadata, TapeSection

log = logging.getLogger("tape_parser")


def parse_tape(content: str) -> tuple[TapeMetadata, list[TapeSection]]:
    """
    Parse a complete tape file into metadata and body sections.

    The tape format is:
    ```
    ---
    <YAML frontmatter>
    ---
    --{boundary} <role> [qualifier...]
    <metadata key-value lines>

    <body text>
    --{boundary} ...
    ...
    --{boundary}--
    ```

    Returns a tuple of (`TapeMetadata`, list of `TapeSection`).
    """
    metadata, body_text = _split_frontmatter(content)
    sections = _parse_body_sections(body_text, metadata.boundary)
    return metadata, sections


def parse_tape_frontmatter(content: str) -> TapeMetadata:
    """
    Parse only the YAML frontmatter of a tape file (fast path for indexing).

    Stops parsing at the closing `---` delimiter and does not process
    the MIME body sections.

    Returns a `TapeMetadata` object.
    """
    metadata, _ = _split_frontmatter(content)
    return metadata


def _split_frontmatter(content: str) -> tuple[TapeMetadata, str]:
    """
    Split tape content into YAML frontmatter and remaining body text.

    Expects the content to start with `---\\n` and have a second `---\\n`
    delimiter marking the end of frontmatter.

    Returns a tuple of (`TapeMetadata`, body_text).
    """
    # Strip leading BOM or whitespace
    content = content.lstrip("\ufeff")

    if not content.startswith("---"):
        raise ValueError("Tape file must start with '---' frontmatter delimiter")

    # Find the closing --- delimiter (skip the opening one)
    first_newline = content.index("\n")
    rest = content[first_newline + 1:]
    closing_idx = rest.find("\n---")
    if closing_idx == -1:
        raise ValueError("Tape file missing closing '---' frontmatter delimiter")

    yaml_text = rest[:closing_idx]
    # Body starts after the closing ---\n
    body_start = first_newline + 1 + closing_idx + 4  # len("\n---") = 4
    # Skip the newline after ---
    if body_start < len(content) and content[body_start] == "\n":
        body_start += 1
    body_text = content[body_start:]

    metadata = _parse_yaml_frontmatter(yaml_text)
    return metadata, body_text


def _parse_yaml_frontmatter(yaml_text: str) -> TapeMetadata:
    """
    Parse YAML frontmatter text into a `TapeMetadata` object.

    Handles the `sampling` nested dict and converts it to `SamplingParams`.
    """
    data: dict[str, Any] = yaml.safe_load(yaml_text) or {}

    # Extract and convert sampling params
    sampling_data = data.pop("sampling", None)
    if sampling_data and isinstance(sampling_data, dict):
        sampling = SamplingParams(**sampling_data)
    else:
        sampling = SamplingParams()

    # Build TapeMetadata, letting Pydantic validate fields
    data["sampling"] = sampling
    return TapeMetadata.model_validate(data)


def _parse_body_sections(body_text: str, boundary: str) -> list[TapeSection]:
    """
    Parse the MIME-style body sections of a tape file.

    Splits on `--{boundary}` markers and parses each section's header,
    metadata key-value lines, and body text.

    The terminal boundary `--{boundary}--` marks the end of sections.

    Returns a list of `TapeSection` objects.
    """
    if not boundary or not body_text.strip():
        return []

    marker = f"--{boundary}"
    terminal = f"--{boundary}--"

    sections: list[TapeSection] = []
    # Split on boundary markers that appear at start of line
    parts = body_text.split(f"\n{marker}")
    # The first part might start with the marker directly (no leading newline)
    if parts and parts[0].startswith(marker):
        parts[0] = parts[0][len(marker):]
    else:
        # Text before the first boundary — discard
        parts = parts[1:]

    for part in parts:
        # Check for terminal boundary
        stripped = part.strip()
        if stripped == "--" or stripped.startswith("--\n") or stripped == "":
            # Terminal boundary `--{boundary}--` — we split on `--{boundary}` so remainder is `--`
            break

        section = _parse_single_section(part)
        if section is not None:
            sections.append(section)

    return sections


def _parse_single_section(part: str) -> TapeSection | None:
    """
    Parse a single section from text following a boundary marker.

    Expected format:
    ```
     <header>
    Key: Value
    Key: Value

    <body text>
    ```

    The header is on the first line (starts with a space after the boundary marker).
    Metadata key-value lines follow until the first blank line.
    Everything after the first blank line is the body.
    """
    # The part starts with the header (may have leading space/newline after boundary)
    lines = part.split("\n")

    # First line is the header (role + qualifiers)
    header_line = ""
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped:
            header_line = stripped
            start_idx = i + 1
            break

    if not header_line:
        return None

    # Parse header to determine section kind
    kind, reply_number, tool_name, tool_call_id = _parse_header(header_line)

    # Parse metadata key-value pairs and body
    metadata: dict[str, str] = {}
    body_lines: list[str] = []
    in_body = False

    for line in lines[start_idx:]:
        if in_body:
            body_lines.append(line)
        elif line.strip() == "":
            # Blank line separates metadata from body
            in_body = True
        elif ":" in line and not in_body:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()
        else:
            # No colon and not blank — treat as start of body
            in_body = True
            body_lines.append(line)

    # Strip trailing empty lines from body
    body = "\n".join(body_lines).rstrip("\n")

    return TapeSection(
        kind=kind,
        header=header_line,
        metadata=metadata,
        body=body,
        reply_number=reply_number,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
    )


def _parse_header(header: str) -> tuple[SectionKind, int | None, str | None, str | None]:
    """
    Parse a section header string into kind, reply number, tool name, and tool call ID.

    Examples:
        "system"                              → (SYSTEM, None, None, None)
        "user"                                → (USER, None, None, None)
        "user attachment"                     → (USER_ATTACHMENT, None, None, None)
        "assistant prefill"                   → (ASSISTANT_PREFILL, None, None, None)
        "tools"                               → (TOOLS, None, None, None)
        "reply 1"                             → (REPLY, 1, None, None)
        "reply 1 tool_call get_weather id123" → (REPLY_TOOL_CALL, 1, "get_weather", "id123")

    Returns a tuple of (SectionKind, reply_number, tool_name, tool_call_id).
    """
    parts = header.split()
    if not parts:
        return SectionKind.UNKNOWN, None, None, None

    role = parts[0].lower()

    if role == "system":
        return SectionKind.SYSTEM, None, None, None
    elif role == "user":
        if len(parts) > 1 and parts[1].lower() == "attachment":
            return SectionKind.USER_ATTACHMENT, None, None, None
        return SectionKind.USER, None, None, None
    elif role == "assistant":
        if len(parts) > 1 and parts[1].lower() == "prefill":
            return SectionKind.ASSISTANT_PREFILL, None, None, None
        return SectionKind.UNKNOWN, None, None, None
    elif role == "tools":
        return SectionKind.TOOLS, None, None, None
    elif role == "reply":
        if len(parts) < 2:
            return SectionKind.REPLY, None, None, None
        try:
            reply_num = int(parts[1])
        except ValueError:
            return SectionKind.UNKNOWN, None, None, None

        if len(parts) >= 4 and parts[2].lower() == "tool_call":
            tool_name = parts[3] if len(parts) > 3 else None
            tool_call_id = parts[4] if len(parts) > 4 else None
            return SectionKind.REPLY_TOOL_CALL, reply_num, tool_name, tool_call_id

        # "reply N reasoning" — chain-of-thought / reasoning content for reply N.
        if len(parts) >= 3 and parts[2].lower() == "reasoning":
            return SectionKind.REPLY_REASONING, reply_num, None, None

        return SectionKind.REPLY, reply_num, None, None

    return SectionKind.UNKNOWN, None, None, None
