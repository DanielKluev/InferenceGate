"""
Output formatting helpers for the InferenceGate CLI.

Provides functions for rendering cassette data as human-readable tables
or JSON suitable for AI agentic consumption. Used by the `cassette`
command group in `cli.py`.

Key functions: `format_index_rows_table`, `format_index_rows_json`,
`format_tape_detail`, `format_tape_detail_json`, `format_stats`, `format_stats_json`
"""

import json
from typing import Any

from inference_gate.recording.models import IndexRow, SectionKind, TapeMetadata, TapeSection


def index_row_to_dict(row: IndexRow) -> dict[str, Any]:
    """
    Convert an IndexRow to a JSON-serializable dict for CLI output.
    """
    return {
        "id": row.content_hash,
        "model": row.model,
        "is_greedy": row.is_greedy,
        "temperature": row.temperature,
        "replies": row.replies,
        "max_replies": row.max_replies,
        "has_logprobs": row.has_logprobs,
        "has_tool_use": row.has_tool_use,
        "slug": row.slug,
        "recorded": row.recorded,
        "first_user_message": row.first_user_message,
        "tokens_in": row.tokens_in,
        "tokens_out": row.tokens_out,
    }


def format_index_rows_json(rows: list[IndexRow]) -> str:
    """
    Format a list of IndexRows as a JSON string.
    """
    return json.dumps([index_row_to_dict(r) for r in rows], indent=2, ensure_ascii=False)


def format_index_rows_table(rows: list[IndexRow]) -> str:
    """
    Format a list of IndexRows as a human-readable table.

    Columns: ID, Model, Temp, Replies, Tokens (in/out), Recorded, First Message.
    """
    if not rows:
        return "No cassettes found."

    lines = []
    # Header
    lines.append(f"{'ID':<14} {'Model':<28} {'Temp':>5} {'Repl':>4} {'Tok In':>7} {'Tok Out':>7} {'Recorded':<22} {'First Message'}")
    lines.append("-" * 120)

    for row in rows:
        temp_str = row.temperature if row.temperature else "-"
        tok_in = row.tokens_in if row.tokens_in else "-"
        tok_out = row.tokens_out if row.tokens_out else "-"
        # Truncate first_user_message for table display
        msg = row.first_user_message
        if len(msg) > 50:
            msg = msg[:47] + "..."
        # Truncate recorded to date + time (no microseconds)
        recorded = row.recorded[:19] if len(row.recorded) > 19 else row.recorded
        model = row.model if len(row.model) <= 28 else row.model[:25] + "..."

        lines.append(f"{row.content_hash:<14} {model:<28} {temp_str:>5} {row.replies:>4} {tok_in:>7} {tok_out:>7} {recorded:<22} {msg}")

    lines.append(f"\n{len(rows)} cassette(s)")
    return "\n".join(lines)


def section_to_dict(section: TapeSection) -> dict[str, Any]:
    """
    Convert a TapeSection to a JSON-serializable dict.
    """
    d: dict[str, Any] = {
        "kind": section.kind.value,
        "header": section.header,
        "body": section.body,
    }
    if section.metadata:
        d["metadata"] = section.metadata
    if section.reply_number is not None:
        d["reply_number"] = section.reply_number
    if section.tool_name is not None:
        d["tool_name"] = section.tool_name
    if section.tool_call_id is not None:
        d["tool_call_id"] = section.tool_call_id
    return d


def format_tape_detail_json(content_hash: str, metadata: TapeMetadata, sections: list[TapeSection],
                            index_row: IndexRow | None = None) -> str:
    """
    Format a full tape detail as JSON.
    """
    result: dict[str, Any] = {
        "id": content_hash,
        "model": metadata.model,
        "endpoint": metadata.endpoint,
        "sampling": metadata.sampling.model_dump(),
        "content_hash": metadata.content_hash,
        "prompt_model_hash": metadata.prompt_model_hash,
        "prompt_hash": metadata.prompt_hash,
        "recorded": metadata.recorded.isoformat() if metadata.recorded else None,
        "replies": metadata.replies,
        "max_replies": metadata.max_replies,
        "max_tokens": metadata.max_tokens,
        "stop_sequences": metadata.stop_sequences,
        "tools": metadata.tools,
        "logprobs": metadata.logprobs,
        "sections": [section_to_dict(s) for s in sections],
    }
    if index_row is not None:
        result["index"] = index_row_to_dict(index_row)
    return json.dumps(result, indent=2, ensure_ascii=False, default=str)


def format_tape_detail_human(content_hash: str, metadata: TapeMetadata, sections: list[TapeSection],
                             index_row: IndexRow | None = None) -> str:
    """
    Format a tape detail for human-readable display.

    Shows metadata header, prompt messages, and reply summaries.
    """
    lines = []

    # Metadata header
    lines.append(f"Cassette: {content_hash}")
    lines.append(f"Model: {metadata.model or '(not set)'}")
    lines.append(f"Endpoint: {metadata.endpoint}")
    lines.append(f"Recorded: {metadata.recorded.isoformat() if metadata.recorded else '(unknown)'}")

    temp = metadata.sampling.temperature
    lines.append(f"Temperature: {temp if temp is not None else '(not set)'}")
    lines.append(f"Greedy: {'Yes' if metadata.sampling.is_greedy else 'No'}")
    lines.append(f"Replies: {metadata.replies} / {metadata.max_replies}")

    if metadata.max_tokens:
        lines.append(f"Max Tokens: {metadata.max_tokens}")
    if metadata.tools:
        lines.append(f"Tools: {', '.join(metadata.tools)}")
    else:
        lines.append("Tools: none")
    lines.append(f"Logprobs: {'Yes' if metadata.logprobs else 'No'}")

    # Hashes
    lines.append(f"Content Hash: {metadata.content_hash}")
    lines.append(f"Prompt+Model Hash: {metadata.prompt_model_hash}")
    lines.append(f"Prompt Hash: {metadata.prompt_hash}")

    # Prompt messages
    lines.append("")
    lines.append("--- Prompt ---")
    for section in sections:
        if section.kind in (SectionKind.SYSTEM, SectionKind.USER, SectionKind.ASSISTANT_PREFILL):
            role = section.kind.value.replace("_", " ")
            body = section.body.strip()
            if len(body) > 200:
                body = body[:197] + "..."
            lines.append(f"[{role}] {body}")
        elif section.kind == SectionKind.TOOLS:
            lines.append("[tools] (tool definitions)")
        elif section.kind == SectionKind.USER_ATTACHMENT:
            lines.append("[user attachment] (binary content)")

    # Reply summaries
    lines.append("")
    lines.append("--- Replies ---")
    for section in sections:
        if section.kind == SectionKind.REPLY:
            reply_num = section.reply_number or "?"
            stop = section.metadata.get("Stop-Reason", "?")
            tok_in = section.metadata.get("Input-Tokens", "?")
            tok_out = section.metadata.get("Output-Tokens", "?")
            lines.append(f"Reply {reply_num}: {stop} | {tok_in} in / {tok_out} out")
        elif section.kind == SectionKind.REPLY_TOOL_CALL:
            lines.append(f"  Tool call: {section.tool_name}({section.tool_call_id})")

    return "\n".join(lines)


def format_reply_json(sections: list[TapeSection], reply_number: int | None = None, include_prompt: bool = False) -> str:
    """
    Format reply text as JSON.

    If `reply_number` is specified, returns only that reply; otherwise all replies.
    If `include_prompt` is True, also includes prompt sections.
    """
    result: dict[str, Any] = {}

    if include_prompt:
        prompt_sections = []
        for section in sections:
            if section.kind in (SectionKind.SYSTEM, SectionKind.USER, SectionKind.ASSISTANT_PREFILL, SectionKind.TOOLS,
                                SectionKind.USER_ATTACHMENT):
                prompt_sections.append(section_to_dict(section))
        result["prompt"] = prompt_sections

    replies = []
    for section in sections:
        if section.kind == SectionKind.REPLY:
            if reply_number is not None and section.reply_number != reply_number:
                continue
            reply_data: dict[str, Any] = {
                "reply_number": section.reply_number,
                "text": section.body.strip(),
                "stop_reason": section.metadata.get("Stop-Reason"),
                "input_tokens": section.metadata.get("Input-Tokens"),
                "output_tokens": section.metadata.get("Output-Tokens"),
            }
            # Collect any tool calls immediately following this reply
            tool_calls = []
            for tc_section in sections:
                if tc_section.kind == SectionKind.REPLY_TOOL_CALL and tc_section.reply_number == section.reply_number:
                    tool_calls.append({
                        "tool_name": tc_section.tool_name,
                        "tool_call_id": tc_section.tool_call_id,
                        "body": tc_section.body.strip(),
                    })
            if tool_calls:
                reply_data["tool_calls"] = tool_calls
            replies.append(reply_data)

    result["replies"] = replies
    return json.dumps(result, indent=2, ensure_ascii=False)


def format_reply_human(sections: list[TapeSection], reply_number: int | None = None, include_prompt: bool = False) -> str:
    """
    Format reply text for human-readable display.
    """
    lines = []

    if include_prompt:
        lines.append("--- Prompt ---")
        for section in sections:
            if section.kind in (SectionKind.SYSTEM, SectionKind.USER, SectionKind.ASSISTANT_PREFILL):
                role = section.kind.value.replace("_", " ")
                lines.append(f"[{role}] {section.body.strip()}")
            elif section.kind == SectionKind.TOOLS:
                lines.append("[tools] (tool definitions)")
        lines.append("")

    reply_sections = [s for s in sections if s.kind == SectionKind.REPLY]
    if reply_number is not None:
        reply_sections = [s for s in reply_sections if s.reply_number == reply_number]

    for i, section in enumerate(reply_sections):
        if len(reply_sections) > 1:
            lines.append(f"--- Reply {section.reply_number} ---")
        lines.append(section.body.strip())
        # Show tool calls for this reply
        for tc_section in sections:
            if tc_section.kind == SectionKind.REPLY_TOOL_CALL and tc_section.reply_number == section.reply_number:
                lines.append(f"\n[tool_call: {tc_section.tool_name}] {tc_section.body.strip()}")
        if i < len(reply_sections) - 1:
            lines.append("")

    if not reply_sections:
        lines.append("(no matching replies)")

    return "\n".join(lines)


def format_stats_json(total_entries: int, total_replies: int, disk_size: int, greedy_count: int, non_greedy_count: int,
                      entries_by_model: dict[str, int], total_tokens_in: int, total_tokens_out: int) -> str:
    """
    Format cache statistics as JSON.
    """
    return json.dumps(
        {
            "total_entries": total_entries,
            "total_replies": total_replies,
            "disk_size_bytes": disk_size,
            "greedy_entries": greedy_count,
            "non_greedy_entries": non_greedy_count,
            "entries_by_model": entries_by_model,
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
        }, indent=2)


def format_stats_human(total_entries: int, total_replies: int, disk_size: int, greedy_count: int, non_greedy_count: int,
                       entries_by_model: dict[str, int], total_tokens_in: int, total_tokens_out: int, cache_dir: str) -> str:
    """
    Format cache statistics for human-readable display.
    """
    lines = []
    lines.append(f"Cache directory: {cache_dir}")
    lines.append(f"Total cassettes: {total_entries}")
    lines.append(f"Total replies: {total_replies}")
    lines.append(f"Disk size: {_format_bytes(disk_size)}")
    lines.append(f"Greedy: {greedy_count} | Non-greedy: {non_greedy_count}")
    lines.append(f"Total tokens: {total_tokens_in:,} in / {total_tokens_out:,} out")

    if entries_by_model:
        lines.append("")
        lines.append("Entries by model:")
        for model_name, count in sorted(entries_by_model.items()):
            lines.append(f"  {model_name}: {count}")

    return "\n".join(lines)


def _format_bytes(size: int) -> str:
    """
    Format a byte count as a human-readable string.
    """
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    else:
        return f"{size / (1024 * 1024 * 1024):.1f} GB"
