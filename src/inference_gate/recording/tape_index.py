"""
TSV-based index for fast cassette lookup across the tape storage.

Manages `index.tsv` — a tab-separated file with one row per cassette.
On startup, the TSV is loaded into in-memory dicts keyed by the three
hash tiers (`content_hash`, `prompt_model_hash`, `prompt_hash`) for
O(1) lookup at each fuzzy level.

The index is regenerable from tape frontmatter via `rebuild()`.

Key classes: `TapeIndex`
"""

import csv
import io
import logging
from pathlib import Path

from inference_gate.recording.atomic_io import atomic_write_text
from inference_gate.recording.models import IndexRow, SectionKind, TapeMetadata, TapeSection
from inference_gate.recording.tape_parser import parse_tape

# TSV column order — must match IndexRow fields
_TSV_COLUMNS = [
    "content_hash",
    "prompt_model_hash",
    "prompt_hash",
    "model",
    "endpoint",
    "is_greedy",
    "temperature",
    "tokens_in",
    "tokens_out",
    "replies",
    "max_replies",
    "has_logprobs",
    "has_tool_use",
    "status_code",
    "has_reasoning",
    "engine",
    "slug",
    "recorded",
    "first_user_message",
]


class TapeIndex:
    """
    In-memory index backed by a TSV file for fast cassette lookup.

    Provides three lookup dicts:
    - `by_content_hash`: exact match (one entry per hash)
    - `by_prompt_model_hash`: sampling-fuzzy (multiple entries per hash)
    - `by_prompt_hash`: model-fuzzy (multiple entries per hash)
    """

    def __init__(self, index_path: Path) -> None:
        """
        Initialize the tape index.

        `index_path` is the path to the `index.tsv` file.
        The index is loaded from disk if the file exists, otherwise
        it starts empty.
        """
        self.log = logging.getLogger("TapeIndex")
        self.index_path = index_path
        self.by_content_hash: dict[str, IndexRow] = {}
        self.by_prompt_model_hash: dict[str, list[IndexRow]] = {}
        self.by_prompt_hash: dict[str, list[IndexRow]] = {}
        self._rows: list[IndexRow] = []
        self._needs_rebuild = False

        if index_path.exists():
            self._load()

    def _load(self) -> None:
        """
        Load the index from the TSV file into memory.
        """
        self.by_content_hash.clear()
        self.by_prompt_model_hash.clear()
        self.by_prompt_hash.clear()
        self._rows.clear()

        try:
            text = self.index_path.read_text(encoding="utf-8")
        except OSError:
            self.log.warning("Could not read index file %s", self.index_path)
            return

        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        # Detect column mismatch — if the TSV was written by an older version
        # with fewer columns, flag for rebuild so that tape frontmatter data
        # (e.g. endpoint) can be backfilled.
        if reader.fieldnames and set(reader.fieldnames) != set(_TSV_COLUMNS):
            self._needs_rebuild = True
            self.log.info("Index columns differ from current schema — will rebuild from tapes on next opportunity")
        for raw_row in reader:
            try:
                row = _parse_tsv_row(raw_row)
            except (ValueError, KeyError) as exc:
                self.log.debug("Skipping malformed index row: %s", exc)
                continue
            self._add_row_to_indexes(row)

        self.log.debug("Loaded %d entries from index", len(self._rows))

    def _add_row_to_indexes(self, row: IndexRow) -> None:
        """
        Add an IndexRow to all in-memory lookup structures.
        """
        self._rows.append(row)
        self.by_content_hash[row.content_hash] = row
        self.by_prompt_model_hash.setdefault(row.prompt_model_hash, []).append(row)
        self.by_prompt_hash.setdefault(row.prompt_hash, []).append(row)

    def add(self, row: IndexRow) -> None:
        """
        Add (or update) an entry in the index and append to the TSV file.

        If an entry with the same `content_hash` already exists, it is replaced.
        """
        # Remove old entry if it exists
        existing = self.by_content_hash.get(row.content_hash)
        if existing is not None:
            self._remove_row(existing)

        self._add_row_to_indexes(row)
        self._write_tsv()

    def update_replies(self, content_hash: str, new_replies_count: int, tokens_in: str = "", tokens_out: str = "",
                       has_reasoning: bool | None = None) -> None:
        """
        Update the reply count (and optionally token counts / reasoning flag) for an existing entry.

        When ``has_reasoning`` is provided, the flag is OR-ed with the existing value so
        that a multi-reply cassette keeps its ``has_reasoning=True`` status once any
        reply carries reasoning content.
        """
        row = self.by_content_hash.get(content_hash)
        if row is None:
            self.log.warning("Cannot update replies: content_hash %s not in index", content_hash)
            return

        # Pydantic model — create updated copy
        update_fields: dict[str, object] = {
            "replies": new_replies_count,
            "tokens_in": tokens_in or row.tokens_in,
            "tokens_out": tokens_out or row.tokens_out,
        }
        if has_reasoning is not None:
            update_fields["has_reasoning"] = row.has_reasoning or has_reasoning
        updated = row.model_copy(update=update_fields)

        # Replace in all structures
        self._remove_row(row)
        self._add_row_to_indexes(updated)
        self._write_tsv()

    def _remove_row(self, row: IndexRow) -> None:
        """
        Remove a row from all in-memory structures.
        """
        if row in self._rows:
            self._rows.remove(row)

        if self.by_content_hash.get(row.content_hash) is row:
            del self.by_content_hash[row.content_hash]

        pm_list = self.by_prompt_model_hash.get(row.prompt_model_hash, [])
        if row in pm_list:
            pm_list.remove(row)
            if not pm_list:
                del self.by_prompt_model_hash[row.prompt_model_hash]

        p_list = self.by_prompt_hash.get(row.prompt_hash, [])
        if row in p_list:
            p_list.remove(row)
            if not p_list:
                del self.by_prompt_hash[row.prompt_hash]

    def remove(self, content_hash: str) -> bool:
        """
        Remove a row from the index by content_hash.

        Updates in-memory structures and rewrites the TSV file.
        Returns True if the row was found and removed, False otherwise.
        """
        row = self.by_content_hash.get(content_hash)
        if row is None:
            return False
        self._remove_row(row)
        self._write_tsv()
        return True

    def rebuild(self, requests_dir: Path) -> None:
        """
        Rebuild the entire index from tape files in `requests_dir`.

        Scans all `.tape` files, parses frontmatter and body sections,
        and regenerates ``index.tsv`` from scratch.  Body sections are
        parsed so that ``first_user_message`` can be extracted from the
        conversation content (including raw ``prompt`` tapes).
        """
        self.by_content_hash.clear()
        self.by_prompt_model_hash.clear()
        self.by_prompt_hash.clear()
        self._rows.clear()

        for tape_file in sorted(requests_dir.glob("*.tape")):
            try:
                content = tape_file.read_text(encoding="utf-8")
                meta, sections = parse_tape(content)
                slug = _extract_slug_from_filename(tape_file.name)
                # Extract first user message from body sections
                first_user_msg = _extract_first_user_from_sections(sections)
                # Any non-empty reasoning sub-section marks the cassette as having reasoning.
                has_reasoning = any(
                    section.kind == SectionKind.REPLY_REASONING and bool(section.body) for section in sections)
                row = IndexRow.from_tape_metadata(meta, slug=slug, first_user_message=first_user_msg, has_reasoning=has_reasoning)
                self._add_row_to_indexes(row)
            except Exception as exc:
                self.log.warning("Skipping unreadable tape %s during reindex: %s", tape_file.name, exc)
                continue

        self._write_tsv()
        self.log.info("Rebuilt index with %d entries from %s", len(self._rows), requests_dir)

    def _write_tsv(self) -> None:
        """
        Write the full index to the TSV file.
        """
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=_TSV_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in self._rows:
            writer.writerow(_row_to_tsv_dict(row))
        atomic_write_text(self.index_path, buf.getvalue())

    def __len__(self) -> int:
        return len(self._rows)

    def __contains__(self, content_hash: str) -> bool:
        return content_hash in self.by_content_hash


def _row_to_tsv_dict(row: IndexRow) -> dict[str, str]:
    """
    Convert an IndexRow to a dict of string values for TSV writing.
    """
    return {
        "content_hash": row.content_hash,
        "prompt_model_hash": row.prompt_model_hash,
        "prompt_hash": row.prompt_hash,
        "model": row.model,
        "endpoint": row.endpoint,
        "is_greedy": "true" if row.is_greedy else "false",
        "temperature": row.temperature,
        "tokens_in": row.tokens_in,
        "tokens_out": row.tokens_out,
        "replies": str(row.replies),
        "max_replies": str(row.max_replies),
        "has_logprobs": "true" if row.has_logprobs else "false",
        "has_tool_use": "true" if row.has_tool_use else "false",
        "status_code": str(row.status_code),
        "has_reasoning": "true" if row.has_reasoning else "false",
        "engine": row.engine,
        "slug": row.slug,
        "recorded": row.recorded,
        "first_user_message": row.first_user_message,
    }


def _parse_tsv_row(raw: dict[str, str | None]) -> IndexRow:
    """
    Parse a raw TSV row dict into an IndexRow.

    Tolerates missing ``status_code`` / ``has_reasoning`` columns (v1 index files
    written before v2 format support) by defaulting to 200 / False.  When such
    columns are absent the ``TapeIndex`` load path flags the file for rebuild
    so that the values can be backfilled from tape frontmatter / body sections.
    """
    status_str = raw.get("status_code", "") or ""
    try:
        status_code = int(status_str) if status_str else 200
    except ValueError:
        status_code = 200
    return IndexRow(
        content_hash=raw.get("content_hash", "") or "",
        prompt_model_hash=raw.get("prompt_model_hash", "") or "",
        prompt_hash=raw.get("prompt_hash", "") or "",
        model=raw.get("model", "") or "",
        endpoint=raw.get("endpoint", "") or "",
        is_greedy=(raw.get("is_greedy", "") or "").lower() == "true",
        temperature=raw.get("temperature", "") or "",
        tokens_in=raw.get("tokens_in", "") or "",
        tokens_out=raw.get("tokens_out", "") or "",
        replies=int(raw.get("replies", "0") or "0"),
        max_replies=int(raw.get("max_replies", "1") or "1"),
        has_logprobs=(raw.get("has_logprobs", "") or "").lower() == "true",
        has_tool_use=(raw.get("has_tool_use", "") or "").lower() == "true",
        status_code=status_code,
        has_reasoning=(raw.get("has_reasoning", "") or "").lower() == "true",
        engine=raw.get("engine", "") or "",
        slug=raw.get("slug", "") or "",
        recorded=raw.get("recorded", "") or "",
        first_user_message=raw.get("first_user_message", "") or "",
    )


def _extract_slug_from_filename(filename: str) -> str:
    """
    Extract the slug portion from a tape filename.

    Given `a3f8c1b2d9e7__hello-world.tape`, returns `hello-world`.
    """
    name = filename.removesuffix(".tape")
    if "__" in name:
        return name.split("__", 1)[1]
    return ""


def _extract_first_user_from_sections(sections: list[TapeSection], max_length: int = 120) -> str:
    """
    Extract the first user message text from parsed tape sections.

    Looks for the first ``USER`` section and returns its body text,
    truncated to ``max_length`` with newlines/tabs replaced by spaces.
    Used during index rebuild to populate ``first_user_message``.
    """
    for section in sections:
        if section.kind == SectionKind.USER and section.body:
            text = section.body[:max_length]
            text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
            return text
    return ""
