"""
`atomic_io` provides atomic write helpers used by the storage layer.

Writes go to a temp file in the same directory and are then `os.replace()`-d
to their final path so concurrent readers never observe a partial file.
Used by tape, response JSON, NDJSON and index.tsv writers.

Key functions: `atomic_write_text`, `atomic_write_bytes`.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """
    Atomically write `content` to `path` via a temp file + `os.replace`.

    Creates parent directories as needed. The temp file is created in the same
    directory so the final rename is atomic on POSIX and Windows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        tmp.write_text(content, encoding=encoding)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """
    Atomically write `data` to `path` via a temp file + `os.replace`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
