"""
End-to-end test imitating a downstream consumer using InferenceGate as a
``pytest`` fixture/plugin.

The test invokes ``sample_cli`` (a stdlib-only mini chat-completions client)
against the live ``inference_gate_url`` provided by the InferenceGate plugin.
The Gate runs in replay-only mode against the bundled cassettes \u2014 no live
upstream is contacted \u2014 so this test stays deterministic in CI.

This is the canonical integration smoke test for a "third-party" project
depending on the Gate plugin: zero conftest wiring, just declare the
fixture and go.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

# Load `sample_cli.py` as a sibling module without requiring an __init__.py.
# A real downstream project would simply ``import myapp`` from its installed
# package; we use importlib here only to keep this directory a flat, no-package
# example mirroring a "scripts + tests" layout.
_HERE = pathlib.Path(__file__).parent
SAMPLE_CLI_PATH = _HERE / "sample_cli.py"
_spec = importlib.util.spec_from_file_location("sample_cli", SAMPLE_CLI_PATH)
assert _spec is not None and _spec.loader is not None
sample_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sample_cli)

# Cassette-backed prompt; matches a recording committed in tests/cassettes/.
OK_PROMPT = 'This is a test prompt. Reply with **ONLY** "OK." to confirm that everything is ok. DO NOT output anything else.'
CASSETTE_MODEL = "openai/gpt-oss-120b"


def test_sample_cli_function_call(inference_gate_url: str) -> None:
    """
    Calling :func:`sample_cli.chat_completion` directly returns the cassette
    reply text without raising.

    Verifies the in-process function path: ``base_url`` is parsed correctly,
    HTTP request is dispatched, response JSON is unwrapped to a string.
    """
    reply = sample_cli.chat_completion(inference_gate_url, OK_PROMPT, model=CASSETTE_MODEL)
    assert isinstance(reply, str)
    # The recorded cassette reply contains "OK" \u2014 we accept any substring
    # match to stay robust against minor whitespace/punctuation differences.
    assert "OK" in reply, f"unexpected cassette reply: {reply!r}"


def test_sample_cli_subprocess(inference_gate_url: str) -> None:
    """
    Running ``sample_cli`` as a subprocess against ``inference_gate_url``
    succeeds and prints the cassette reply on stdout.

    This proves the plugin's session-scoped Gate URL is genuinely usable
    by an external process \u2014 the URL is on a real TCP socket, not a
    pytest-internal in-memory transport.
    """
    cmd = [
        sys.executable,
        str(SAMPLE_CLI_PATH),
        "--base-url",
        inference_gate_url,
        "--prompt",
        OK_PROMPT,
        "--model",
        CASSETTE_MODEL,
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
    assert completed.returncode == 0, (f"sample_cli failed (rc={completed.returncode})\n"
                                       f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}")
    assert "OK" in completed.stdout, f"sample_cli stdout missing 'OK': {completed.stdout!r}"
