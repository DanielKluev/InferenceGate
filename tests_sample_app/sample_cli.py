"""
`sample_cli` is a deliberately tiny, stdlib-only "downstream consumer" that
demonstrates how a third-party CLI can use InferenceGate as a drop-in
OpenAI-compatible base URL.

The CLI takes ``--base-url`` and ``--prompt``, posts a chat-completions
request, and prints the assistant's reply.  It is intentionally written
against the stdlib (no ``openai`` SDK, no ``httpx``) so that the surrounding
test in ``test_sample_app.py`` proves Gate integration works for *any*
HTTP client \u2014 not just the SDKs the project happens to vendor.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import urllib.parse

DEFAULT_MODEL = "openai/gpt-oss-120b"


def chat_completion(base_url: str, prompt: str, *, model: str = DEFAULT_MODEL, max_tokens: int = 200, timeout: float = 10.0) -> str:
    """
    Post a single chat-completions request to ``{base_url}/v1/chat/completions``.

    `base_url` is the InferenceGate proxy URL (no trailing ``/v1``).
    `prompt` is the single user-turn text.  Returns the assistant reply
    string.  Raises ``RuntimeError`` on non-200 responses with the body
    snippet attached.
    """
    parsed = urllib.parse.urlparse(base_url.rstrip("/"))
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported scheme in base_url: {base_url!r}")
    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parsed.hostname, parsed.port, timeout=timeout)
    body = json.dumps({
        "model": model,
        "messages": [{
            "role": "user",
            "content": prompt
        }],
        "max_tokens": max_tokens,
    })
    try:
        path = (parsed.path.rstrip("/") if parsed.path else "") + "/v1/chat/completions"
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"InferenceGate returned status {resp.status}: {raw[:200]!r}")
        data = json.loads(raw)
        return data["choices"][0]["message"]["content"]
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point.  Parse ``argv``, dispatch to :func:`chat_completion`,
    print the reply on stdout, and return a process exit code.
    """
    parser = argparse.ArgumentParser(prog="sample_cli", description="Tiny OpenAI-compatible client for InferenceGate")
    parser.add_argument("--base-url", required=True, help="InferenceGate base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--prompt", required=True, help="User prompt text")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args(argv)
    try:
        reply = chat_completion(args.base_url, args.prompt, model=args.model, max_tokens=args.max_tokens)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
