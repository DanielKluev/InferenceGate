"""
Tests for faithful recording of HTTP error responses and reasoning / CoT content in cassettes.

Covers three areas:

1. **Status code preservation** — upstream non-200 responses (e.g. vLLM 400 validation
   errors) are stored in both YAML frontmatter (``status_code``) and per-reply
   ``Status:`` metadata, and replayed to clients with their original HTTP status.
2. **Reasoning content** — ``message.reasoning_content`` (used by DeepSeek-R1,
   gpt-oss, Qwen3-thinking, etc.) is extracted into ``ReplyInfo.reasoning`` and
   rendered as a dedicated ``reply N reasoning`` MIME sub-section for readability.
3. **v1 → v2 migration** — tapes predating the ``Status:`` / reasoning fields are
   rewritten in-place on first ``CacheStorage`` open and stamped ``status_code=200``.

These tests must not mock inference output beyond the synthetic bodies needed to
represent each recording scenario: the goal is round-tripping through the real
``CacheStorage`` writer / parser / index so that on-disk behaviour is verified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from inference_gate.inflow.server import InflowServer
from inference_gate.modes import Mode
from inference_gate.outflow.client import OutflowClient
from inference_gate.recording.models import SectionKind
from inference_gate.recording.storage import CachedRequest, CachedResponse, CacheEntry, CacheStorage
from inference_gate.recording.tape_parser import parse_tape
from inference_gate.router.router import Router

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# The exact vLLM 400 payload from the bug report: prompt_token_ids with no prompt /
# prompt_embeds triggers a validation failure on the /v1/completions endpoint.
VLLM_400_REQUEST_BODY: dict[str, Any] = {
    "model": "Gemma-4-31B",
    "max_tokens": 1,
    "temperature": 0.0,
    "prompt_logprobs": 20,
    "prompt_token_ids": [105, 9731, 107, 98, 106, 107, 105, 2364, 107, 3689, 563, 506, 5279, 529, 7001, 236881, 106, 107, 105, 4368, 107, 100, 45518, 107, 101],
}

VLLM_400_RESPONSE_BODY: dict[str, Any] = {
    "error": {
        "message": ("1 validation error:\n  {'type': 'value_error', 'loc': ('body',), "
                    "'msg': 'Value error, Either prompt or prompt_embeds must be provided and non-empty.'}"),
        "type": "Bad Request",
        "param": None,
        "code": 400,
    }
}


@pytest.fixture
def storage(tmp_path) -> CacheStorage:
    """
    Build a ``CacheStorage`` rooted at a fresh temporary directory.
    """
    return CacheStorage(tmp_path / "cache")


# ---------------------------------------------------------------------------
# Status code round-trip (unit level)
# ---------------------------------------------------------------------------


class TestStatusCodeRecordingAndReplay:
    """
    Verifies that non-200 upstream responses are stored faithfully in tapes and
    replayed to clients with their original HTTP status.
    """

    def test_record_400_stamps_status_in_tape_and_index(self, storage):
        """
        Recording an upstream 400 response writes ``status_code: 400`` to YAML frontmatter,
        injects a ``Status: 400`` metadata header on the reply section, and populates
        ``IndexRow.status_code`` with 400.
        """
        request = CachedRequest(
            method="POST",
            path="/v1/completions",
            headers={"content-type": "application/json"},
            body=VLLM_400_REQUEST_BODY,
        )
        response = CachedResponse(
            status_code=400,
            headers={"content-type": "application/json"},
            body=VLLM_400_RESPONSE_BODY,
            is_streaming=False,
        )
        entry = CacheEntry(request=request, response=response, model="Gemma-4-31B", temperature=0.0)

        content_hash = storage.put(entry)

        # Tape file: frontmatter + per-reply Status header.
        tape_path = storage._find_tape_file(content_hash)
        assert tape_path is not None
        tape_text = tape_path.read_text(encoding="utf-8")
        assert "status_code: 400" in tape_text, "YAML frontmatter must record HTTP 400"
        assert "Status: 400" in tape_text, "Per-reply Status header must record HTTP 400"

        metadata, sections = parse_tape(tape_text)
        assert metadata.tape_version == 2
        assert metadata.status_code == 400
        reply_sections = [s for s in sections if s.kind == SectionKind.REPLY]
        assert len(reply_sections) == 1
        assert reply_sections[0].metadata.get("Status") == "400"

        # Index row: status_code column is populated with the integer 400.
        row = storage.index.by_content_hash[content_hash]
        assert row.status_code == 400

    def test_200_recording_stamps_explicit_status(self, storage):
        """
        Normal 200 responses also record an explicit ``Status: 200`` header so that
        v2 tapes are unambiguous (distinguishable from legacy v1 tapes lacking
        the field).
        """
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}], "temperature": 0.0},
        )
        response = CachedResponse(
            status_code=200,
            headers={},
            body={"choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}]},
        )
        entry = CacheEntry(request=request, response=response, model="gpt-4")

        content_hash = storage.put(entry)
        tape_text = storage._find_tape_file(content_hash).read_text(encoding="utf-8")
        assert "status_code: 200" in tape_text
        assert "Status: 200" in tape_text
        assert storage.index.by_content_hash[content_hash].status_code == 200

    def test_load_entry_returns_original_status_code(self, storage):
        """
        Loading a recorded entry (non-streaming) returns a ``CachedResponse`` whose
        ``status_code`` matches the originally recorded value (400), not hardcoded 200.
        """
        request = CachedRequest(method="POST", path="/v1/completions", headers={}, body=VLLM_400_REQUEST_BODY)
        response = CachedResponse(status_code=400, headers={}, body=VLLM_400_RESPONSE_BODY, is_streaming=False)
        storage.put(CacheEntry(request=request, response=response, model="Gemma-4-31B", temperature=0.0))

        entry = storage.get(request)
        assert entry is not None
        assert entry.response.status_code == 400
        assert entry.response.body == VLLM_400_RESPONSE_BODY

    def test_get_reply_metadata_exposes_status_code(self, storage):
        """
        ``CacheStorage.get_reply_metadata`` returns (response_hash, status_code)
        tuples so that the router can replay replies with their original HTTP status.
        """
        request = CachedRequest(method="POST", path="/v1/completions", headers={}, body=VLLM_400_REQUEST_BODY)
        response = CachedResponse(status_code=400, headers={}, body=VLLM_400_RESPONSE_BODY)
        content_hash = storage.put(CacheEntry(request=request, response=response, model="Gemma-4-31B", temperature=0.0))

        meta = storage.get_reply_metadata(content_hash)
        assert len(meta) == 1
        response_hash, status_code = meta[0]
        assert status_code == 400
        assert response_hash  # non-empty


# ---------------------------------------------------------------------------
# Replay-level: 400 comes back through the HTTP server with correct status
# ---------------------------------------------------------------------------


class TestStatusCodeReplayThroughServer:
    """
    Ensures the InflowServer surfaces the recorded status code to clients rather
    than unconditionally returning 200.
    """

    async def test_replay_of_400_returns_400_to_client(self, tmp_path):
        """
        After recording a 400 response into storage, the inflow server replays it
        with the original 400 status and body — not the previous ``status_code=200``
        behaviour which masked upstream validation errors.
        """
        cache_dir = tmp_path / "cache"
        storage = CacheStorage(cache_dir)

        # Pre-populate storage with a 400 cassette.
        request = CachedRequest(method="POST", path="/v1/completions", headers={}, body=VLLM_400_REQUEST_BODY)
        response = CachedResponse(status_code=400, headers={}, body=VLLM_400_RESPONSE_BODY)
        storage.put(CacheEntry(request=request, response=response, model="Gemma-4-31B", temperature=0.0))

        router = Router(mode=Mode.REPLAY_ONLY, storage=storage)
        server = InflowServer(host="127.0.0.1", port=0, router=router)
        app = server._create_app()

        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/v1/completions", json=VLLM_400_REQUEST_BODY)
            assert resp.status == 400, "Recorded 400 must replay with HTTP 400, not 200"
            data = await resp.json()
            assert data == VLLM_400_RESPONSE_BODY


# ---------------------------------------------------------------------------
# End-to-end: fake upstream returns 400 → InferenceGate records → replays correctly
# ---------------------------------------------------------------------------


class TestEndToEndErrorRecording:
    """
    Full record-and-replay flow with an aiohttp fake upstream that returns the
    exact vLLM 400 payload from the bug report.
    """

    async def test_record_400_from_fake_upstream_then_replay(self, tmp_path):
        """
        Start a fake aiohttp upstream that returns HTTP 400 on ``/v1/completions``.
        Route a first request through InferenceGate in ``RECORD_AND_REPLAY`` mode
        so the 400 is captured.  Send the same request a second time and confirm
        the cache replays HTTP 400 with the exact body — no ghost 200s.
        """
        # --- fake upstream ---
        upstream_hits = {"count": 0}

        async def handle_completions(request: web.Request) -> web.Response:
            upstream_hits["count"] += 1
            return web.json_response(VLLM_400_RESPONSE_BODY, status=400)

        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/completions", handle_completions)

        async with TestServer(upstream_app) as upstream_server:
            upstream_url = str(upstream_server.make_url("/")).rstrip("/")

            # --- InferenceGate router in record-and-replay ---
            storage = CacheStorage(tmp_path / "cache")
            outflow = OutflowClient(upstream_base_url=upstream_url)
            await outflow.start()
            try:
                router = Router(mode=Mode.RECORD_AND_REPLAY, storage=storage, outflow=outflow)
                gate_server = InflowServer(host="127.0.0.1", port=0, router=router)
                gate_app = gate_server._create_app()

                async with TestClient(TestServer(gate_app)) as client:
                    # First call — forwarded to upstream, response recorded.
                    resp1 = await client.post("/v1/completions", json=VLLM_400_REQUEST_BODY)
                    assert resp1.status == 400
                    body1 = await resp1.json()
                    assert body1 == VLLM_400_RESPONSE_BODY
                    assert upstream_hits["count"] == 1

                    # Second call — same request, served from cassette.
                    resp2 = await client.post("/v1/completions", json=VLLM_400_REQUEST_BODY)
                    assert resp2.status == 400, "Cached 400 must replay as HTTP 400, not 200"
                    body2 = await resp2.json()
                    assert body2 == VLLM_400_RESPONSE_BODY
                    # Upstream must not have been hit again.
                    assert upstream_hits["count"] == 1, "Cached request should not re-hit upstream"
            finally:
                await outflow.stop()

        # After the run, the tape must be written with status_code=400.
        tape_files = list((tmp_path / "cache" / "requests").glob("*.tape"))
        assert len(tape_files) == 1
        tape_text = tape_files[0].read_text(encoding="utf-8")
        assert "status_code: 400" in tape_text
        assert "Status: 400" in tape_text


# ---------------------------------------------------------------------------
# Reasoning content (non-streaming)
# ---------------------------------------------------------------------------


class TestReasoningRecordingNonStreaming:
    """
    Verifies that ``message.reasoning_content`` and ``message.reasoning`` are extracted
    from non-streaming response bodies and written as a ``reply N reasoning`` sub-section.
    """

    def test_reasoning_content_extracted_and_written(self, storage):
        """
        A response with ``message.reasoning_content`` yields a tape containing a
        ``reply 1 reasoning`` section whose body matches the CoT text, and
        ``IndexRow.has_reasoning`` is True.
        """
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={"model": "deepseek-r1", "messages": [{"role": "user", "content": "Solve x+1=3"}], "temperature": 0.0},
        )
        response = CachedResponse(
            status_code=200,
            headers={},
            body={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "x = 2",
                        "reasoning_content": "Let me subtract 1 from both sides. x = 3 - 1 = 2.",
                    },
                    "finish_reason": "stop",
                }]
            },
        )

        content_hash = storage.put(CacheEntry(request=request, response=response, model="deepseek-r1"))
        tape_text = storage._find_tape_file(content_hash).read_text(encoding="utf-8")

        # Sub-section header present in raw tape.
        assert "reply 1 reasoning" in tape_text

        # Parsed sections expose REPLY_REASONING kind with the reasoning body.
        _, sections = parse_tape(tape_text)
        reasoning_sections = [s for s in sections if s.kind == SectionKind.REPLY_REASONING]
        assert len(reasoning_sections) == 1
        assert reasoning_sections[0].reply_number == 1
        assert reasoning_sections[0].body == "Let me subtract 1 from both sides. x = 3 - 1 = 2."

        # Index flag set.
        row = storage.index.by_content_hash[content_hash]
        assert row.has_reasoning is True

    def test_reasoning_field_fallback(self, storage):
        """
        Providers using the shorter ``reasoning`` key (DeepSeek-R1 variant) are also
        captured into ``ReplyInfo.reasoning``.
        """
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={"model": "qwen3-thinking", "messages": [{"role": "user", "content": "Why?"}], "temperature": 0.0},
        )
        response = CachedResponse(
            status_code=200,
            headers={},
            body={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Because.",
                        "reasoning": "The user is asking a rhetorical question.",
                    },
                    "finish_reason": "stop",
                }]
            },
        )

        content_hash = storage.put(CacheEntry(request=request, response=response, model="qwen3-thinking"))
        tape_text = storage._find_tape_file(content_hash).read_text(encoding="utf-8")
        assert "The user is asking a rhetorical question." in tape_text

    def test_no_reasoning_means_no_subsection(self, storage):
        """
        Responses without reasoning content produce a tape with no
        ``REPLY_REASONING`` sections and ``has_reasoning=False`` in the index.
        """
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}], "temperature": 0.0},
        )
        response = CachedResponse(
            status_code=200,
            headers={},
            body={"choices": [{"message": {"role": "assistant", "content": "Hello."}, "finish_reason": "stop"}]},
        )

        content_hash = storage.put(CacheEntry(request=request, response=response, model="gpt-4"))
        _, sections = parse_tape(storage._find_tape_file(content_hash).read_text(encoding="utf-8"))
        assert not any(s.kind == SectionKind.REPLY_REASONING for s in sections)
        assert storage.index.by_content_hash[content_hash].has_reasoning is False


# ---------------------------------------------------------------------------
# Reasoning content (streaming)
# ---------------------------------------------------------------------------


class TestReasoningRecordingStreaming:
    """
    Streaming SSE chunks containing ``reasoning_content`` deltas reassemble into a
    merged ``message.reasoning_content`` field, which then flows through the normal
    ``_build_reply_info`` path into the tape.
    """

    def test_streaming_reasoning_deltas_captured_in_tape(self, storage):
        """
        SSE chunks with interleaved ``reasoning_content`` and ``content`` deltas
        produce a tape with a ``reply 1 reasoning`` sub-section containing the
        concatenated reasoning text and the main reply body containing the
        concatenated content text.
        """

        def sse(event: dict[str, Any]) -> str:
            return "data: " + json.dumps(event) + "\n\n"

        chunks = [
            sse({"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                 "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}),
            sse({"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                 "choices": [{"index": 0, "delta": {"reasoning_content": "Step 1. "}, "finish_reason": None}]}),
            sse({"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                 "choices": [{"index": 0, "delta": {"reasoning_content": "Step 2."}, "finish_reason": None}]}),
            sse({"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                 "choices": [{"index": 0, "delta": {"content": "Answer: "}, "finish_reason": None}]}),
            sse({"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                 "choices": [{"index": 0, "delta": {"content": "42"}, "finish_reason": "stop"}]}),
            "data: [DONE]\n\n",
        ]

        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={"model": "m", "messages": [{"role": "user", "content": "Think"}], "temperature": 0.0, "stream": True},
        )
        response = CachedResponse(status_code=200, headers={}, chunks=chunks, is_streaming=True)

        content_hash = storage.put(CacheEntry(request=request, response=response, model="m"))
        _, sections = parse_tape(storage._find_tape_file(content_hash).read_text(encoding="utf-8"))

        reply_sections = [s for s in sections if s.kind == SectionKind.REPLY]
        reasoning_sections = [s for s in sections if s.kind == SectionKind.REPLY_REASONING]
        assert len(reply_sections) == 1
        assert len(reasoning_sections) == 1
        assert reply_sections[0].body == "Answer: 42"
        assert reasoning_sections[0].body == "Step 1. Step 2."


# ---------------------------------------------------------------------------
# v1 → v2 migration
# ---------------------------------------------------------------------------


class TestV1ToV2Migration:
    """
    Verifies that tapes written in the v1 format (no ``Status:``, no ``status_code`` in
    YAML, ``tape_version: 1``) are transparently rewritten on first ``CacheStorage``
    open and stamped with ``status_code=200``.
    """

    def _write_v1_tape(self, cache_dir: Path) -> Path:
        """
        Hand-craft a minimal v1 tape + response file for migration tests.
        """
        requests_dir = cache_dir / "requests"
        responses_dir = cache_dir / "responses"
        requests_dir.mkdir(parents=True, exist_ok=True)
        responses_dir.mkdir(parents=True, exist_ok=True)

        # Write a response JSON file.
        response_hash = "deadbeefcafe"
        (responses_dir / f"{response_hash}.json").write_text(
            json.dumps({"choices": [{"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}]}),
            encoding="utf-8",
        )

        # Write a v1 tape with no Status: header and tape_version: 1.
        boundary = "aabbccdd"
        content_hash = "c0ffee001122"
        tape_text = (
            "---\n"
            "tape_version: 1\n"
            f"content_hash: {content_hash}\n"
            f"prompt_model_hash: {content_hash}\n"
            "prompt_hash: 1234567890abcdef\n"
            "model: legacy-model\n"
            "endpoint: /v1/chat/completions\n"
            "sampling:\n"
            "  is_greedy: true\n"
            "  temperature: 0.0\n"
            "max_tokens: null\n"
            "stop_sequences: []\n"
            "tools: []\n"
            "tool_choice: null\n"
            "logprobs: false\n"
            "top_logprobs: null\n"
            "recorded: '2025-01-01T00:00:00+00:00'\n"
            "replies: 1\n"
            "max_replies: 1\n"
            f"boundary: {boundary}\n"
            "---\n"
            f"--{boundary} user\n\n"
            "Hi\n"
            f"--{boundary} reply 1\n"
            f"Response: {response_hash}.json\n"
            "Stop-Reason: stop\n\n"
            "Hi\n"
            f"--{boundary}--\n"
        )
        tape_path = requests_dir / f"{content_hash}__legacy.tape"
        tape_path.write_text(tape_text, encoding="utf-8")
        return tape_path

    def test_v1_tape_is_migrated_to_v2_on_open(self, tmp_path):
        """
        Opening a ``CacheStorage`` whose directory contains v1 tapes rewrites each
        tape with ``tape_version: 2``, ``status_code: 200`` in YAML, and ``Status: 200``
        in every reply section.  The tape body is otherwise preserved.
        """
        cache_dir = tmp_path / "cache"
        tape_path = self._write_v1_tape(cache_dir)

        # Confirm the pre-migration file is v1 with no Status: header.
        pre_text = tape_path.read_text(encoding="utf-8")
        assert "tape_version: 1" in pre_text
        assert "Status:" not in pre_text

        storage = CacheStorage(cache_dir)

        # Tape file is now v2 with explicit status info.
        post_text = tape_path.read_text(encoding="utf-8")
        assert "tape_version: 2" in post_text
        assert "status_code: 200" in post_text
        assert "Status: 200" in post_text

        metadata, sections = parse_tape(post_text)
        assert metadata.tape_version == 2
        assert metadata.status_code == 200
        reply_sections = [s for s in sections if s.kind == SectionKind.REPLY]
        assert reply_sections[0].metadata.get("Status") == "200"

        # Body unchanged — reply text still "Hi".
        assert reply_sections[0].body == "Hi"

        # Index picked up the defaults.
        assert len(storage.index) == 1
        row = next(iter(storage.index.by_content_hash.values()))
        assert row.status_code == 200
        assert row.has_reasoning is False

    def test_migration_is_idempotent(self, tmp_path):
        """
        Running ``CacheStorage`` twice on the same directory does not re-migrate
        already-v2 tapes: the file content is byte-identical between the two opens.
        """
        cache_dir = tmp_path / "cache"
        self._write_v1_tape(cache_dir)

        CacheStorage(cache_dir)  # first open migrates
        tape_path = next((cache_dir / "requests").glob("*.tape"))
        first_content = tape_path.read_text(encoding="utf-8")

        CacheStorage(cache_dir)  # second open should be a no-op for already-v2 tapes
        second_content = tape_path.read_text(encoding="utf-8")

        assert first_content == second_content


# ---------------------------------------------------------------------------
# Parser regression: reasoning header recognition
# ---------------------------------------------------------------------------


class TestParserReasoningHeader:
    """
    Direct parser tests for the new ``reply N reasoning`` section header.
    """

    def test_reasoning_header_classified_as_reply_reasoning(self):
        """
        A tape containing a ``reply 1 reasoning`` section is parsed as
        ``SectionKind.REPLY_REASONING`` with the correct ``reply_number``.
        """
        boundary = "abcdef"
        tape_text = (
            "---\n"
            "tape_version: 2\n"
            "content_hash: aa\n"
            "prompt_model_hash: aa\n"
            "prompt_hash: bb\n"
            "model: m\n"
            "endpoint: /v1/chat/completions\n"
            "sampling:\n"
            "  is_greedy: true\n"
            "  temperature: 0.0\n"
            "recorded: '2026-04-19T00:00:00+00:00'\n"
            "replies: 1\n"
            "max_replies: 1\n"
            "status_code: 200\n"
            f"boundary: {boundary}\n"
            "---\n"
            f"--{boundary} user\n\nHi\n"
            f"--{boundary} reply 1\nResponse: x.json\nStatus: 200\n\nAnswer\n"
            f"--{boundary} reply 1 reasoning\n\nThinking out loud\n"
            f"--{boundary}--\n"
        )

        metadata, sections = parse_tape(tape_text)
        assert metadata.status_code == 200
        reasoning = [s for s in sections if s.kind == SectionKind.REPLY_REASONING]
        assert len(reasoning) == 1
        assert reasoning[0].reply_number == 1
        assert reasoning[0].body == "Thinking out loud"
