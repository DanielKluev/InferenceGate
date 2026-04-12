"""Tests for InferenceGate recording/reassembly module."""

import json

import pytest

from inference_gate.recording.reassembly import (
    _parse_sse_events,
    reassemble_chat_completion,
    reassemble_responses_api,
    reassemble_streaming_response,
    reassemble_text_completion,
)


class TestParseSSEEvents:
    """Tests for SSE event parsing."""

    def test_basic_sse_parsing(self):
        """Test that basic SSE data lines are parsed into JSON objects."""
        chunks = ['data: {"id":"1","choices":[]}\n\n', 'data: {"id":"2","choices":[]}\n\n']
        events = _parse_sse_events(chunks)
        assert len(events) == 2
        assert events[0]["id"] == "1"
        assert events[1]["id"] == "2"

    def test_done_marker_skipped(self):
        """Test that data: [DONE] lines are skipped."""
        chunks = ['data: {"id":"1"}\n\ndata: [DONE]\n\n']
        events = _parse_sse_events(chunks)
        assert len(events) == 1

    def test_empty_chunks(self):
        """Test that empty chunks return no events."""
        events = _parse_sse_events([])
        assert events == []
        events = _parse_sse_events(["", "\n\n"])
        assert events == []

    def test_non_json_lines_skipped(self):
        """Test that non-JSON data lines are skipped gracefully."""
        chunks = ['data: not-json\n\ndata: {"id":"1"}\n\n']
        events = _parse_sse_events(chunks)
        assert len(events) == 1
        assert events[0]["id"] == "1"

    def test_event_split_across_chunks(self):
        """Test that events split across network chunks are reassembled correctly."""
        # A single SSE event split into two raw chunks
        chunks = ['data: {"id":"1","cho', 'ices":[]}\n\ndata: {"id":"2","choices":[]}\n\n']
        events = _parse_sse_events(chunks)
        assert len(events) == 2

    def test_multiple_events_in_one_chunk(self):
        """Test that multiple events in a single chunk are all parsed."""
        chunks = ['data: {"id":"1"}\n\ndata: {"id":"2"}\n\ndata: {"id":"3"}\n\n']
        events = _parse_sse_events(chunks)
        assert len(events) == 3

    def test_non_data_lines_ignored(self):
        """Test that event:, id:, and comment lines are ignored."""
        chunks = ['event: message\nid: 123\n: comment\ndata: {"id":"1"}\n\n']
        events = _parse_sse_events(chunks)
        assert len(events) == 1


class TestReassembleChatCompletion:
    """Tests for Chat Completions streaming reassembly."""

    def test_simple_content_reassembly(self):
        """Test reassembly of simple content deltas into a complete message."""
        chunks = [
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" World"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n',
        ]
        result = reassemble_chat_completion(chunks)

        assert result["id"] == "chatcmpl-123"
        assert result["object"] == "chat.completion"
        assert result["model"] == "gpt-4"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["choices"][0]["message"]["content"] == "Hello World"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_with_usage_data(self):
        """Test that usage data from the final chunk is included in reassembled response."""
        chunks = [
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}\n\n',
            'data: [DONE]\n\n',
        ]
        result = reassemble_chat_completion(chunks)

        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 1
        assert result["usage"]["total_tokens"] == 11

    def test_multiple_choices(self):
        """Test reassembly with multiple choices (n > 1 parameter)."""
        chunks = [
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"A"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":1,"delta":{"role":"assistant","content":"B"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":1,"delta":{},"finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n',
        ]
        result = reassemble_chat_completion(chunks)

        assert len(result["choices"]) == 2
        assert result["choices"][0]["message"]["content"] == "A"
        assert result["choices"][0]["index"] == 0
        assert result["choices"][1]["message"]["content"] == "B"
        assert result["choices"][1]["index"] == 1

    def test_tool_calls_reassembly(self):
        """Test reassembly of tool call deltas with argument concatenation."""
        chunks = [
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\""}}]},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":": \\"London\\"}"}}]},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
            'data: [DONE]\n\n',
        ]
        result = reassemble_chat_completion(chunks)

        message = result["choices"][0]["message"]
        assert "tool_calls" in message
        assert len(message["tool_calls"]) == 1
        tc = message["tool_calls"][0]
        assert tc["id"] == "call_abc"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "London"}
        assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_empty_chunks_returns_empty(self):
        """Test that empty chunk list returns empty dict."""
        assert reassemble_chat_completion([]) == {}

    def test_only_done_returns_empty(self):
        """Test that a stream with only [DONE] returns empty dict."""
        assert reassemble_chat_completion(["data: [DONE]\n\n"]) == {}


class TestReassembleResponsesAPI:
    """Tests for Responses API streaming reassembly."""

    def test_response_completed_event(self):
        """Test that response.completed event is extracted correctly."""
        response_obj = {
            "id": "resp_123",
            "object": "response",
            "status": "completed",
            "output": [{
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": "Hello World"
                }]
            }],
        }
        chunks = [
            'event: response.created\ndata: {"id":"resp_123","status":"in_progress"}\n\n',
            'event: response.output_item.added\ndata: {"type":"message"}\n\n',
            'event: response.output_text.delta\ndata: {"delta":"Hello "}\n\n',
            'event: response.output_text.delta\ndata: {"delta":"World"}\n\n',
            f'event: response.completed\ndata: {json.dumps(response_obj)}\n\n',
        ]
        result = reassemble_responses_api(chunks)

        assert result["id"] == "resp_123"
        assert result["status"] == "completed"
        assert result["output"][0]["content"][0]["text"] == "Hello World"

    def test_fallback_without_completed_event(self):
        """Test fallback behavior when response.completed event is missing."""
        chunks = [
            'data: {"id":"resp_123","status":"in_progress"}\n\n',
            'data: {"id":"resp_123","status":"partial"}\n\n',
        ]
        result = reassemble_responses_api(chunks)
        # Should fall back to last event
        assert result["id"] == "resp_123"

    def test_empty_chunks(self):
        """Test that empty chunks return empty dict."""
        assert reassemble_responses_api([]) == {}


class TestReassembleStreamingResponse:
    """Tests for the dispatcher function."""

    def test_chat_completions_path(self):
        """Test that /v1/chat/completions path dispatches to Chat Completions reassembly."""
        chunks = [
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n',
        ]
        result = reassemble_streaming_response(chunks, "/v1/chat/completions")
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hi"

    def test_responses_path(self):
        """Test that /v1/responses path dispatches to Responses API reassembly."""
        response_obj = {"id": "resp_1", "object": "response", "status": "completed", "output": []}
        chunks = [f'event: response.completed\ndata: {json.dumps(response_obj)}\n\n']
        result = reassemble_streaming_response(chunks, "/v1/responses")
        assert result["id"] == "resp_1"

    def test_unknown_path_defaults_to_chat(self):
        """Test that an unknown path defaults to Chat Completions reassembly."""
        chunks = [
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"OK"},"finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n',
        ]
        result = reassemble_streaming_response(chunks, "/custom/path")
        assert result["choices"][0]["message"]["content"] == "OK"

    def test_v1_completions_dispatches_to_text_completion(self):
        """Test that /v1/completions path dispatches to text completion reassembly."""
        chunks = [
            'data: {"id":"cmpl-1","object":"text_completion","created":1700000000,"model":"gpt-4","choices":[{"index":0,"text":"Hello","finish_reason":null}]}\n\n',
            'data: {"id":"cmpl-1","object":"text_completion","created":1700000000,"model":"gpt-4","choices":[{"index":0,"text":" World","finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n',
        ]
        result = reassemble_streaming_response(chunks, "/v1/completions")
        assert result["object"] == "text_completion"
        assert result["choices"][0]["text"] == "Hello World"

    def test_chat_completions_not_confused_with_completions(self):
        """Test that /v1/chat/completions is not matched by the /completions text completion handler."""
        chunks = [
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n',
        ]
        result = reassemble_streaming_response(chunks, "/v1/chat/completions")
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hi"


class TestLogprobsAccumulation:
    """Tests for correct logprobs accumulation across streaming chunks."""

    def test_chat_completion_logprobs_accumulated(self):
        """Test that logprobs.content entries are accumulated across chunks, not overwritten."""
        chunks = [
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"logprobs":{"content":[{"token":"Hello","logprob":-1.5,"bytes":[72,101,108,108,111],"top_logprobs":[]}]},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" World"},"logprobs":{"content":[{"token":" World","logprob":-0.8,"bytes":[32,87,111,114,108,100],"top_logprobs":[]}]},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n',
        ]
        result = reassemble_chat_completion(chunks)

        logprobs = result["choices"][0]["logprobs"]
        assert logprobs is not None
        assert len(logprobs["content"]) == 2
        assert logprobs["content"][0]["token"] == "Hello"
        assert logprobs["content"][0]["logprob"] == -1.5
        assert logprobs["content"][1]["token"] == " World"
        assert logprobs["content"][1]["logprob"] == -0.8

    def test_chat_completion_logprobs_refusal_accumulated(self):
        """Test that logprobs.refusal entries are accumulated across chunks."""
        chunks = [
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"refusal":"I"},"logprobs":{"refusal":[{"token":"I","logprob":-0.2,"bytes":[73],"top_logprobs":[]}]},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"refusal":" cannot"},"logprobs":{"refusal":[{"token":" cannot","logprob":-0.5,"bytes":[32,99],"top_logprobs":[]}]},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n',
        ]
        result = reassemble_chat_completion(chunks)

        logprobs = result["choices"][0]["logprobs"]
        assert logprobs is not None
        assert len(logprobs["refusal"]) == 2
        assert logprobs["refusal"][0]["token"] == "I"
        assert logprobs["refusal"][1]["token"] == " cannot"

    def test_chat_completion_no_logprobs_stays_none(self):
        """Test that logprobs stays None when no chunks carry logprobs data."""
        chunks = [
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n',
        ]
        result = reassemble_chat_completion(chunks)
        assert result["choices"][0]["logprobs"] is None


class TestReassembleTextCompletion:
    """Tests for text completions streaming reassembly (``/v1/completions``)."""

    def test_simple_text_reassembly(self):
        """Test that text fragments are concatenated across SSE chunks."""
        chunks = [
            'data: {"id":"cmpl-1","object":"text_completion","created":1700000000,"model":"gpt-4","choices":[{"index":0,"text":"Hello","finish_reason":null}]}\n\n',
            'data: {"id":"cmpl-1","object":"text_completion","created":1700000000,"model":"gpt-4","choices":[{"index":0,"text":" World","finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n',
        ]
        result = reassemble_text_completion(chunks)

        assert result["id"] == "cmpl-1"
        assert result["object"] == "text_completion"
        assert result["model"] == "gpt-4"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["text"] == "Hello World"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_prompt_logprobs_preserved(self):
        """Test that prompt_logprobs from vLLM's first chunk are preserved in the reassembled response."""
        prompt_logprobs = [
            None, {
                "100": {
                    "logprob": -0.5,
                    "rank": 1,
                    "decoded_token": "B"
                }
            }, {
                "200": {
                    "logprob": -1.2,
                    "rank": 1,
                    "decoded_token": "."
                }
            }
        ]
        first_chunk = {
            "id": "cmpl-1",
            "object": "text_completion",
            "created": 1700000000,
            "model": "qwen3-4b-it",
            "choices": [{
                "index": 0,
                "text": "",
                "finish_reason": None,
                "prompt_logprobs": prompt_logprobs,
            }],
        }
        gen_chunk = {
            "id": "cmpl-1",
            "object": "text_completion",
            "created": 1700000000,
            "model": "qwen3-4b-it",
            "choices": [{
                "index": 0,
                "text": "ok",
                "finish_reason": "length",
            }],
        }
        chunks = [
            f'data: {json.dumps(first_chunk)}\n\n',
            f'data: {json.dumps(gen_chunk)}\n\n',
            'data: [DONE]\n\n',
        ]
        result = reassemble_text_completion(chunks)

        assert result["choices"][0]["prompt_logprobs"] is not None
        assert len(result["choices"][0]["prompt_logprobs"]) == 3
        assert result["choices"][0]["prompt_logprobs"][0] is None
        assert "100" in result["choices"][0]["prompt_logprobs"][1]
        assert result["choices"][0]["text"] == "ok"

    def test_generation_logprobs_accumulated(self):
        """Test that per-token generation logprobs are accumulated across chunks."""
        chunks = [
            'data: {"id":"cmpl-1","object":"text_completion","created":1700000000,"model":"gpt-4","choices":[{"index":0,"text":"A","logprobs":{"tokens":["A"],"token_logprobs":[-1.0],"top_logprobs":[{"A":-1.0,"B":-2.0}],"text_offset":[0]},"finish_reason":null}]}\n\n',
            'data: {"id":"cmpl-1","object":"text_completion","created":1700000000,"model":"gpt-4","choices":[{"index":0,"text":"B","logprobs":{"tokens":["B"],"token_logprobs":[-0.5],"top_logprobs":[{"B":-0.5,"A":-1.5}],"text_offset":[1]},"finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n',
        ]
        result = reassemble_text_completion(chunks)

        logprobs = result["choices"][0]["logprobs"]
        assert logprobs is not None
        assert logprobs["tokens"] == ["A", "B"]
        assert logprobs["token_logprobs"] == [-1.0, -0.5]
        assert len(logprobs["top_logprobs"]) == 2
        assert logprobs["text_offset"] == [0, 1]

    def test_usage_captured(self):
        """Test that usage data from the final chunk is included in reassembled response."""
        chunks = [
            'data: {"id":"cmpl-1","object":"text_completion","created":1700000000,"model":"gpt-4","choices":[{"index":0,"text":"Hi","finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}\n\n',
            'data: [DONE]\n\n',
        ]
        result = reassemble_text_completion(chunks)

        assert result["usage"]["prompt_tokens"] == 5
        assert result["usage"]["completion_tokens"] == 1

    def test_empty_chunks_returns_empty(self):
        """Test that empty chunk list returns empty dict."""
        assert reassemble_text_completion([]) == {}
