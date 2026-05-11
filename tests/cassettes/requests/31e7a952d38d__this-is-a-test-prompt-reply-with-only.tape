---
tape_version: 2
content_hash: 31e7a952d38d
prompt_model_hash: 31e7a952d38d
prompt_hash: 767a18b7152d15ef
model: openai/gpt-oss-120b
endpoint: /v1/chat/completions
max_tokens: 200
stop_sequences: []
tools: []
logprobs: false
recorded: '2026-05-11T15:54:48.952373Z'
replies: 1
max_replies: 5
status_code: 200
boundary: c1955e
sampling:
  is_greedy: false
---
--c1955e user

This is a test prompt. Reply with **ONLY** "OK." to confirm that everything is ok. DO NOT output anything else.
--c1955e reply 1
Response: 23ccf77fd88d.json
Stream: 23ccf77fd88d.chunks.ndjson
Status: 200
Stop-Reason: stop
Input-Tokens: 34
Output-Tokens: 2

OK.
--c1955e reply 1 reasoning

The user wants only "OK." and nothing else. Must comply.
--c1955e--
