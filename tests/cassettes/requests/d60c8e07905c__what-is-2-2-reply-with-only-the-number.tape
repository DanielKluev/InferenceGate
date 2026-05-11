---
tape_version: 2
content_hash: d60c8e07905c
prompt_model_hash: d60c8e07905c
prompt_hash: 8c81e05fef75a909
model: openai/gpt-oss-120b
endpoint: /v1/chat/completions
max_tokens: 200
stop_sequences: []
tools: []
logprobs: false
recorded: '2026-05-11T15:54:50.137365Z'
replies: 1
max_replies: 5
status_code: 200
boundary: c66600
sampling:
  is_greedy: false
---
--c66600 user

What is 2+2? Reply with only the number.
--c66600 reply 1
Response: 4d5d48107e7b.json
Stream: 4d5d48107e7b.chunks.ndjson
Status: 200
Stop-Reason: stop
Input-Tokens: 20
Output-Tokens: 1

4
--c66600 reply 1 reasoning

The user asks "What is 2+2? Reply with only the number." So answer "4".
--c66600--
