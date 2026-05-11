# Inference Gate Architecture Overview

## InferenceGate class (src/inference_gate/inference_gate.py)

Central class responsible for gluing together the various components of the system, including the HTTP server, request routing, replay database, and filtering mechanisms. It manages the lifecycle of the application, handles incoming requests, and coordinates the flow of data between components.

## Inflow Component (src/inference_gate/inflow/)

HTTP server that accepts incoming requests from clients (e.g., applications making inference calls). It is built using `aiohttp` and is designed to be fully asynchronous. The Inflow component is responsible for parsing incoming requests and passing to the router for further processing.

## Router Component (src/inference_gate/router/)

The Router component is responsible for determining how to handle incoming requests. It decides whether to replay an inference from the local storage or forward the request to the real AI model endpoint based on user-defined rules and configurations.

### Streaming-First Design

The Router implements a **streaming-first** strategy: on cache miss, requests sent to the upstream API are forced to use `stream: true` (with `stream_options.include_usage: true` to preserve token usage data). This only applies to **generation endpoints** that support streaming and have matching reassembly logic:
- `/v1/chat/completions`
- `/v1/responses` (and `/responses`)

Non-generation endpoints such as `/tokenize`, `/detokenize`, `/v1/models`, and `/v1/completions` are **not** forced to stream. They are proxied as-is and cached as standard JSON responses. This avoids corrupting responses from endpoints that don't return SSE streams.

When a response is served back to the client (from cache or live), the Inflow server **adapts** the format to match the client's original `stream` preference:
- If the client requested `stream: true`, SSE chunks are returned directly.
- If the client requested `stream: false`, streaming chunks are reassembled into a single JSON response.

This design ensures a single cassette per unique prompt can serve both streaming and non-streaming clients.

Models that do not support streaming can be exempted via the `non_streaming_models` configuration list.

## Recording Component (src/inference_gate/recording/)

Handles the storage of captured inferences. Provides functionality to save incoming requests and their corresponding responses to a local database for future replay, and to retrieve stored inferences when requested by the Router.

The `reassembly` submodule converts streaming SSE chunks into non-streaming JSON response bodies when a non-streaming client replays a streaming cassette. It supports the Chat Completions API, the Responses API, and the text Completions API (`/v1/completions`).

### Cassette tape format (v2)

Cassette tapes are human-readable MIME-style files stored in `cache/requests/*.tape` with YAML frontmatter. Version 2 of the format faithfully records two pieces of information that were lost in v1:

- **HTTP status code.** Non-200 upstream responses (e.g. vLLM 400 validation errors) are preserved so replay returns the original status instead of a spurious 200. The status is written both in the YAML frontmatter (`status_code: <int>`) and as a per-reply `Status: <int>` header. The index TSV gains a `status_code` column.
- **Reasoning / chain-of-thought content.** `message.reasoning_content` (and the shorter `message.reasoning` fallback used by some providers) is extracted from non-streaming bodies and from streaming SSE reasoning deltas, then written as a dedicated `reply N reasoning` MIME sub-section. Keeping reasoning separate from the primary reply body makes tapes easy to skim and diff. The index TSV gains a boolean `has_reasoning` column.

**Migration.** Tapes written under v1 are migrated in-place the first time a `CacheStorage` instance opens the directory: `tape_version` is bumped to 2 and an explicit `Status: 200` header is injected on every reply. Migration is idempotent — tapes already at v2 are left untouched. The index is rebuilt after migration so the new columns are populated.

## Outflow Component (src/inference_gate/outflow/)

Responsible for forwarding requests to the real AI model endpoint when a replay is not possible. It uses `aiohttp` to make asynchronous HTTP requests to the AI model and returns the responses back to the Router.

## Frontend Component (src/inference_gate/frontend/)

Optional Web UI for monitoring requests and responses and managing stored inferences.
