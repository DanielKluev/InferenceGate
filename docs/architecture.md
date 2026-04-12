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

## Outflow Component (src/inference_gate/outflow/)

Responsible for forwarding requests to the real AI model endpoint when a replay is not possible. It uses `aiohttp` to make asynchronous HTTP requests to the AI model and returns the responses back to the Router.

## Frontend Component (src/inference_gate/frontend/)

Optional Web UI for monitoring requests and responses and managing stored inferences.
