# Inference Gate Architecture Overview

## InferenceGate class (src/inference_gate/inference_gate.py)

Central class responsible for gluing together the various components of the system, including the HTTP server, request routing, replay database, and filtering mechanisms. It manages the lifecycle of the application, handles incoming requests, and coordinates the flow of data between components.

## Inflow Component (src/inference_gate/inflow/)

HTTP server that accepts incoming requests from clients (e.g., applications making inference calls). It is built using `aiohttp` and is designed to be fully asynchronous. The Inflow component is responsible for parsing incoming requests and passing to the router for further processing.

## Router Component (src/inference_gate/router/)

The Router component is responsible for determining how to handle incoming requests. It decides whether to replay an inference from the local storage or forward the request to the real AI model endpoint based on user-defined rules and configurations.

## Recording Component (src/inference_gate/recording/)

Handles the storage of captured inferences. Provides functionality to save incoming requests and their corresponding responses to a local database for future replay, and to retrieve stored inferences when requested by the Router.

## Outflow Component (src/inference_gate/outflow/)

Responsible for forwarding requests to the real AI model endpoint when a replay is not possible. It uses `aiohttp` to make asynchronous HTTP requests to the AI model and returns the responses back to the Router.

## Frontend Component (src/inference_gate/frontend/)

Optional Web UI for monitoring requests and responses and managing stored inferences.
