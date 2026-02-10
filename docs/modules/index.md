# Module-Specific Documentation Index

## Core Modules

### `config.py`
Configuration management for InferenceGate. Handles loading and saving configuration from YAML files.

**Key classes:**
- `Config` - Pydantic model containing all configurable options
- `ConfigManager` - Manages loading/saving configuration files

**Default config location:** `$USERDIR/.InferenceGate/config.yaml`

### `cli.py`
Command-line interface for InferenceGate. See [cli.md](../cli.md) for full documentation.

**Entry point:** `main()` (registered as `inference-gate` console script)

### `inference_gate.py`
Main orchestrator class that coordinates all components.

**Key classes:**
- `InferenceGate` - Main application class

### `modes.py`
Defines operating modes for the proxy server.

**Key classes:**
- `Mode` - Enum with `RECORD_AND_REPLAY` and `REPLAY_ONLY` modes

## Submodules

### `inflow/`
Handles incoming HTTP requests.

- `server.py` - aiohttp-based HTTP server

### `outflow/`
Handles outgoing requests to upstream APIs.

- `client.py` - aiohttp-based HTTP client for forwarding requests

### `router/`
Routes requests between inflow, cache, and outflow.

- `router.py` - Request routing logic

### `recording/`
Handles caching and storage of inference data.

- `storage.py` - File-based cache storage