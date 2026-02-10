# Command-Line Interface (CLI) Reference

InferenceGate provides a comprehensive CLI for managing the inference proxy server, cache, and configuration.

## Global Options

These options can be used with any command:

| Option | Description |
|--------|-------------|
| `--config, -C` | Path to configuration file (default: `$USERDIR/.InferenceGate/config.yaml`) |
| `--version` | Show version information |
| `--help` | Show help message |

## Server Commands

### `start` - Record-and-Replay Mode

Start the proxy server in record-and-replay mode. This is the default mode for development.

**Behavior:**
- Attempts to replay inferences from the local cache
- On cache miss, forwards the request to the upstream API
- Records the response and stores it for future replays

```bash
inference-gate start [OPTIONS]
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--port` | `-p` | Port to run the server on | 8080 |
| `--host` | `-h` | Host to bind the server to | 127.0.0.1 |
| `--cache-dir` | `-c` | Directory to store cached responses | .inference_cache |
| `--upstream` | `-u` | Upstream OpenAI API base URL | https://api.openai.com |
| `--api-key` | `-k` | OpenAI API key | $OPENAI_API_KEY |
| `--verbose` | `-v` | Enable verbose (DEBUG) logging | false |

**Example:**

```bash
# Basic usage with API key from environment
inference-gate start

# Custom port and verbose logging
inference-gate start --port 9000 --verbose

# Custom upstream endpoint (e.g., Azure OpenAI)
inference-gate start --upstream https://my-resource.openai.azure.com
```

### `replay` - Replay-Only Mode

Start the proxy server in replay-only mode. This mode is intended for unit tests and CI pipelines.

**Behavior:**
- Only attempts to replay inferences from the local cache
- Returns an error response if a matching inference is not found
- Does not forward any requests to the upstream API
- Does not require an API key

```bash
inference-gate replay [OPTIONS]
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--port` | `-p` | Port to run the server on | 8080 |
| `--host` | `-h` | Host to bind the server to | 127.0.0.1 |
| `--cache-dir` | `-c` | Directory to store cached responses | .inference_cache |
| `--verbose` | `-v` | Enable verbose (DEBUG) logging | false |

**Example:**

```bash
# Start replay server for testing
inference-gate replay --cache-dir ./test_cache

# Use in CI with specific cache directory
inference-gate replay --cache-dir ./fixtures/api_cache
```

## Test Commands

### `test-gate` - Test a Running InferenceGate Instance

Send a test prompt to a running InferenceGate proxy to verify it is accepting and processing requests correctly.

**Behavior:**
- Sends a test prompt to the InferenceGate proxy at the configured host/port
- By default, uses a prompt asking the model to reply with "OK."
- No API key is needed — the running instance already has it configured
- Uses the same host/port from the configuration file, so no extra options needed in most cases
- Reports success or failure with response details

```bash
inference-gate test-gate [OPTIONS]
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--host` | `-h` | Host of the running InferenceGate instance | 127.0.0.1 |
| `--port` | `-p` | Port of the running InferenceGate instance | 8080 |
| `--model` | `-m` | Model to use for the test | gpt-4o-mini |
| `--prompt` | | Custom prompt to send | (built-in test prompt) |
| `--verbose` | `-v` | Enable verbose (DEBUG) logging | false |

**Default Test Prompt:**

```
This is a test prompt. Reply with **ONLY** "OK." to confirm that everything is ok. DO NOT output anything else.
```

**Example:**

```bash
# Test the running instance (uses host/port from config)
inference-gate test-gate

# Test instance on a custom port
inference-gate test-gate --port 9000

# Test with a custom model
inference-gate test-gate --model gpt-4

# Test with a custom prompt
inference-gate test-gate --prompt "Say hello!"
```

**Exit Codes:**
- `0` - Test passed successfully
- `1` - Test failed (connection refused, proxy error, etc.)

### `test-upstream` - Test Upstream API Directly

Send a test prompt directly to the upstream API, bypassing InferenceGate, to verify that the API key and endpoint are working correctly.

**Behavior:**
- Sends a test prompt directly to the upstream API
- By default, uses a prompt asking the model to reply with "OK."
- Requires an API key (via `--api-key`, `OPENAI_API_KEY` env var, or config file)
- Reports success or failure with response details

```bash
inference-gate test-upstream [OPTIONS]
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--upstream` | `-u` | Upstream OpenAI API base URL | https://api.openai.com |
| `--api-key` | `-k` | OpenAI API key | $OPENAI_API_KEY |
| `--model` | `-m` | Model to use for the test | gpt-4o-mini |
| `--prompt` | | Custom prompt to send | (built-in test prompt) |
| `--verbose` | `-v` | Enable verbose (DEBUG) logging | false |

**Example:**

```bash
# Test with API key from environment
inference-gate test-upstream

# Test with specific API key
inference-gate test-upstream --api-key sk-...

# Test with custom model
inference-gate test-upstream --model gpt-4

# Test Azure OpenAI endpoint
inference-gate test-upstream --upstream https://my-resource.openai.azure.com
```

**Exit Codes:**
- `0` - Test passed successfully
- `1` - Test failed (connection error, authentication error, etc.)

## Cache Management Commands

### `cache list` - List Cached Entries

Display all cached inference entries with their metadata.

```bash
inference-gate cache list [OPTIONS]
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--cache-dir` | `-c` | Directory where cached responses are stored | .inference_cache |

**Example:**

```bash
inference-gate cache list
inference-gate cache list --cache-dir ./test_cache
```

**Output:**

```
Found 3 cached entries:

  [abc123def456]
    Path: POST /v1/chat/completions
    Model: gpt-4
    Temperature: 0.7
    Streaming: false

  [789ghi012jkl]
    Path: POST /v1/chat/completions
    Model: gpt-3.5-turbo
    Temperature: 0
    Streaming: true
```

### `cache info` - Show Cache Statistics

Display statistics about the cache contents.

```bash
inference-gate cache info [OPTIONS]
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--cache-dir` | `-c` | Directory where cached responses are stored | .inference_cache |

**Example:**

```bash
inference-gate cache info
```

**Output:**

```
Cache directory: .inference_cache
Total entries: 15
Streaming responses: 3
Models:
  gpt-3.5-turbo: 8
  gpt-4: 7
```

### `cache clear` - Clear All Cached Entries

Delete all cached entries from the cache directory.

```bash
inference-gate cache clear [OPTIONS]
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--cache-dir` | `-c` | Directory where cached responses are stored | .inference_cache |
| `--yes` | `-y` | Skip confirmation prompt | false |

**Example:**

```bash
# Interactive (asks for confirmation)
inference-gate cache clear

# Non-interactive (skips confirmation)
inference-gate cache clear --yes
```

## Configuration Commands

### `config show` - Show Current Configuration

Display the current configuration settings, including values from the config file, environment variables, and defaults.

```bash
inference-gate config show
```

**Example Output:**

```
Configuration file: C:\Users\User\.InferenceGate\config.yaml
File exists: true

Current settings:
  host: 127.0.0.1
  port: 8080
  upstream: https://api.openai.com
  api_key: ***abc1
  cache_dir: .inference_cache
  verbose: false
  test_model: gpt-4o-mini
  test_prompt: This is a test prompt. Reply with **ONLY** "OK."...
```

### `config init` - Initialize Configuration File

Create a default configuration file at the default location or the path specified with `--config`.

```bash
inference-gate config init [OPTIONS]
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--force` | `-f` | Overwrite existing configuration file | false |

**Example:**

```bash
# Create default config
inference-gate config init

# Overwrite existing config
inference-gate config init --force

# Initialize at custom location
inference-gate --config ./my-config.yaml config init
```

### `config path` - Show Configuration File Path

Display the path to the configuration file being used.

```bash
inference-gate config path
```

**Example Output:**

```
C:\Users\User\.InferenceGate\config.yaml
```

## Configuration File

InferenceGate uses a YAML configuration file to store default settings. Command-line options always override configuration file values.

### File Location

- **Windows**: `%USERPROFILE%\.InferenceGate\config.yaml`
- **macOS/Linux**: `~/.InferenceGate/config.yaml`

Use `--config` to specify a custom path:

```bash
inference-gate --config /path/to/config.yaml start
```

### Configuration Options

```yaml
# Server settings
host: "127.0.0.1"          # Host to bind the server to
port: 8080                  # Port to run the server on

# Upstream API settings
upstream: "https://api.openai.com"  # Upstream OpenAI API base URL
# Note: api_key is NOT stored in the config file for security
# Use the OPENAI_API_KEY environment variable instead

# Storage settings
cache_dir: ".inference_cache"  # Directory to store cached responses

# Logging settings
verbose: false              # Enable verbose (DEBUG) logging

# Test command settings
test_model: "gpt-4o-mini"   # Default model for the test command
test_prompt: "This is a test prompt..."  # Default prompt for the test command
```

### Configuration Priority

Settings are loaded in the following order (later overrides earlier):

1. **Built-in defaults** - Hardcoded default values
2. **Configuration file** - Values from YAML config file
3. **Environment variables** - `OPENAI_API_KEY` (for API key only)
4. **Command-line options** - Explicit options passed to the command

### Security Note

The `api_key` setting is intentionally NOT saved to the configuration file when using `config init`. Always use the `OPENAI_API_KEY` environment variable or the `--api-key` command-line option for API key management.

## Environment Variables

| Variable | Description | Used By |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | `start`, `test-upstream` |

## Examples

### Development Workflow

```bash
# 1. Initialize configuration
inference-gate config init

# 2. Test upstream API connection
export OPENAI_API_KEY=sk-...
inference-gate test-upstream

# 3. Start the proxy
inference-gate start

# 4. Test the running proxy (in another terminal)
inference-gate test-gate

# 5. Run your application pointing to http://localhost:8080/v1
```

### CI/CD Workflow

```bash
# Pre-recorded responses in fixtures directory
inference-gate replay --cache-dir ./fixtures/api_cache &

# Run tests against the replay server
pytest tests/
```

### Custom Configuration

```bash
# Use project-specific configuration
inference-gate --config ./.inference-gate.yaml start

# Initialize project-specific config
inference-gate --config ./.inference-gate.yaml config init
```