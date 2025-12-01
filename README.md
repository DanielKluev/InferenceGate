# InferenceReplay

Python library for efficient and convenient AI inference replay in testing, debugging and development, saving costs and time on repeated prompts.

## Installation

```bash
pip install inference-replay
```

## Features

- **Record Mode**: Transparently proxy requests to OpenAI API and cache all responses
- **Development Mode**: Replay from cache when available, record new requests
- **Replay Mode**: Only serve cached responses (for unit tests and CI)
- Supports streaming responses
- Preserves prompt, temperature, model, and other metadata
- CLI tools for easy management

## Usage

### CLI Commands

#### Record Mode
Proxy all requests to upstream OpenAI API and cache responses:

```bash
inference-replay record --port 8080 --upstream https://api.openai.com --api-key $OPENAI_API_KEY
```

#### Development Mode
Replay from cache if available, otherwise proxy and record:

```bash
inference-replay dev --port 8080 --cache-dir .inference_cache
```

#### Replay Mode (Unit Tests / Copilot)
Only replay cached responses, fail if not in cache:

```bash
inference-replay replay --port 8080 --cache-dir .inference_cache
```

### Cache Management

List cached entries:
```bash
inference-replay cache list
```

Show cache statistics:
```bash
inference-replay cache info
```

Clear cache:
```bash
inference-replay cache clear --yes
```

### Using with OpenAI Client

Point your OpenAI client to the proxy:

```python
from openai import OpenAI

client = OpenAI(
    api_key="any-key",  # Not needed in replay mode
    base_url="http://localhost:8080/v1"
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key (used in record/dev modes)

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--port, -p` | Server port | 8080 |
| `--host, -h` | Server host | 127.0.0.1 |
| `--cache-dir, -c` | Cache directory | .inference_cache |
| `--upstream, -u` | Upstream API URL | https://api.openai.com |
| `--api-key, -k` | OpenAI API key | $OPENAI_API_KEY |
| `--verbose, -v` | Enable verbose logging | false |

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check src/ tests/
```

## License

MIT License
