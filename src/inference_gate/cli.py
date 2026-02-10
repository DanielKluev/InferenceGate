"""
CLI for InferenceGate.

Provides command-line interface for starting the proxy server
in different modes and managing the inference cache.

Entry point: `main()` (registered as `inference-gate` console script).

Configuration is loaded from YAML files at `$USERDIR/.InferenceGate/config.yaml`
by default, or from a user-specified path via the `--config` option.
Command-line options override configuration file values.
"""

import asyncio
import logging

import aiohttp
import click

from inference_gate.config import Config, ConfigManager
from inference_gate.inference_gate import InferenceGate
from inference_gate.modes import Mode
from inference_gate.recording.storage import CacheStorage

# Global config manager and config, set up via pass_context
pass_config = click.make_pass_decorator(Config, ensure=True)


def setup_logging(verbose: bool) -> None:
    """
    Set up logging configuration.

    `verbose` enables DEBUG level logging, otherwise INFO level is used.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def load_config(ctx: click.Context, config_path: str | None) -> Config:
    """
    Load configuration and store it in context.

    Loads from `config_path` if specified, otherwise from default location.
    """
    manager = ConfigManager(config_path)
    config = manager.load()
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["config_manager"] = manager
    return config


def get_config(ctx: click.Context) -> Config:
    """
    Get config from context, or load default if not present.
    """
    if ctx.obj is None:
        ctx.obj = {}
    if "config" not in ctx.obj:
        manager = ConfigManager()
        ctx.obj["config"] = manager.load()
        ctx.obj["config_manager"] = manager
    return ctx.obj["config"]


@click.group()
@click.version_option(version="0.1.0", prog_name="inference-gate")
@click.option("--config", "-C", "config_path", type=click.Path(), default=None,
              help="Path to configuration file (default: $USERDIR/.InferenceGate/config.yaml)")
@click.pass_context
def main(ctx: click.Context, config_path: str | None) -> None:
    """
    InferenceGate - AI inference replay for testing, debugging and development.

    This tool provides an OpenAI-compatible API proxy that can record and replay
    AI inference calls, helping you save costs and time during development and testing.

    Configuration is loaded from $USERDIR/.InferenceGate/config.yaml by default.
    Use --config to specify a custom configuration file path.
    Command-line options override configuration file values.
    """
    load_config(ctx, config_path)


@main.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the server on (default: 8080)")
@click.option("--host", "-h", default=None, help="Host to bind the server to (default: 127.0.0.1)")
@click.option("--cache-dir", "-c", default=None, help="Directory to store cached responses (default: .inference_cache)")
@click.option("--upstream", "-u", default=None, help="Upstream OpenAI API base URL (default: https://api.openai.com)")
@click.option("--api-key", "-k", envvar="OPENAI_API_KEY", default=None, help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
@click.option("--verbose", "-v", is_flag=True, default=None, help="Enable verbose logging")
@click.pass_context
def start(ctx: click.Context, port: int | None, host: str | None, cache_dir: str | None, upstream: str | None, api_key: str | None,
          verbose: bool | None) -> None:
    """
    Start in record-and-replay mode (default).

    Replays cached inferences when available. On cache miss, forwards to upstream,
    records the response, and stores it for future replays.

    Command-line options override configuration file values.
    """
    config = get_config(ctx)

    # Apply config defaults, command-line overrides
    actual_port = port if port is not None else config.port
    actual_host = host if host is not None else config.host
    actual_cache_dir = cache_dir if cache_dir is not None else config.cache_dir
    actual_upstream = upstream if upstream is not None else config.upstream
    actual_api_key = api_key if api_key is not None else config.api_key
    actual_verbose = verbose if verbose is not None else config.verbose

    setup_logging(actual_verbose)

    gate = InferenceGate(host=actual_host, port=actual_port, mode=Mode.RECORD_AND_REPLAY, cache_dir=actual_cache_dir,
                         upstream_base_url=actual_upstream, api_key=actual_api_key)

    click.echo("Starting InferenceGate in record-and-replay mode")
    click.echo(f"  Proxy: http://{actual_host}:{actual_port}")
    click.echo(f"  Upstream: {actual_upstream}")
    click.echo(f"  Cache dir: {actual_cache_dir}")

    asyncio.run(gate.run_forever())


@main.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the server on (default: 8080)")
@click.option("--host", "-h", default=None, help="Host to bind the server to (default: 127.0.0.1)")
@click.option("--cache-dir", "-c", default=None, help="Directory to store cached responses (default: .inference_cache)")
@click.option("--verbose", "-v", is_flag=True, default=None, help="Enable verbose logging")
@click.pass_context
def replay(ctx: click.Context, port: int | None, host: str | None, cache_dir: str | None, verbose: bool | None) -> None:
    """
    Start in replay-only mode.

    Only returns cached responses. Returns an error if a matching inference
    is not found in the cache. Useful for unit tests and CI pipelines.

    Command-line options override configuration file values.
    """
    config = get_config(ctx)

    # Apply config defaults, command-line overrides
    actual_port = port if port is not None else config.port
    actual_host = host if host is not None else config.host
    actual_cache_dir = cache_dir if cache_dir is not None else config.cache_dir
    actual_verbose = verbose if verbose is not None else config.verbose

    setup_logging(actual_verbose)

    gate = InferenceGate(host=actual_host, port=actual_port, mode=Mode.REPLAY_ONLY, cache_dir=actual_cache_dir)

    click.echo("Starting InferenceGate in replay-only mode")
    click.echo(f"  Proxy: http://{actual_host}:{actual_port}")
    click.echo(f"  Cache dir: {actual_cache_dir}")

    asyncio.run(gate.run_forever())


@main.group()
@click.pass_context
def cache(ctx: click.Context) -> None:
    """Cache management commands."""


@cache.command(name="list")
@click.option("--cache-dir", "-c", default=None, help="Directory where cached responses are stored")
@click.pass_context
def cache_list(ctx: click.Context, cache_dir: str | None) -> None:
    """List all cached entries."""
    config = get_config(ctx)
    actual_cache_dir = cache_dir if cache_dir is not None else config.cache_dir

    storage = CacheStorage(actual_cache_dir)
    entries = storage.list_entries()

    if not entries:
        click.echo("No cached entries found.")
        return

    click.echo(f"Found {len(entries)} cached entries:\n")
    for cache_key, entry in entries:
        click.echo(f"  [{cache_key}]")
        click.echo(f"    Path: {entry.request.method} {entry.request.path}")
        if entry.model:
            click.echo(f"    Model: {entry.model}")
        if entry.temperature is not None:
            click.echo(f"    Temperature: {entry.temperature}")
        click.echo(f"    Streaming: {entry.response.is_streaming}")
        click.echo()


@cache.command(name="clear")
@click.option("--cache-dir", "-c", default=None, help="Directory where cached responses are stored")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def cache_clear(ctx: click.Context, cache_dir: str | None, yes: bool) -> None:
    """Clear all cached entries."""
    config = get_config(ctx)
    actual_cache_dir = cache_dir if cache_dir is not None else config.cache_dir

    storage = CacheStorage(actual_cache_dir)
    entries = storage.list_entries()

    if not entries:
        click.echo("No cached entries to clear.")
        return

    if not yes:
        if not click.confirm(f"Are you sure you want to clear {len(entries)} cached entries?"):
            click.echo("Aborted.")
            return

    count = storage.clear()
    click.echo(f"Cleared {count} cached entries.")


@cache.command(name="info")
@click.option("--cache-dir", "-c", default=None, help="Directory where cached responses are stored")
@click.pass_context
def cache_info(ctx: click.Context, cache_dir: str | None) -> None:
    """Show cache statistics."""
    config = get_config(ctx)
    actual_cache_dir = cache_dir if cache_dir is not None else config.cache_dir

    storage = CacheStorage(actual_cache_dir)
    entries = storage.list_entries()

    click.echo(f"Cache directory: {actual_cache_dir}")
    click.echo(f"Total entries: {len(entries)}")

    if entries:
        models: dict[str, int] = {}
        streaming_count = 0
        for _, entry in entries:
            if entry.model:
                models[entry.model] = models.get(entry.model, 0) + 1
            if entry.response.is_streaming:
                streaming_count += 1

        click.echo(f"Streaming responses: {streaming_count}")
        if models:
            click.echo("Models:")
            for model, count in sorted(models.items()):
                click.echo(f"  {model}: {count}")


# =============================================================================
# Test commands
# =============================================================================


async def _send_test_prompt(url: str, headers: dict[str, str], model: str, prompt: str, verbose: bool) -> tuple[bool, str]:
    """
    Send a test prompt to a Chat Completions endpoint.

    `url` is the full URL to the chat completions endpoint.
    `headers` is a dict of HTTP headers to include.
    `model` is the model name to request.
    `prompt` is the user message to send.

    Returns a tuple of (success, response_text).
    """
    setup_logging(verbose)
    log = logging.getLogger("TestCommand")

    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 50}

    log.debug("Sending test request to %s", url)
    log.debug("Using model: %s", model)

    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return False, f"HTTP {resp.status}: {error_text}"

                data = await resp.json()
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0].get("message", {}).get("content", "")
                    return True, content.strip()
                else:
                    return False, f"Unexpected response format: {data}"
        except aiohttp.ClientConnectorError as e:
            return False, f"Connection refused: {e}"
        except aiohttp.ClientError as e:
            return False, f"Connection error: {e}"
        except Exception as e:
            return False, f"Error: {e}"


def _print_test_result(success: bool, response: str, ctx: click.Context) -> None:
    """
    Print the result of a test prompt and exit with appropriate code.
    """
    if success:
        click.echo(f"\nResponse: {response}")
        if response.strip().rstrip(".").upper() == "OK":
            click.echo("\n[SUCCESS] Test passed!")
        else:
            click.echo("\n[WARNING] Received a response, but with unexpected content.")
            click.echo("This may indicate the endpoint is working but the model did not follow the test prompt exactly.")
    else:
        click.echo(f"\n[FAILED] {response}", err=True)
        ctx.exit(1)


@main.command(name="test-gate")
@click.option("--host", "-h", default=None, help="Host of the running InferenceGate instance")
@click.option("--port", "-p", default=None, type=int, help="Port of the running InferenceGate instance")
@click.option("--model", "-m", default=None, help="Model to use for the test (default: gpt-4o-mini)")
@click.option("--prompt", default=None, help="Custom prompt to send")
@click.option("--verbose", "-v", is_flag=True, default=None, help="Enable verbose logging")
@click.pass_context
def test_gate(ctx: click.Context, host: str | None, port: int | None, model: str | None, prompt: str | None,
              verbose: bool | None) -> None:
    """
    Test a running InferenceGate instance.

    Sends a test prompt to a running InferenceGate proxy to verify it is
    accepting and processing requests correctly. Uses the same host/port
    from the configuration, so you don't need to pass them explicitly.

    No API key is needed — the running instance already has it configured.

    Command-line options override configuration file values.
    """
    config = get_config(ctx)

    # Apply config defaults, command-line overrides
    actual_host = host if host is not None else config.host
    actual_port = port if port is not None else config.port
    actual_model = model if model is not None else config.test_model
    actual_prompt = prompt if prompt is not None else config.test_prompt
    actual_verbose = verbose if verbose is not None else config.verbose

    gate_url = f"http://{actual_host}:{actual_port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    click.echo(f"Testing InferenceGate at http://{actual_host}:{actual_port}...")
    click.echo(f"Using model: {actual_model}")

    success, response = asyncio.run(_send_test_prompt(gate_url, headers, actual_model, actual_prompt, actual_verbose))
    _print_test_result(success, response, ctx)


@main.command(name="test-upstream")
@click.option("--upstream", "-u", default=None, help="Upstream OpenAI API base URL")
@click.option("--api-key", "-k", envvar="OPENAI_API_KEY", default=None, help="OpenAI API key")
@click.option("--model", "-m", default=None, help="Model to use for the test (default: gpt-4o-mini)")
@click.option("--prompt", default=None, help="Custom prompt to send")
@click.option("--verbose", "-v", is_flag=True, default=None, help="Enable verbose logging")
@click.pass_context
def test_upstream(ctx: click.Context, upstream: str | None, api_key: str | None, model: str | None, prompt: str | None,
                  verbose: bool | None) -> None:
    """
    Test the connection to the upstream API directly.

    Sends a test prompt directly to the upstream API (bypassing InferenceGate)
    to verify that the API key and endpoint are working correctly.

    Command-line options override configuration file values.
    """
    config = get_config(ctx)

    # Apply config defaults, command-line overrides
    actual_upstream = upstream if upstream is not None else config.upstream
    actual_api_key = api_key if api_key is not None else config.api_key
    actual_model = model if model is not None else config.test_model
    actual_prompt = prompt if prompt is not None else config.test_prompt
    actual_verbose = verbose if verbose is not None else config.verbose

    if not actual_api_key:
        click.echo("Error: No API key provided. Set OPENAI_API_KEY environment variable, use --api-key, or configure in config file.",
                   err=True)
        ctx.exit(1)

    url = f"{actual_upstream.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {actual_api_key}"}

    click.echo(f"Testing upstream API at {actual_upstream}...")
    click.echo(f"Using model: {actual_model}")

    success, response = asyncio.run(_send_test_prompt(url, headers, actual_model, actual_prompt, actual_verbose))
    _print_test_result(success, response, ctx)


# =============================================================================
# Config command group
# =============================================================================


@main.group()
@click.pass_context
def config(ctx: click.Context) -> None:
    """Configuration management commands."""


@config.command(name="show")
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Show current configuration."""
    cfg = get_config(ctx)
    manager: ConfigManager = ctx.obj["config_manager"]

    click.echo(f"Configuration file: {manager.get_config_path()}")
    click.echo(f"File exists: {manager.exists()}")
    click.echo()
    click.echo("Current settings:")
    click.echo(f"  host: {cfg.host}")
    click.echo(f"  port: {cfg.port}")
    click.echo(f"  upstream: {cfg.upstream}")
    click.echo(f"  api_key: {'***' + cfg.api_key[-4:] if cfg.api_key and len(cfg.api_key) > 4 else '(not set)'}")
    click.echo(f"  cache_dir: {cfg.cache_dir}")
    click.echo(f"  verbose: {cfg.verbose}")
    click.echo(f"  test_model: {cfg.test_model}")
    click.echo(f"  test_prompt: {cfg.test_prompt[:50]}..." if len(cfg.test_prompt) > 50 else f"  test_prompt: {cfg.test_prompt}")


@config.command(name="init")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing configuration file")
@click.pass_context
def config_init(ctx: click.Context, force: bool) -> None:
    """Initialize a default configuration file."""
    manager: ConfigManager = ctx.obj["config_manager"]

    if manager.exists() and not force:
        click.echo(f"Configuration file already exists at {manager.get_config_path()}")
        click.echo("Use --force to overwrite.")
        return

    cfg = manager.create_default()
    click.echo(f"Created default configuration file at {manager.get_config_path()}")
    click.echo()
    click.echo("Edit this file to customize your settings.")
    click.echo("You can also set OPENAI_API_KEY environment variable for API key.")


@config.command(name="path")
@click.pass_context
def config_path(ctx: click.Context) -> None:
    """Show the path to the configuration file."""
    manager: ConfigManager = ctx.obj["config_manager"]
    click.echo(manager.get_config_path())


if __name__ == "__main__":
    main()
