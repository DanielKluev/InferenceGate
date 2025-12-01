"""CLI for InferenceReplay."""

import logging

import click
import uvicorn

from inference_replay.modes import Mode
from inference_replay.server import ProxyConfig, create_app
from inference_replay.storage import CacheStorage


def setup_logging(verbose: bool) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@click.group()
@click.version_option(version="0.1.0", prog_name="inference-replay")
def main() -> None:
    """InferenceReplay - AI inference replay for testing, debugging and development.

    This tool provides an OpenAI-compatible API proxy that can record and replay
    AI inference calls, helping you save costs and time during development and testing.
    """


@main.command()
@click.option(
    "--port",
    "-p",
    default=8080,
    type=int,
    help="Port to run the server on",
)
@click.option(
    "--host",
    "-h",
    default="127.0.0.1",
    help="Host to bind the server to",
)
@click.option(
    "--cache-dir",
    "-c",
    default=".inference_cache",
    help="Directory to store cached responses",
)
@click.option(
    "--upstream",
    "-u",
    default="https://api.openai.com",
    help="Upstream OpenAI API base URL",
)
@click.option(
    "--api-key",
    "-k",
    envvar="OPENAI_API_KEY",
    help="OpenAI API key (defaults to OPENAI_API_KEY env var)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def record(
    port: int,
    host: str,
    cache_dir: str,
    upstream: str,
    api_key: str | None,
    verbose: bool,
) -> None:
    """Run in RECORD mode - proxy all requests and cache responses.

    This mode transparently proxies all requests to the upstream OpenAI API
    and caches responses for later replay.
    """
    setup_logging(verbose)

    config = ProxyConfig(
        mode=Mode.RECORD,
        cache_dir=cache_dir,
        upstream_base_url=upstream,
        api_key=api_key,
    )
    app = create_app(config)

    click.echo("Starting InferenceReplay in RECORD mode")
    click.echo(f"  Proxy: http://{host}:{port}")
    click.echo(f"  Upstream: {upstream}")
    click.echo(f"  Cache dir: {cache_dir}")

    uvicorn.run(app, host=host, port=port, log_level="info" if verbose else "warning")


@main.command()
@click.option(
    "--port",
    "-p",
    default=8080,
    type=int,
    help="Port to run the server on",
)
@click.option(
    "--host",
    "-h",
    default="127.0.0.1",
    help="Host to bind the server to",
)
@click.option(
    "--cache-dir",
    "-c",
    default=".inference_cache",
    help="Directory to store cached responses",
)
@click.option(
    "--upstream",
    "-u",
    default="https://api.openai.com",
    help="Upstream OpenAI API base URL",
)
@click.option(
    "--api-key",
    "-k",
    envvar="OPENAI_API_KEY",
    help="OpenAI API key (defaults to OPENAI_API_KEY env var)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def dev(
    port: int,
    host: str,
    cache_dir: str,
    upstream: str,
    api_key: str | None,
    verbose: bool,
) -> None:
    """Run in DEVELOPMENT mode - replay from cache or record if missing.

    This mode returns cached responses when available, and proxies to upstream
    when no cache exists (recording the response for future use).
    """
    setup_logging(verbose)

    config = ProxyConfig(
        mode=Mode.DEVELOPMENT,
        cache_dir=cache_dir,
        upstream_base_url=upstream,
        api_key=api_key,
    )
    app = create_app(config)

    click.echo("Starting InferenceReplay in DEVELOPMENT mode")
    click.echo(f"  Proxy: http://{host}:{port}")
    click.echo(f"  Upstream: {upstream}")
    click.echo(f"  Cache dir: {cache_dir}")

    uvicorn.run(app, host=host, port=port, log_level="info" if verbose else "warning")


@main.command()
@click.option(
    "--port",
    "-p",
    default=8080,
    type=int,
    help="Port to run the server on",
)
@click.option(
    "--host",
    "-h",
    default="127.0.0.1",
    help="Host to bind the server to",
)
@click.option(
    "--cache-dir",
    "-c",
    default=".inference_cache",
    help="Directory to store cached responses",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def replay(
    port: int,
    host: str,
    cache_dir: str,
    verbose: bool,
) -> None:
    """Run in REPLAY mode - only serve cached responses.

    This mode only returns cached responses and will error if a request
    is not found in the cache. Useful for unit tests and CI pipelines.
    """
    setup_logging(verbose)

    config = ProxyConfig(
        mode=Mode.REPLAY,
        cache_dir=cache_dir,
        upstream_base_url="",  # Not used in replay mode
        api_key=None,
    )
    app = create_app(config)

    click.echo("Starting InferenceReplay in REPLAY mode")
    click.echo(f"  Proxy: http://{host}:{port}")
    click.echo(f"  Cache dir: {cache_dir}")

    uvicorn.run(app, host=host, port=port, log_level="info" if verbose else "warning")


@main.group()
def cache() -> None:
    """Cache management commands."""


@cache.command(name="list")
@click.option(
    "--cache-dir",
    "-c",
    default=".inference_cache",
    help="Directory where cached responses are stored",
)
def cache_list(cache_dir: str) -> None:
    """List all cached entries."""
    storage = CacheStorage(cache_dir)
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
@click.option(
    "--cache-dir",
    "-c",
    default=".inference_cache",
    help="Directory where cached responses are stored",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Skip confirmation prompt",
)
def cache_clear(cache_dir: str, yes: bool) -> None:
    """Clear all cached entries."""
    storage = CacheStorage(cache_dir)
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
@click.option(
    "--cache-dir",
    "-c",
    default=".inference_cache",
    help="Directory where cached responses are stored",
)
def cache_info(cache_dir: str) -> None:
    """Show cache statistics."""
    storage = CacheStorage(cache_dir)
    entries = storage.list_entries()

    click.echo(f"Cache directory: {cache_dir}")
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


if __name__ == "__main__":
    main()
