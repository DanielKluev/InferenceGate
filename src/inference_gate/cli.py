"""
CLI for InferenceGate.

Provides command-line interface for starting the proxy server
in different modes and managing the inference cache.

Entry point: `main()` (registered as `inference-gate` console script).
"""

import asyncio
import logging

import click

from inference_gate.inference_gate import InferenceGate
from inference_gate.modes import Mode
from inference_gate.recording.storage import CacheStorage


def setup_logging(verbose: bool) -> None:
    """
    Set up logging configuration.

    `verbose` enables DEBUG level logging, otherwise INFO level is used.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


@click.group()
@click.version_option(version="0.1.0", prog_name="inference-gate")
def main() -> None:
    """
    InferenceGate - AI inference replay for testing, debugging and development.

    This tool provides an OpenAI-compatible API proxy that can record and replay
    AI inference calls, helping you save costs and time during development and testing.
    """


@main.command()
@click.option("--port", "-p", default=8080, type=int, help="Port to run the server on")
@click.option("--host", "-h", default="127.0.0.1", help="Host to bind the server to")
@click.option("--cache-dir", "-c", default=".inference_cache", help="Directory to store cached responses")
@click.option("--upstream", "-u", default="https://api.openai.com", help="Upstream OpenAI API base URL")
@click.option("--api-key", "-k", envvar="OPENAI_API_KEY", help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def start(port: int, host: str, cache_dir: str, upstream: str, api_key: str | None, verbose: bool) -> None:
    """
    Start in record-and-replay mode (default).

    Replays cached inferences when available. On cache miss, forwards to upstream,
    records the response, and stores it for future replays.
    """
    setup_logging(verbose)

    gate = InferenceGate(host=host, port=port, mode=Mode.RECORD_AND_REPLAY, cache_dir=cache_dir, upstream_base_url=upstream, api_key=api_key)

    click.echo("Starting InferenceGate in record-and-replay mode")
    click.echo(f"  Proxy: http://{host}:{port}")
    click.echo(f"  Upstream: {upstream}")
    click.echo(f"  Cache dir: {cache_dir}")

    asyncio.run(gate.run_forever())


@main.command()
@click.option("--port", "-p", default=8080, type=int, help="Port to run the server on")
@click.option("--host", "-h", default="127.0.0.1", help="Host to bind the server to")
@click.option("--cache-dir", "-c", default=".inference_cache", help="Directory to store cached responses")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def replay(port: int, host: str, cache_dir: str, verbose: bool) -> None:
    """
    Start in replay-only mode.

    Only returns cached responses. Returns an error if a matching inference
    is not found in the cache. Useful for unit tests and CI pipelines.
    """
    setup_logging(verbose)

    gate = InferenceGate(host=host, port=port, mode=Mode.REPLAY_ONLY, cache_dir=cache_dir)

    click.echo("Starting InferenceGate in replay-only mode")
    click.echo(f"  Proxy: http://{host}:{port}")
    click.echo(f"  Cache dir: {cache_dir}")

    asyncio.run(gate.run_forever())


@main.group()
def cache() -> None:
    """Cache management commands."""


@cache.command(name="list")
@click.option("--cache-dir", "-c", default=".inference_cache", help="Directory where cached responses are stored")
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
@click.option("--cache-dir", "-c", default=".inference_cache", help="Directory where cached responses are stored")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
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
@click.option("--cache-dir", "-c", default=".inference_cache", help="Directory where cached responses are stored")
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
