"""
Pytest plugin for InferenceGate integration.

Automatically registers CLI options, ini settings, and fixtures that
start an InferenceGate proxy server within the test session. Tests use
the ``inference_gate_url`` fixture to point their AI client at the proxy.

**No top-level imports of InferenceGate internals** — everything is
imported lazily inside fixtures/hooks so the main package stays
installable and usable without pytest present.

Usage in an external project::

    # pytest.ini / pyproject.toml
    [tool.pytest.ini_options]
    inferencegate_mode = "replay"
    inferencegate_cache_dir = "tests/cassettes"

    # conftest.py — nothing needed, plugin auto-discovers via entry point

    # test_example.py
    def test_ai_feature(inference_gate_url):
        from openai import OpenAI
        client = OpenAI(base_url=f"{inference_gate_url}/v1", api_key="unused")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello"}],
        )
        assert resp.choices[0].message.content
"""

import asyncio
import logging
import os
import threading
import time
from typing import Any, Generator

import pytest

log = logging.getLogger("InferenceGatePlugin")

# ---------------------------------------------------------------------------
# Option resolution
# ---------------------------------------------------------------------------

# Maps user-facing mode names to internal option keys
_OPTION_DEFS: dict[str, dict[str, Any]] = {
    "mode": {
        "cli": "--inferencegate-mode",
        "env": "INFERENCEGATE_MODE",
        "ini": "inferencegate_mode",
        "default": "replay",
    },
    "cache_dir": {
        "cli": "--inferencegate-cache-dir",
        "env": "INFERENCEGATE_CACHE_DIR",
        "ini": "inferencegate_cache_dir",
        "default": "tests/cassettes",
    },
    "config": {
        "cli": "--inferencegate-config",
        "env": "INFERENCEGATE_CONFIG",
        "ini": "inferencegate_config",
        "default": None,
    },
    "port": {
        "cli": "--inferencegate-port",
        "env": "INFERENCEGATE_PORT",
        "ini": "inferencegate_port",
        "default": "0",
    },
    "fuzzy_model": {
        "cli": "inferencegate_fuzzy_model",
        "env": "INFERENCEGATE_FUZZY_MODEL",
        "ini": "inferencegate_fuzzy_model",
        "default": None,
    },
    "fuzzy_sampling": {
        "cli": "--inferencegate-fuzzy-sampling",
        "env": "INFERENCEGATE_FUZZY_SAMPLING",
        "ini": "inferencegate_fuzzy_sampling",
        "default": "off",
    },
    "max_non_greedy_replies": {
        "cli": "--inferencegate-max-non-greedy-replies",
        "env": "INFERENCEGATE_MAX_NON_GREEDY_REPLIES",
        "ini": "inferencegate_max_non_greedy_replies",
        "default": "5",
    },
    "proxy": {
        "cli": "--inferencegate-proxy",
        "env": "INFERENCEGATE_PROXY",
        "ini": "inferencegate_proxy",
        "default": None,
    },
}


def _resolve_option(config: pytest.Config, name: str) -> str | bool | None:
    """
    Resolve an InferenceGate option using CLI > env var > ini > default.

    `name` is a key in ``_OPTION_DEFS`` (e.g. ``"mode"``, ``"cache_dir"``).
    Returns the resolved value, or None if no value is set and the
    default is None.  Boolean CLI flags (e.g. ``fuzzy_model_matching``)
    may return ``bool`` directly.
    """
    defn = _OPTION_DEFS[name]

    # 1. CLI flag (highest priority)
    cli_val = config.getoption(defn["cli"], default=None)
    if cli_val is not None:
        return cli_val

    # 2. Environment variable
    env_val = os.environ.get(defn["env"])
    if env_val:
        return env_val

    # 3. Ini option
    ini_val = config.getini(defn["ini"])
    if ini_val:
        return ini_val

    # 4. Default
    return defn["default"]


# ---------------------------------------------------------------------------
# Pytest hooks
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Register InferenceGate CLI arguments and ini options.
    """
    group = parser.getgroup("inferencegate", "InferenceGate inference replay proxy")

    group.addoption("--inferencegate-mode", choices=["replay", "record"], default=None,
                    help='Operating mode: "replay" (cached only, default) or "record" (cache + upstream).')
    group.addoption("--inferencegate-cache-dir", default=None, help="Directory for cached cassettes (default: tests/cassettes).")
    group.addoption("--inferencegate-config", default=None, help="Path to InferenceGate config.yaml (default: auto-detect).")
    group.addoption("--inferencegate-port", default=None, help="Proxy server port. 0 = OS-assigned (default: 0).")
    group.addoption("--inferencegate-fuzzy-model", action="store_true", dest="inferencegate_fuzzy_model", default=None,
                    help="Enable fuzzy model matching: on cache miss, reuse entries with the same prompt but a different model.")
    group.addoption("--no-inferencegate-fuzzy-model", action="store_false", dest="inferencegate_fuzzy_model", default=None,
                    help="Disable fuzzy model matching, overriding ini/env settings for a single test run.")
    group.addoption("--inferencegate-fuzzy-sampling", default=None, choices=["off", "soft", "aggressive"],
                    help="Sampling parameter fuzzy matching level: off, soft, or aggressive.")
    group.addoption("--inferencegate-max-non-greedy-replies", default=None, type=int,
                    help="Max replies to collect per non-greedy cassette before cycling (default: 5).")
    group.addoption("--inferencegate-proxy", default=None, help="HTTP proxy URL for upstream requests (e.g. http://127.0.0.1:8888/).")

    parser.addini("inferencegate_mode", default="", help="InferenceGate operating mode: replay or record.")
    parser.addini("inferencegate_cache_dir", default="", help="Directory for cached cassettes.")
    parser.addini("inferencegate_config", default="", help="Path to InferenceGate config.yaml.")
    parser.addini("inferencegate_port", default="", help="Proxy server port. 0 = OS-assigned.")
    parser.addini("inferencegate_fuzzy_model", default="", help="Enable fuzzy model matching (true/false).")
    parser.addini("inferencegate_fuzzy_sampling", default="", help="Sampling fuzzy matching level: off, soft, or aggressive.")
    parser.addini("inferencegate_max_non_greedy_replies", default="", help="Max replies per non-greedy cassette.")
    parser.addini("inferencegate_proxy", default="", help="HTTP proxy URL for upstream requests.")


def pytest_configure(config: pytest.Config) -> None:
    """
    Register custom markers.
    """
    config.addinivalue_line("markers", "requires_recording: skip this test in replay mode (cassette not yet recorded).")
    config.addinivalue_line(
        "markers", "inferencegate_strict: force exact cassette matching for this test — "
        "disables fuzzy_model and fuzzy_sampling regardless of session defaults. "
        "Use for tests that assert exact token-level behavior (logprobs, prompt logprobs, "
        "max_tokens-sensitive outputs) where a fuzzy cassette hit would silently mask drift.")
    config.addinivalue_line(
        "markers", "inferencegate(fuzzy_model=None, fuzzy_sampling=None): override cassette "
        "matching policy for a single test. Either/both kwargs may be set; omitted kwargs "
        "keep the session default. Supersedes inferencegate_strict when both are present.")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """
    Skip tests marked with ``@pytest.mark.requires_recording`` when running in replay mode.
    """
    mode = _resolve_option(config, "mode")
    if mode == "replay":
        skip_marker = pytest.mark.skip(reason="Skipped in replay mode — run with --inferencegate-mode record to record cassettes first.")
        for item in items:
            if item.get_closest_marker("requires_recording"):
                item.add_marker(skip_marker)


# ---------------------------------------------------------------------------
# Background server runner
# ---------------------------------------------------------------------------


class _ServerThread:
    """
    Runs an InferenceGate server in a daemon thread with its own event loop.

    The thread starts the server, signals readiness, and blocks until
    ``request_stop()`` is called.
    """

    def __init__(self, gate: Any) -> None:
        """
        Initialize the server thread wrapper.

        `gate` is an ``InferenceGate`` instance (not yet started).
        """
        self.gate = gate
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop_event: asyncio.Event | None = None

    def start(self) -> None:
        """
        Start the server in a background daemon thread.

        Blocks until the server is ready to accept connections (or raises
        on timeout).
        """
        self._thread = threading.Thread(target=self._run, daemon=True, name="inferencegate-server")
        self._thread.start()
        # Wait for server to signal readiness (health check passes)
        if not self._ready.wait(timeout=15):
            raise RuntimeError("InferenceGate server failed to start within 15 seconds")

    def _run(self) -> None:
        """
        Thread target: create event loop, start server, wait for stop signal.
        """
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._stop_event = asyncio.Event()

        async def _serve() -> None:
            await self.gate.start()
            self._ready.set()
            # Block until stop is requested
            await self._stop_event.wait()
            await self.gate.stop()

        self._loop.run_until_complete(_serve())
        self._loop.close()

    def request_stop(self) -> None:
        """
        Signal the server to shut down and wait for the thread to exit.
        """
        if self._loop is not None and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread is not None:
            self._thread.join(timeout=10)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _resolve_gate_kwargs(pytest_config: pytest.Config) -> dict[str, Any]:
    """
    Resolve all InferenceGate configuration from CLI flags, env vars, and ini options.

    Returns a kwargs dict ready to be passed to ``InferenceGate()``,
    with model_routes support for multi-upstream recording.

    External callers may override individual keys before constructing
    the ``InferenceGate`` instance.
    """
    from inference_gate.config import ConfigManager
    from inference_gate.modes import Mode
    from inference_gate.outflow.model_router import UpstreamConfig

    mode_str = _resolve_option(pytest_config, "mode")
    cache_dir = _resolve_option(pytest_config, "cache_dir")
    config_path = _resolve_option(pytest_config, "config")
    port_str = _resolve_option(pytest_config, "port") or "0"
    port = int(port_str)
    fuzzy_raw = _resolve_option(pytest_config, "fuzzy_model")
    if isinstance(fuzzy_raw, bool):
        fuzzy_model = fuzzy_raw
    elif fuzzy_raw is not None:
        fuzzy_model = str(fuzzy_raw).lower() in ("true", "1", "yes")
    else:
        fuzzy_model = False

    fuzzy_sampling = str(_resolve_option(pytest_config, "fuzzy_sampling") or "off")
    max_replies_raw = _resolve_option(pytest_config, "max_non_greedy_replies")
    max_non_greedy_replies = int(max_replies_raw) if max_replies_raw is not None else 5
    proxy = _resolve_option(pytest_config, "proxy") or None

    # Map user-facing mode names to internal Mode enum
    if mode_str == "record":
        mode = Mode.RECORD_AND_REPLAY
    else:
        mode = Mode.REPLAY_ONLY

    # Load upstream config from InferenceGate's config file (needed for record mode)
    upstream_base_url = "https://api.openai.com"
    api_key = None
    non_streaming_models: list[str] = []
    cfg_proxy: str | None = None
    model_routes: dict[str, UpstreamConfig] | None = None
    if mode == Mode.RECORD_AND_REPLAY:
        try:
            cfg_manager = ConfigManager(config_path=config_path)
            cfg = cfg_manager.load()
            upstream_base_url = cfg.upstream
            api_key = cfg.api_key
            non_streaming_models = cfg.non_streaming_models
            cfg_proxy = cfg.proxy
            # Parse model_routes from YAML dicts into UpstreamConfig objects
            if cfg.model_routes:
                model_routes = {}
                for pattern, route_cfg in cfg.model_routes.items():
                    model_routes[pattern] = UpstreamConfig(url=route_cfg["upstream"], api_key=route_cfg.get("api_key", api_key),
                                                           timeout=route_cfg.get("timeout", cfg.upstream_timeout),
                                                           proxy=route_cfg.get("proxy", cfg_proxy))
        except Exception:
            log.warning("Could not load InferenceGate config file; using defaults for record mode")

    # CLI/env/ini proxy overrides config file proxy
    actual_proxy = proxy if proxy is not None else cfg_proxy

    return {
        "host": "127.0.0.1",
        "port": port,
        "mode": mode,
        "cache_dir": cache_dir,
        "upstream_base_url": upstream_base_url,
        "api_key": api_key,
        "non_streaming_models": non_streaming_models,
        "fuzzy_model": fuzzy_model,
        "fuzzy_sampling": fuzzy_sampling,
        "max_non_greedy_replies": max_non_greedy_replies,
        "proxy": actual_proxy,
        "model_routes": model_routes,
    }


def _start_gate(kwargs: dict[str, Any]) -> tuple[Any, "_ServerThread"]:
    """
    Create, start, and health-check an InferenceGate instance.

    Returns a ``(gate, server_thread)`` tuple.
    """
    from inference_gate.inference_gate import InferenceGate

    gate = InferenceGate(**kwargs)
    server_thread = _ServerThread(gate)
    server_thread.start()
    _wait_for_health(gate.base_url, timeout=10)
    return gate, server_thread


@pytest.fixture(scope="session")
def inference_gate_factory(request: pytest.FixtureRequest) -> Generator[Any, None, None]:
    """
    Session-scoped factory fixture for creating InferenceGate proxy instances.

    Returns a callable ``create(**overrides)`` that spins up an InferenceGate
    server.  Configuration is resolved from CLI/env/ini options first, then
    ``overrides`` are merged on top (e.g. ``model_routes``, ``mode``).

    All servers created through the factory are stopped on session teardown.

    Usage in downstream ``conftest.py``::

        @pytest.fixture(scope="session")
        def inference_gate(inference_gate_factory):
            return inference_gate_factory()          # default settings

        @pytest.fixture(scope="session")
        def inference_gate(inference_gate_factory):
            routes = {"Gemma4:*": UpstreamConfig(url="http://...")}
            return inference_gate_factory(model_routes=routes)
    """
    base_kwargs = _resolve_gate_kwargs(request.config)
    created: list[tuple[Any, _ServerThread]] = []

    def _create(**overrides: Any) -> Any:
        """
        Create and start an InferenceGate instance.

        Keyword arguments override the session-level defaults resolved from
        CLI/env/ini options.  Returns the ``InferenceGate`` instance.
        """
        kwargs = {**base_kwargs, **overrides}
        gate, server_thread = _start_gate(kwargs)
        created.append((gate, server_thread))
        return gate

    yield _create

    # Teardown: stop all servers created during the session
    for _, server_thread in reversed(created):
        server_thread.request_stop()


@pytest.fixture(scope="session")
def inference_gate(inference_gate_factory: Any) -> Any:
    """
    Session-scoped fixture that starts an InferenceGate proxy server.

    Resolves all configuration from CLI flags, environment variables, and
    ini options, then launches the server in a background thread. Yields
    the ``InferenceGate`` instance. The server is stopped on teardown.

    In ``"replay"`` mode (default), no upstream connection is made — all
    responses come from cached cassettes. In ``"record"`` mode, cache misses
    are forwarded to the real AI endpoint and recorded.

    Projects that need to customise the Gate (e.g. inject ``model_routes``
    from a Glue config) should override this fixture in their own
    ``conftest.py`` and call ``inference_gate_factory(**overrides)`` directly.
    """
    return inference_gate_factory()


@pytest.fixture(scope="session")
def inference_gate_url(inference_gate: Any) -> str:
    """
    Session-scoped fixture returning the base URL of the InferenceGate proxy.

    Returns a string like ``"http://127.0.0.1:54321"``. Point your AI
    client's ``base_url`` at ``f"{inference_gate_url}/v1"`` to route
    through the proxy.
    """
    return inference_gate.base_url


@pytest.fixture(autouse=True)
def _inferencegate_match_policy(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """
    Apply per-test cassette matching overrides declared via the
    ``inferencegate_strict`` / ``inferencegate`` markers.

    - ``@pytest.mark.inferencegate_strict`` forces ``fuzzy_model=False`` and
      ``fuzzy_sampling="off"`` for the duration of the test, so only exact
      ``content_hash`` cassette hits are served.  Intended for tests that
      assert exact token-level behavior (sampled logprobs, prompt logprobs,
      ``max_tokens``-sensitive outputs) where a fuzzy match would silently
      mask drift.
    - ``@pytest.mark.inferencegate(fuzzy_model=..., fuzzy_sampling=...)``
      overrides either/both individual flags.  When both markers are
      present, kwargs from ``inferencegate`` take precedence.

    Only activates (and pulls in the session-scoped ``inference_gate``
    fixture) when at least one of the markers is applied to the test —
    unmarked tests incur no overhead.
    """
    strict_marker = request.node.get_closest_marker("inferencegate_strict")
    custom_marker = request.node.get_closest_marker("inferencegate")
    if strict_marker is None and custom_marker is None:
        yield
        return

    # Lazily resolve inference_gate so tests that don't otherwise use it
    # still pay nothing for simply importing the plugin.
    gate = request.getfixturevalue("inference_gate")
    router = gate.router
    if router is None:
        # Gate was yielded but not started — nothing to override.
        yield
        return

    # Build override values: strict baseline, then apply custom kwargs.
    overrides: dict[str, Any] = {}
    if strict_marker is not None:
        overrides["fuzzy_model"] = False
        overrides["fuzzy_sampling"] = "off"
    if custom_marker is not None:
        for key in ("fuzzy_model", "fuzzy_sampling"):
            if key in custom_marker.kwargs:
                overrides[key] = custom_marker.kwargs[key]

    # Snapshot + apply.
    saved = {key: getattr(router, key) for key in overrides}
    for key, value in overrides.items():
        setattr(router, key, value)
    log.debug("inferencegate match-policy override for %s: %s (restored on teardown)", request.node.nodeid, overrides)

    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(router, key, value)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_health(base_url: str, timeout: float = 10) -> None:
    """
    Poll the proxy's ``/health`` endpoint until it responds or timeout.

    Uses plain ``http.client`` to avoid importing aiohttp in the main thread.
    """
    import http.client
    import urllib.parse

    parsed = urllib.parse.urlparse(base_url)
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=2)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            if resp.status == 200:
                conn.close()
                return
            conn.close()
        except (ConnectionRefusedError, OSError):
            pass
        time.sleep(0.1)

    raise RuntimeError(f"InferenceGate server at {base_url} did not become healthy within {timeout}s")
