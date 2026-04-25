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

# InferenceGlue is an optional peer dependency; when installed, the
# match-policy fixture forwards Require-* / Metadata-* headers via
# Glue's request-context ContextVar so they reach the Gate over HTTP
# (works in both in-process and subprocess/xdist modes).  When absent
# (e.g. Gate self-tests), the fixture falls back to mutating the
# in-process router directly.
try:
    from inference_glue.request_context import update_headers as _glue_update_headers  # type: ignore[import-not-found]
    from inference_glue.request_context import reset_headers as _glue_reset_headers  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised only when Glue is absent
    _glue_update_headers = None  # type: ignore[assignment]
    _glue_reset_headers = None  # type: ignore[assignment]

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
    Register custom markers, then \u2014 on the xdist controller \u2014 pre-spawn the
    shared subprocess Gate so workers inherit ``INFERENCEGATE_URL``.
    """
    config.addinivalue_line("markers", "requires_recording: skip this test in replay mode (cassette not yet recorded).")
    config.addinivalue_line(
        "markers", "inferencegate_strict: force exact cassette matching for this test \u2014 "
        "disables fuzzy_model and fuzzy_sampling regardless of session defaults. "
        "Use for tests that assert exact token-level behavior (logprobs, prompt logprobs, "
        "max_tokens-sensitive outputs) where a fuzzy cassette hit would silently mask drift.")
    config.addinivalue_line(
        "markers", "inferencegate(fuzzy_model=None, fuzzy_sampling=None): override cassette "
        "matching policy for a single test. Either/both kwargs may be set; omitted kwargs "
        "keep the session default. Supersedes inferencegate_strict when both are present.")

    # Spawn the shared subprocess Gate as early as possible on the xdist
    # controller so that workers inherit INFERENCEGATE_URL from the
    # environment.  No-op on workers and in non-xdist sessions.
    _maybe_spawn_for_xdist(config)


def pytest_unconfigure(config: pytest.Config) -> None:
    """
    Tear down the subprocess Gate on session end (controller only).
    """
    _maybe_stop_xdist_subprocess(config)


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


class _SubprocessServer:
    """
    Runs an InferenceGate server in a child process via ``inference-gate serve``.

    Required for ``pytest-xdist`` because xdist workers are separate processes
    that cannot share an in-process server.  The controller spawns one
    ``_SubprocessServer`` per session, exposes its URL via the
    ``INFERENCEGATE_URL`` environment variable, and the subprocess inherits a
    ``--parent-pid`` watchdog so it dies cleanly when the test session ends.

    Communication contract:

    1. Child prints ``INFERENCEGATE_URL=http://host:port\\n`` on stdout once
       it is listening.
    2. Parent reads that line synchronously to obtain the actual port.
    3. On teardown the parent terminates the child; the watchdog also exits
       the child if the parent dies first.
    """

    def __init__(self, kwargs: dict[str, Any]) -> None:
        """
        Capture the kwargs that should be forwarded to ``inference-gate serve``.

        ``kwargs`` is the dict produced by :func:`_resolve_gate_kwargs`; only
        the subset that ``serve`` accepts is mapped to CLI flags below.
        """
        self.kwargs = kwargs
        self._proc: Any = None
        self.base_url: str | None = None

    def start(self) -> str:
        """
        Spawn the subprocess and read the URL announcement.

        Returns the announced base URL (e.g. ``"http://127.0.0.1:54321"``).
        Raises ``RuntimeError`` if the child fails to announce within the
        timeout window.
        """
        import os
        import subprocess
        import sys

        cmd = self._build_cmd()
        log.debug("Spawning InferenceGate subprocess: %s", cmd)
        # Inherit stderr so child logs surface in pytest -s output; keep stdout piped for URL parsing.
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, bufsize=1, env=dict(os.environ))

        deadline = time.monotonic() + 30.0
        url: str | None = None
        while time.monotonic() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                # Pipe closed before announcement: the subprocess died early.
                rc = self._proc.poll()
                raise RuntimeError(f"InferenceGate subprocess exited before announcing URL (rc={rc})")
            line = line.strip()
            if line.startswith("INFERENCEGATE_URL="):
                url = line[len("INFERENCEGATE_URL="):]
                break
        if url is None:
            self.request_stop()
            raise RuntimeError("InferenceGate subprocess did not announce URL within 30s")

        self.base_url = url
        # Drain remaining stdout to stderr in a daemon thread so child logs are not lost.
        self._start_stdout_drain()
        return url

    def _build_cmd(self) -> list[str]:
        """
        Translate the resolved gate kwargs into a ``inference-gate serve`` command line.

        Subprocess mode only supports the kwargs ``serve`` knows about; richer
        config (model_routes) must be provided via the ``--config`` flag /
        config file rather than per-flag.
        """
        import os
        import sys
        from inference_gate.modes import Mode as _Mode

        mode_value = self.kwargs.get("mode")
        if isinstance(mode_value, _Mode):
            mode_str = "record" if mode_value == _Mode.RECORD_AND_REPLAY else "replay"
        else:
            mode_str = "replay"

        cmd = [
            sys.executable, "-m", "inference_gate.cli", "serve",
            "--mode", mode_str,
            "--host", str(self.kwargs.get("host", "127.0.0.1")),
            "--port", str(self.kwargs.get("port", 0) or 0),
            "--print-url",
            "--parent-pid", str(os.getpid()),
        ]
        cache_dir = self.kwargs.get("cache_dir")
        if cache_dir:
            cmd.extend(["--cache-dir", str(cache_dir)])
        if self.kwargs.get("fuzzy_model"):
            cmd.append("--fuzzy-model")
        else:
            cmd.append("--no-fuzzy-model")
        fs = self.kwargs.get("fuzzy_sampling")
        if fs:
            cmd.extend(["--fuzzy-sampling", str(fs)])
        max_replies = self.kwargs.get("max_non_greedy_replies")
        if max_replies is not None:
            cmd.extend(["--max-non-greedy-replies", str(max_replies)])
        proxy = self.kwargs.get("proxy")
        if proxy:
            cmd.extend(["--proxy", str(proxy)])
        upstream = self.kwargs.get("upstream_base_url")
        if upstream and mode_str == "record":
            cmd.extend(["--upstream", str(upstream)])
        api_key = self.kwargs.get("api_key")
        if api_key and mode_str == "record":
            cmd.extend(["--api-key", str(api_key)])
        return cmd

    def _start_stdout_drain(self) -> None:
        """
        Forward subsequent child stdout to parent stderr so logs are not lost.
        """
        import sys

        def _drain() -> None:
            try:
                assert self._proc is not None and self._proc.stdout is not None
                for line in self._proc.stdout:
                    sys.stderr.write(line)
            except Exception:  # pylint: disable=broad-except
                pass

        t = threading.Thread(target=_drain, daemon=True, name="inferencegate-subproc-drain")
        t.start()

    def request_stop(self) -> None:
        """
        Terminate the subprocess and reap it.
        """
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except Exception:  # pylint: disable=broad-except
                    self._proc.kill()
                    self._proc.wait(timeout=5)
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("Error stopping InferenceGate subprocess: %s", exc)
        finally:
            self._proc = None


def _is_xdist_worker(config: pytest.Config) -> bool:
    """
    Return True when running inside a pytest-xdist worker process.

    Workers receive a ``workerinput`` dict via ``pytest_configure``; the
    controller does not.  This is the canonical detection.
    """
    return hasattr(config, "workerinput")


def _is_xdist_active(config: pytest.Config) -> bool:
    """
    Return True when xdist is enabled for this session (controller side).

    Detected by either the resolved ``-n`` option (``numprocesses``) or by an
    explicit ``--dist`` distribution mode being chosen.  Workers will already
    be in :func:`_is_xdist_worker`.
    """
    if _is_xdist_worker(config):
        return True
    numproc = config.getoption("numprocesses", default=None)
    if numproc not in (None, 0, "0"):
        return True
    dist = config.getoption("dist", default=None)
    if dist and dist not in ("no", None):
        return True
    return False


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


class _GateHandle:
    """
    Lightweight surface that ``inference_gate``-consumers can rely on.

    In single-process mode this wraps the live :class:`InferenceGate`, so
    ``handle.gate`` exposes the in-process object (with its ``router``).
    In subprocess mode (xdist) only ``base_url`` is available; ``gate``,
    ``router`` and ``storage`` are ``None`` and consumers that need to
    reach into the running gate must do so via the ``/gate/*`` HTTP API.

    The handle is intentionally duck-typed against the historical contract
    (``.base_url`` / ``.router``) so existing fixtures that pulled
    ``inference_gate.router`` keep working when running in-process.
    """

    def __init__(self, base_url: str, gate: Any = None) -> None:
        """
        ``base_url`` is the listening URL.  ``gate`` is the in-process
        :class:`InferenceGate` when available, else ``None``.
        """
        self.base_url = base_url
        self.gate = gate

    @property
    def router(self) -> Any:
        """
        Return the in-process Router when available; ``None`` under xdist.

        Tests that rely on direct router mutation (e.g. the
        ``inferencegate_strict`` marker) should fall back to header-based
        overrides when this property is ``None``.
        """
        return self.gate.router if self.gate is not None else None

    @property
    def storage(self) -> Any:
        """
        Return the in-process CacheStorage when available; ``None`` under xdist.
        """
        return self.gate._storage if self.gate is not None else None  # pylint: disable=protected-access

    def __getattr__(self, name: str) -> Any:
        """
        Forward unknown attribute lookups to the wrapped :class:`InferenceGate`.

        Keeps backward compatibility for callers that previously accessed
        ``inference_gate.actual_port``, ``.mode``, ``.cache_dir`` etc.  In
        subprocess mode the wrapped gate is ``None``, so attribute access for
        anything beyond ``base_url``/``router``/``storage`` raises a clear
        ``AttributeError``.
        """
        gate = object.__getattribute__(self, "gate")
        if gate is not None and hasattr(gate, name):
            return getattr(gate, name)
        raise AttributeError(f"_GateHandle has no attribute {name!r}; in subprocess mode only "
                             "'base_url' is reliably available — use Gate's /gate/* HTTP API for live introspection.")


def _spawn_subprocess_gate(config: pytest.Config) -> _SubprocessServer:
    """
    Spawn the session-wide subprocess Gate from the xdist controller.

    Sets ``INFERENCEGATE_URL`` on the controller's environment so xdist
    workers inherit it when they fork.  Returns the running subprocess
    handle so the caller can stash it for teardown.
    """
    import os

    kwargs = _resolve_gate_kwargs(config)
    server = _SubprocessServer(kwargs)
    url = server.start()
    os.environ["INFERENCEGATE_URL"] = url
    log.info("InferenceGate subprocess listening at %s (workers will inherit INFERENCEGATE_URL)", url)
    return server


def _maybe_spawn_for_xdist(config: pytest.Config) -> None:
    """
    Controller-side hook invoked from :func:`pytest_configure` to pre-spawn the
    subprocess Gate when xdist is active.

    No-ops on workers (they consume the inherited ``INFERENCEGATE_URL``) and
    in non-xdist runs (the in-thread path is started lazily by fixture).
    """
    if _is_xdist_worker(config):
        return
    if not _is_xdist_active(config):
        return
    if os.environ.get("INFERENCEGATE_URL"):
        # Already spawned by an outer harness (e.g. CI) — reuse it.
        log.info("Reusing externally-provided INFERENCEGATE_URL=%s", os.environ["INFERENCEGATE_URL"])
        return
    server = _spawn_subprocess_gate(config)
    config._inferencegate_subprocess = server  # type: ignore[attr-defined]


def _maybe_stop_xdist_subprocess(config: pytest.Config) -> None:
    """
    Controller-side counterpart to :func:`_maybe_spawn_for_xdist`.

    Tears down the subprocess Gate at session end.  Safe to call in any
    process; workers and in-thread sessions hold no subprocess handle.
    """
    server: _SubprocessServer | None = getattr(config, "_inferencegate_subprocess", None)
    if server is not None:
        server.request_stop()
        config._inferencegate_subprocess = None  # type: ignore[attr-defined]


@pytest.fixture(scope="session")
def inference_gate_factory(request: pytest.FixtureRequest) -> Generator[Any, None, None]:
    """
    Session-scoped factory fixture for creating InferenceGate proxy handles.

    Returns a callable ``create(**overrides)`` that returns a :class:`_GateHandle`.

    Resolution strategy:

    - **xdist worker / pre-existing INFERENCEGATE_URL**: returns a thin handle
      pointing at the inherited URL; ``overrides`` are ignored because the
      subprocess is shared session-wide.  A warning is emitted if overrides
      were requested.
    - **xdist controller**: the subprocess was already spawned in
      :func:`pytest_configure`; the factory just wraps that URL.
    - **single-process mode (no xdist)**: legacy in-thread path — each
      ``create()`` call spins up its own :class:`_ServerThread`, allowing
      multiple co-existing gates with different model_routes.

    All servers created through the factory are stopped on session teardown.
    """
    base_kwargs = _resolve_gate_kwargs(request.config)
    created_threads: list[tuple[Any, _ServerThread]] = []

    # If the controller already exported INFERENCEGATE_URL (xdist or external harness)
    # we run in subprocess-shared mode: every create() returns the same handle.
    shared_url = os.environ.get("INFERENCEGATE_URL")
    is_worker = _is_xdist_worker(request.config)

    def _create(**overrides: Any) -> _GateHandle:
        """
        Create or look up the InferenceGate handle for this session.

        In subprocess-shared mode, ``overrides`` other than ``mode`` are
        ignored with a warning because the subprocess is fixed at spawn time.
        """
        if shared_url is not None or is_worker:
            if overrides:
                log.warning("Ignoring inference_gate_factory overrides %s in subprocess-shared mode", sorted(overrides))
            url = shared_url or os.environ.get("INFERENCEGATE_URL")
            if url is None:
                raise RuntimeError(
                    "INFERENCEGATE_URL not set: xdist worker has no subprocess Gate to consume. "
                    "Did the controller spawn the subprocess in pytest_configure?")
            return _GateHandle(base_url=url, gate=None)
        # Legacy in-thread path
        kwargs = {**base_kwargs, **overrides}
        gate, server_thread = _start_gate(kwargs)
        created_threads.append((gate, server_thread))
        return _GateHandle(base_url=gate.base_url, gate=gate)

    yield _create

    # Teardown: stop all in-thread servers created during the session.
    # Subprocess teardown is handled by pytest_unconfigure on the controller.
    for _, server_thread in reversed(created_threads):
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
    Apply per-test cassette matching overrides and auto-attach test-identity
    metadata.

    Two layers run in tandem:

    1. **Marker translation** \u2014 ``@pytest.mark.inferencegate_strict`` and
       ``@pytest.mark.inferencegate(fuzzy_model=..., fuzzy_sampling=...)`` are
       translated into ``X-InferenceGate-Require-*`` headers pushed onto
       :mod:`inference_glue.request_context` (when InferenceGlue is importable).
       This works uniformly in single-process and subprocess/xdist modes
       because the headers travel with each HTTP request to the Gate.
       Strict baseline (``Require-Exact: true``) is applied first; the
       ``inferencegate`` marker's kwargs override individual flags.
    2. **Test-identity metadata** \u2014 every test transparently pushes
       ``X-InferenceGate-Metadata-Test-NodeID`` (the pytest nodeid) and
       ``X-InferenceGate-Metadata-Worker-ID`` (the xdist worker name, or
       ``"master"``) so recorded tapes carry provenance for free.

    Legacy in-process behaviour: when the gate is in-process *and* InferenceGlue
    is not importable, the old router-mutation path still runs as a fallback so
    Gate's own self-tests \u2014 which do not depend on Glue \u2014 keep working.
    """
    strict_marker = request.node.get_closest_marker("inferencegate_strict")
    custom_marker = request.node.get_closest_marker("inferencegate")

    # Build header overrides from markers (regardless of transport mode).
    header_overrides: dict[str, str] = {}
    if strict_marker is not None:
        header_overrides["X-InferenceGate-Require-Exact"] = "true"
    if custom_marker is not None:
        if "fuzzy_model" in custom_marker.kwargs:
            value = custom_marker.kwargs["fuzzy_model"]
            header_overrides["X-InferenceGate-Require-Fuzzy-Model"] = "on" if value else "off"
        if "fuzzy_sampling" in custom_marker.kwargs:
            header_overrides["X-InferenceGate-Require-Fuzzy-Sampling"] = str(custom_marker.kwargs["fuzzy_sampling"])
    # Auto test-identity metadata.
    header_overrides["X-InferenceGate-Metadata-Test-NodeID"] = request.node.nodeid
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")
    header_overrides["X-InferenceGate-Metadata-Worker-ID"] = worker_id

    # Try the Glue ContextVar path first \u2014 works for both in-process and
    # subprocess/xdist modes.  If Glue is not installed (e.g. Gate self-tests),
    # fall back to legacy router mutation when possible.
    glue_token = None
    if _glue_update_headers is not None:
        glue_token = _glue_update_headers(header_overrides)

    # Legacy in-process fallback for marker-only overrides when Glue is absent.
    router_saved: dict[str, Any] | None = None
    router = None
    if _glue_update_headers is None and (strict_marker is not None or custom_marker is not None):
        gate = request.getfixturevalue("inference_gate")
        router = gate.router
        if router is not None:
            overrides: dict[str, Any] = {}
            if strict_marker is not None:
                overrides["fuzzy_model"] = False
                overrides["fuzzy_sampling"] = "off"
            if custom_marker is not None:
                for key in ("fuzzy_model", "fuzzy_sampling"):
                    if key in custom_marker.kwargs:
                        overrides[key] = custom_marker.kwargs[key]
            router_saved = {key: getattr(router, key) for key in overrides}
            for key, value in overrides.items():
                setattr(router, key, value)
            log.debug("inferencegate router override for %s: %s (Glue unavailable, legacy path)", request.node.nodeid, overrides)
        else:
            log.warning("inferencegate marker on %s has no effect: Glue not installed and no in-process router available.",
                        request.node.nodeid)

    try:
        yield
    finally:
        if glue_token is not None and _glue_reset_headers is not None:
            _glue_reset_headers(glue_token)
        if router_saved is not None and router is not None:
            for key, value in router_saved.items():
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
