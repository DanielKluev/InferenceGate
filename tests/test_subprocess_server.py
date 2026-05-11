"""
Tests for the subprocess server path used by pytest-xdist.

Validates the ``inference-gate serve`` CLI subcommand and the plugin's
``_SubprocessServer`` wrapper end-to-end:

- ``serve --print-url`` announces the listening URL on stdout.
- The subprocess responds to ``/gate/health``.
- ``--parent-pid`` watchdog terminates the subprocess when the parent dies.
- xdist worker detection forces consumption of ``INFERENCEGATE_URL`` rather
  than spawning a second server.
"""

import http.client
import os
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

import pytest


CASSETTES_DIR = str(Path(__file__).parent / "cassettes")


def _wait_for_url_announcement(proc: subprocess.Popen, timeout: float = 30.0) -> str:
    """
    Read the child's stdout until the ``INFERENCEGATE_URL=...`` line appears.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = proc.stdout.readline()  # type: ignore[union-attr]
        if not line:
            rc = proc.poll()
            raise RuntimeError(f"Subprocess exited before announcing URL (rc={rc})")
        line = line.strip()
        if line.startswith("INFERENCEGATE_URL="):
            return line[len("INFERENCEGATE_URL="):]
    raise RuntimeError("Subprocess did not announce URL within timeout")


def _http_get_status(url: str, path: str) -> int:
    """
    Perform a synchronous GET against ``url + path`` and return status code.
    """
    parsed = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
    try:
        conn.request("GET", path)
        return conn.getresponse().status
    finally:
        conn.close()


class TestServeCli:
    """Tests for the ``inference-gate serve`` subcommand."""

    def test_serve_prints_url_and_responds(self, tmp_path):
        """``serve --print-url`` announces URL; ``/gate/health`` returns 200."""
        cmd = [
            sys.executable, "-m", "inference_gate.cli", "serve",
            "--mode", "replay",
            "--host", "127.0.0.1",
            "--port", "0",
            "--cache-dir", str(tmp_path / "cache"),
            "--print-url",
            "--parent-pid", str(os.getpid()),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        try:
            url = _wait_for_url_announcement(proc, timeout=30.0)
            assert url.startswith("http://127.0.0.1:")
            assert int(url.rsplit(":", 1)[-1]) > 0
            assert _http_get_status(url, "/gate/health") == 200
            assert _http_get_status(url, "/health") == 200  # legacy alias
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    def test_parent_pid_watchdog_terminates_orphan(self, tmp_path):
        """When the supplied parent PID is already dead, the subprocess exits promptly."""
        # Spawn an intermediate parent that immediately exits, then ask serve
        # to watch its PID.  The subprocess should exit on the next watchdog tick.
        ghost = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"])
        ghost.wait(timeout=5)
        ghost_pid = ghost.pid
        # Note: PID may be reused before the test runs.  This is a best-effort
        # test; on a quiet machine it is reliable enough to validate the path.

        cmd = [
            sys.executable, "-m", "inference_gate.cli", "serve",
            "--mode", "replay",
            "--port", "0",
            "--cache-dir", str(tmp_path / "cache"),
            "--print-url",
            "--parent-pid", str(ghost_pid),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        try:
            # The child still announces URL before the watchdog fires.
            url = _wait_for_url_announcement(proc, timeout=30.0)
            assert url.startswith("http://")
            # Watchdog ticks every 1s; allow generous margin.
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pytest.fail("Subprocess did not exit after parent died")
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)


class TestSubprocessPluginIntegration:
    """End-to-end test: plugin spawns subprocess via the xdist controller path."""

    def test_xdist_active_spawns_subprocess(self, pytester, monkeypatch):
        """
        Simulating an xdist controller, the plugin should spawn the subprocess
        in ``pytest_configure`` and export ``INFERENCEGATE_URL`` so child tests
        receive the URL via the env-driven ``_GateHandle``.
        """
        # Without -n we cannot actually fork xdist workers from pytester (which
        # uses a sub-pytest); instead we set INFERENCEGATE_URL on the outer
        # process so ``runpytest_subprocess`` inherits it. This exercises
        # ``_GateHandle`` subprocess mode (gate=None) and validates that
        # ``inference_gate_url`` passes the inherited URL through unchanged.
        pytester.makepyfile("""
            import os
            def test_url_inherits_env(inference_gate_url):
                assert inference_gate_url == os.environ["INFERENCEGATE_URL"]
        """)
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        # Set on this process so the spawned sub-pytest inherits it.
        monkeypatch.setenv("INFERENCEGATE_URL", "http://127.0.0.1:9")
        result = pytester.runpytest_subprocess("-v", "-s")
        result.assert_outcomes(passed=1)
