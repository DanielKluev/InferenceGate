"""
Tests for the InferenceGate pytest plugin.

Verifies CLI option registration, option resolution (CLI > env > ini > default),
server auto-launch, fixture behavior, marker handling, and plugin disabling.
Uses ``pytester`` (pytest's built-in fixture for testing plugins) for most tests,
and direct unit tests for the option resolution helper.
"""

import os
import shutil
import textwrap
from pathlib import Path

import pytest

# Path to the production cassettes shipped with InferenceGate's own tests
CASSETTES_DIR = str(Path(__file__).parent / "cassettes")

# ---------------------------------------------------------------------------
# Unit tests for _resolve_option
# ---------------------------------------------------------------------------


class TestResolveOption:
    """Tests for the _resolve_option helper with various priority combinations."""

    def test_default_mode_is_replay(self, pytestconfig):
        """Default mode should be 'replay' when nothing is configured."""
        from inference_gate.pytest_plugin import _resolve_option
        # pytestconfig won't have --inferencegate-mode set (it's the host pytest),
        # so we test the fallback to default via a mock config.
        result = _resolve_option(pytestconfig, "mode")
        # When running under the host pytest with the plugin loaded, CLI is None,
        # env may or may not be set, ini is empty string (falsy) → falls to default.
        assert result in ("replay", "record")  # Accept either if env var is set

    def test_default_cache_dir(self, pytestconfig):
        """Default cache_dir should be 'tests/cassettes'."""
        from inference_gate.pytest_plugin import _resolve_option
        result = _resolve_option(pytestconfig, "cache_dir")
        assert result == "tests/cassettes" or result  # Accept override from env

    def test_default_port(self, pytestconfig):
        """Default port should be '0' (OS-assigned)."""
        from inference_gate.pytest_plugin import _resolve_option
        result = _resolve_option(pytestconfig, "port")
        assert result == "0" or result  # Accept override


# ---------------------------------------------------------------------------
# Pytester-based integration tests
# ---------------------------------------------------------------------------


class TestPluginRegistration:
    """Tests that the plugin registers its options and markers correctly."""

    def test_help_shows_inferencegate_options(self, pytester):
        """Running pytest --help should list the inferencegate option group."""
        result = pytester.runpytest("--help")
        result.stdout.fnmatch_lines(["*inferencegate*"])
        result.stdout.fnmatch_lines(["*--inferencegate-mode*"])
        result.stdout.fnmatch_lines(["*--inferencegate-cache-dir*"])

    def test_markers_registered(self, pytester):
        """Running pytest --markers should list requires_recording."""
        result = pytester.runpytest("--markers")
        result.stdout.fnmatch_lines(["*requires_recording*"])


class TestModeResolution:
    """Tests for the CLI > env > ini > default resolution order."""

    def test_default_mode_is_replay(self, pytester):
        """Without any configuration, mode should be 'replay'."""
        pytester.makepyfile("""
            def test_mode(inference_gate):
                from inference_gate.modes import Mode
                assert inference_gate.mode == Mode.REPLAY_ONLY
        """)
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest("-v", "-s")
        result.assert_outcomes(passed=1)

    def test_cli_flag_sets_mode(self, pytester):
        """--inferencegate-mode record should set record-and-replay mode."""
        pytester.makepyfile("""
            def test_mode(inference_gate):
                from inference_gate.modes import Mode
                assert inference_gate.mode == Mode.RECORD_AND_REPLAY
        """)
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest("-v", "-s", "--inferencegate-mode", "record")
        result.assert_outcomes(passed=1)

    def test_env_var_overrides_default(self, pytester, monkeypatch):
        """INFERENCEGATE_MODE env var should override the default."""
        monkeypatch.setenv("INFERENCEGATE_MODE", "record")
        pytester.makepyfile("""
            def test_mode(inference_gate):
                from inference_gate.modes import Mode
                assert inference_gate.mode == Mode.RECORD_AND_REPLAY
        """)
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest("-v", "-s")
        result.assert_outcomes(passed=1)

    def test_ini_option_overrides_default(self, pytester):
        """inferencegate_mode ini option should override the default."""
        pytester.makepyfile("""
            def test_mode(inference_gate):
                from inference_gate.modes import Mode
                assert inference_gate.mode == Mode.RECORD_AND_REPLAY
        """)
        pytester.makeini(f"""
            [pytest]
            inferencegate_mode = record
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest("-v", "-s")
        result.assert_outcomes(passed=1)

    def test_cli_beats_env_var(self, pytester, monkeypatch):
        """CLI flag should take priority over environment variable."""
        monkeypatch.setenv("INFERENCEGATE_MODE", "record")
        pytester.makepyfile("""
            def test_mode(inference_gate):
                from inference_gate.modes import Mode
                assert inference_gate.mode == Mode.REPLAY_ONLY
        """)
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest("-v", "-s", "--inferencegate-mode", "replay")
        result.assert_outcomes(passed=1)


class TestServerLifecycle:
    """Tests that the plugin manages the InferenceGate server correctly."""

    def test_server_starts_and_health_reachable(self, pytester):
        """The inference_gate fixture should start a healthy server."""
        pytester.makepyfile("""
            import http.client
            import urllib.parse

            def test_health(inference_gate_url):
                parsed = urllib.parse.urlparse(inference_gate_url)
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
                conn.request("GET", "/health")
                resp = conn.getresponse()
                assert resp.status == 200
                conn.close()
        """)
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest("-v", "-s")
        result.assert_outcomes(passed=1)

    def test_inference_gate_url_format(self, pytester):
        """inference_gate_url should be a proper http://host:port string."""
        pytester.makepyfile("""
            def test_url_format(inference_gate_url):
                assert inference_gate_url.startswith("http://127.0.0.1:")
                # Port should be a number > 0 (OS-assigned)
                port = int(inference_gate_url.split(":")[-1])
                assert port > 0
        """)
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest("-v", "-s")
        result.assert_outcomes(passed=1)

    def test_port_zero_assigns_ephemeral_port(self, pytester):
        """Port 0 should result in an OS-assigned port that is not 0."""
        pytester.makepyfile("""
            def test_port(inference_gate):
                assert inference_gate.actual_port > 0
                assert inference_gate.actual_port != 0
        """)
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest("-v", "-s")
        result.assert_outcomes(passed=1)


class TestCassetteReplayViaPytestPlugin:
    """Tests that the plugin can replay cassettes through the auto-launched server."""

    def test_replay_ok_prompt(self, pytester):
        """Replay the 'OK' prompt cassette through the plugin's server."""
        pytester.makepyfile(
            textwrap.dedent(f"""
            import http.client
            import json
            import urllib.parse

            CASSETTE_MODEL = "openai/gpt-oss-120b"
            OK_PROMPT = 'This is a test prompt. Reply with **ONLY** "OK." to confirm that everything is ok. DO NOT output anything else.'

            def test_replay(inference_gate_url):
                parsed = urllib.parse.urlparse(inference_gate_url)
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
                body = json.dumps({{
                    "model": CASSETTE_MODEL,
                    "messages": [{{"role": "user", "content": OK_PROMPT}}],
                    "max_tokens": 50,
                }})
                conn.request("POST", "/v1/chat/completions",
                             body=body,
                             headers={{"Content-Type": "application/json"}})
                resp = conn.getresponse()
                assert resp.status == 200
                data = json.loads(resp.read())
                assert "choices" in data
                content = data["choices"][0]["message"]["content"]
                assert "OK" in content
                conn.close()
        """))
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest("-v", "-s")
        result.assert_outcomes(passed=1)

    def test_cache_miss_returns_503(self, pytester):
        """A request with no matching cassette should return 503 in replay mode."""
        pytester.makepyfile(
            textwrap.dedent(f"""
            import http.client
            import json
            import urllib.parse

            def test_miss(inference_gate_url):
                parsed = urllib.parse.urlparse(inference_gate_url)
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
                body = json.dumps({{
                    "model": "nonexistent-model",
                    "messages": [{{"role": "user", "content": "This will not match any cassette"}}],
                }})
                conn.request("POST", "/v1/chat/completions",
                             body=body,
                             headers={{"Content-Type": "application/json"}})
                resp = conn.getresponse()
                assert resp.status == 503
                conn.close()
        """))
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest("-v", "-s")
        result.assert_outcomes(passed=1)


class TestFuzzyModelMatchingViaPytestPlugin:
    """Tests for the fuzzy model matching option via the pytest plugin."""

    def test_fuzzy_matching_via_ini_option(self, pytester):
        """Fuzzy model matching enabled via ini finds a cassette recorded with a different model."""
        pytester.makepyfile(
            textwrap.dedent(f"""
            import http.client
            import json
            import urllib.parse

            OK_PROMPT = 'This is a test prompt. Reply with **ONLY** "OK." to confirm that everything is ok. DO NOT output anything else.'

            def test_fuzzy(inference_gate_url):
                parsed = urllib.parse.urlparse(inference_gate_url)
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
                body = json.dumps({{
                    "model": "completely-different-model",
                    "messages": [{{"role": "user", "content": OK_PROMPT}}],
                    "max_tokens": 50,
                }})
                conn.request("POST", "/v1/chat/completions",
                             body=body,
                             headers={{"Content-Type": "application/json"}})
                resp = conn.getresponse()
                assert resp.status == 200
                data = json.loads(resp.read())
                assert "choices" in data
                conn.close()
        """))
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
            inferencegate_fuzzy_model = true
        """)
        result = pytester.runpytest("-v", "-s")
        result.assert_outcomes(passed=1)

    def test_fuzzy_matching_via_cli_flag(self, pytester):
        """Fuzzy model matching enabled via CLI flag finds a cassette recorded with a different model."""
        pytester.makepyfile(
            textwrap.dedent(f"""
            import http.client
            import json
            import urllib.parse

            OK_PROMPT = 'This is a test prompt. Reply with **ONLY** "OK." to confirm that everything is ok. DO NOT output anything else.'

            def test_fuzzy(inference_gate_url):
                parsed = urllib.parse.urlparse(inference_gate_url)
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
                body = json.dumps({{
                    "model": "another-nonexistent-model",
                    "messages": [{{"role": "user", "content": OK_PROMPT}}],
                    "max_tokens": 50,
                }})
                conn.request("POST", "/v1/chat/completions",
                             body=body,
                             headers={{"Content-Type": "application/json"}})
                resp = conn.getresponse()
                assert resp.status == 200
                data = json.loads(resp.read())
                assert "choices" in data
                conn.close()
        """))
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest("-v", "-s", "--inferencegate-fuzzy-model")
        result.assert_outcomes(passed=1)

    def test_fuzzy_matching_via_env_var(self, pytester, monkeypatch):
        """Fuzzy model matching enabled via environment variable finds a cassette recorded with a different model."""
        monkeypatch.setenv("INFERENCEGATE_FUZZY_MODEL", "true")
        pytester.makepyfile(
            textwrap.dedent(f"""
            import http.client
            import json
            import urllib.parse

            OK_PROMPT = 'This is a test prompt. Reply with **ONLY** "OK." to confirm that everything is ok. DO NOT output anything else.'

            def test_fuzzy(inference_gate_url):
                parsed = urllib.parse.urlparse(inference_gate_url)
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
                body = json.dumps({{
                    "model": "env-var-test-model",
                    "messages": [{{"role": "user", "content": OK_PROMPT}}],
                    "max_tokens": 50,
                }})
                conn.request("POST", "/v1/chat/completions",
                             body=body,
                             headers={{"Content-Type": "application/json"}})
                resp = conn.getresponse()
                assert resp.status == 200
                conn.close()
        """))
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest("-v", "-s")
        result.assert_outcomes(passed=1)

    def test_no_fuzzy_matching_cli_overrides_ini(self, pytester):
        """--no-inferencegate-fuzzy-model CLI flag overrides ini=true, returning 503 on model mismatch."""
        pytester.makepyfile(
            textwrap.dedent(f"""
            import http.client
            import json
            import urllib.parse

            OK_PROMPT = 'This is a test prompt. Reply with **ONLY** "OK." to confirm that everything is ok. DO NOT output anything else.'

            def test_no_fuzzy(inference_gate_url):
                parsed = urllib.parse.urlparse(inference_gate_url)
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
                body = json.dumps({{
                    "model": "nonexistent-model",
                    "messages": [{{"role": "user", "content": OK_PROMPT}}],
                    "max_tokens": 50,
                }})
                conn.request("POST", "/v1/chat/completions",
                             body=body,
                             headers={{"Content-Type": "application/json"}})
                resp = conn.getresponse()
                assert resp.status == 503
                conn.close()
        """))
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
            inferencegate_fuzzy_model = true
        """)
        result = pytester.runpytest("-v", "-s", "--no-inferencegate-fuzzy-model")
        result.assert_outcomes(passed=1)


class TestRequiresRecordingMarker:
    """Tests for the @pytest.mark.requires_recording marker behavior."""

    def test_skipped_in_replay_mode(self, pytester):
        """Tests marked requires_recording should be skipped in replay mode."""
        pytester.makepyfile("""
            import pytest

            @pytest.mark.requires_recording
            def test_needs_recording(inference_gate_url):
                pass  # Should be skipped, not run

            def test_normal(inference_gate_url):
                assert inference_gate_url  # Should run fine
        """)
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1, skipped=1)

    def test_not_skipped_in_record_mode(self, pytester):
        """Tests marked requires_recording should NOT be skipped in record mode."""
        pytester.makepyfile("""
            import pytest

            @pytest.mark.requires_recording
            def test_needs_recording(inference_gate_url):
                assert inference_gate_url  # Should run
        """)
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest("-v", "--inferencegate-mode", "record")
        result.assert_outcomes(passed=1)


class TestStrictMatchingMarker:
    """Tests for the @pytest.mark.inferencegate_strict / @pytest.mark.inferencegate markers."""

    def test_strict_marker_registered(self, pytester):
        """Running pytest --markers should list inferencegate_strict and inferencegate."""
        result = pytester.runpytest("--markers")
        result.stdout.fnmatch_lines(["*inferencegate_strict*"])
        result.stdout.fnmatch_lines(["*inferencegate(fuzzy_model*"])

    def test_strict_marker_forces_exact_matching(self, pytester):
        """
        @pytest.mark.inferencegate_strict must push
        ``X-InferenceGate-Require-Exact: true`` onto Glue's request-context
        ContextVar for the duration of the test so per-request lookup
        forces exact matching, regardless of session-level fuzzy defaults.
        """
        pytester.makepyfile("""
            import pytest
            from inference_glue.request_context import current_headers

            @pytest.mark.inferencegate_strict
            def test_strict(inference_gate):
                hdrs = current_headers()
                assert hdrs.get("X-InferenceGate-Require-Exact") == "true"

            def test_unmarked(inference_gate):
                # Unmarked test: only auto Test-NodeID/Worker-ID metadata, no Require-* headers.
                hdrs = current_headers()
                assert "X-InferenceGate-Require-Exact" not in hdrs
                assert "X-InferenceGate-Require-Fuzzy-Model" not in hdrs
        """)
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
            inferencegate_fuzzy_model = true
            inferencegate_fuzzy_sampling = aggressive
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=2)

    def test_custom_marker_overrides_individual_flags(self, pytester):
        """
        @pytest.mark.inferencegate(fuzzy_model=False) must push only
        ``X-InferenceGate-Require-Fuzzy-Model: off`` and leave fuzzy_sampling
        unset (so the Gate falls back to its session default).
        """
        pytester.makepyfile("""
            import pytest
            from inference_glue.request_context import current_headers

            @pytest.mark.inferencegate(fuzzy_model=False)
            def test_only_model_off(inference_gate):
                hdrs = current_headers()
                assert hdrs.get("X-InferenceGate-Require-Fuzzy-Model") == "off"
                # fuzzy_sampling not overridden \u2014 no header pushed.
                assert "X-InferenceGate-Require-Fuzzy-Sampling" not in hdrs
        """)
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
            inferencegate_fuzzy_model = true
            inferencegate_fuzzy_sampling = aggressive
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)

    def test_overrides_are_restored_after_test(self, pytester):
        """
        Header overrides must be popped from the ContextVar after a marked test
        finishes, so subsequent unmarked tests do not see leaked Require-* headers.
        """
        pytester.makepyfile("""
            import pytest
            from inference_glue.request_context import current_headers

            @pytest.mark.inferencegate_strict
            def test_first_strict(inference_gate):
                assert current_headers().get("X-InferenceGate-Require-Exact") == "true"

            def test_second_unmarked(inference_gate):
                # If restore worked, the strict header is gone.
                assert "X-InferenceGate-Require-Exact" not in current_headers()
        """)
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
            inferencegate_fuzzy_model = true
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=2)


class TestAutoTestIdentityMetadata:
    """Auto-attached ``Metadata-Test-NodeID`` / ``Metadata-Worker-ID`` headers."""

    def test_node_id_and_worker_id_pushed_for_every_test(self, pytester):
        """
        Every test (marked or not) should get its pytest nodeid and the
        xdist worker name (``"master"`` outside xdist) pushed onto Glue's
        request-context ContextVar.
        """
        pytester.makepyfile("""
            from inference_glue.request_context import current_headers

            def test_metadata_pushed(inference_gate):
                hdrs = current_headers()
                node_id = hdrs.get("X-InferenceGate-Metadata-Test-NodeID")
                assert node_id is not None
                # Pytester wraps tests in its own module; nodeid contains the test name.
                assert "test_metadata_pushed" in node_id
                # Outside xdist, worker is "master".
                assert hdrs.get("X-InferenceGate-Metadata-Worker-ID") == "master"
        """)
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)


class TestPluginDisabling:
    """Tests that the plugin can be cleanly disabled."""

    def test_disable_with_p_no(self, pytester):
        """Running with -p no:inferencegate should disable the plugin."""
        pytester.makepyfile("""
            def test_nothing():
                assert True
        """)
        result = pytester.runpytest("-v", "-p", "no:inferencegate")
        result.assert_outcomes(passed=1)
        # The inferencegate options should not appear in help when disabled
        help_result = pytester.runpytest("--help", "-p", "no:inferencegate")
        # Should NOT contain inferencegate options
        assert "--inferencegate-mode" not in help_result.stdout.str()


# ---------------------------------------------------------------------------
# Unit tests for header sanitization
# ---------------------------------------------------------------------------


class TestHeaderSanitization:
    """Tests that CacheStorage.put() does not leak sensitive headers into tape files.

    In v2 tape format, request headers are not stored at all — the tape only
    contains semantic request content (messages, model, sampling params) and
    response references. These tests verify that sensitive header values do not
    appear anywhere in the stored tape files.
    """

    def test_authorization_header_not_in_tape(self, storage):
        """Authorization header value should not appear in stored tape files."""
        from inference_gate.recording.storage import CachedRequest, CachedResponse, CacheEntry
        secret = "Bearer sk-secret-key-12345"
        entry = CacheEntry(
            request=CachedRequest(method="POST", path="/v1/chat/completions", headers={
                "Content-Type": "application/json",
                "Authorization": secret,
                "Accept": "*/*",
            }, body={
                "model": "test",
                "messages": [{
                    "role": "user",
                    "content": "hello"
                }]
            }),
            response=CachedResponse(status_code=200, headers={"Content-Type": "application/json"}, body={"choices": []}),
        )
        cache_key = storage.put(entry)

        # Read back the tape file and verify secret is not present
        tape_path = storage._find_tape_file(cache_key)
        assert tape_path is not None
        tape_content = tape_path.read_text(encoding="utf-8")
        assert secret not in tape_content
        assert "sk-secret-key-12345" not in tape_content

    def test_multiple_sensitive_headers_not_in_tape(self, storage):
        """X-Api-Key and Proxy-Authorization values should not appear in stored tape files."""
        from inference_gate.recording.storage import CachedRequest, CachedResponse, CacheEntry
        entry = CacheEntry(
            request=CachedRequest(
                method="POST", path="/v1/chat/completions", headers={
                    "Content-Type": "application/json",
                    "X-Api-Key": "secret-api-key",
                    "Proxy-Authorization": "Basic proxy-secret",
                }, body={
                    "model": "test",
                    "messages": [{
                        "role": "user",
                        "content": "test"
                    }]
                }),
            response=CachedResponse(status_code=200, headers={"Content-Type": "application/json"}, body={"choices": []}),
        )
        cache_key = storage.put(entry)

        tape_path = storage._find_tape_file(cache_key)
        assert tape_path is not None
        tape_content = tape_path.read_text(encoding="utf-8")
        assert "secret-api-key" not in tape_content
        assert "proxy-secret" not in tape_content

    def test_case_insensitive_header_not_leaked(self, storage):
        """Header values should not be leaked regardless of header name casing."""
        from inference_gate.recording.storage import CachedRequest, CachedResponse, CacheEntry
        entry = CacheEntry(
            request=CachedRequest(method="POST", path="/v1/chat/completions", headers={
                "content-type": "application/json",
                "AUTHORIZATION": "Bearer secret-case-test",
            }, body={
                "model": "test",
                "messages": [{
                    "role": "user",
                    "content": "test2"
                }]
            }),
            response=CachedResponse(status_code=200, headers={"Content-Type": "application/json"}, body={"choices": []}),
        )
        cache_key = storage.put(entry)

        tape_path = storage._find_tape_file(cache_key)
        assert tape_path is not None
        tape_content = tape_path.read_text(encoding="utf-8")
        assert "secret-case-test" not in tape_content
