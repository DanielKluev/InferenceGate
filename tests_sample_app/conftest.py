"""
`tests_sample_app` is a stand-alone, top-level test directory that
demonstrates how a downstream pure-client app would consume InferenceGate's
pytest plugin.

Crucially, this directory does **not** share `tests/conftest.py` with the
Gate's own self-tests; it lives at the repository root specifically to prove
that the plugin works with zero project-side wiring.  A consumer's only
requirement is:

* install ``inference-gate`` (registers the plugin via the ``pytest11`` entry
  point);
* tell it where their cassettes live.

In a real consuming repository that's a one-liner in ``pyproject.toml``::

    [tool.pytest.ini_options]
    inferencegate_cache_dir = "tests/cassettes"

Here we set ``INFERENCEGATE_CACHE_DIR`` to an *absolute* path so the test is
runnable both from the repo root (``pytest tests_sample_app/``) and from
inside the directory (``cd tests_sample_app && pytest``).  A downstream
consumer typically would not need this conftest at all.
"""

from __future__ import annotations

import os
import pathlib

# Point the InferenceGate plugin at the dev cassettes shipped alongside the
# Gate's own tests.  A downstream project would normally use its own
# ``tests/cassettes`` directory.  Setting the env var here \u2014 at conftest
# import time \u2014 is early enough: the plugin reads it inside
# ``pytest_configure`` when spawning the subprocess.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("INFERENCEGATE_CACHE_DIR", str(_REPO_ROOT / "tests" / "cassettes"))
