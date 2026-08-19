"""
tests.test_regressions — one test per bug that shipped.

The E2B tests deserve a word. The E2B backend is the only execution backend
production allows, and it had never been exercised: CI runs the subprocess
backend, so 45 green tests said nothing about the code path the app refuses to
boot without. These tests cover it without a network call or an E2B account, by
checking the two things that were actually wrong — the package's call surface,
and how a failed execution is turned into text.
"""

import sys

import pytest

from core.store import RunStore
from sandbox.executor import (
    _execute_subprocess,
    _format_execution_error,
    _output_csv_for,
    _rewrite_path_constant,
)

# ── E2B: the pinned package must expose what executor.py calls ────────────────


def test_e2b_package_exposes_the_api_the_executor_calls():
    """
    requirements.txt pinned e2b-code-interpreter==0.0.10 — yanked on PyPI —
    while sandbox/executor.py was written against the v1+ API. The pinned
    package exported CodeInterpreter with exec_cell()/.process/.filesystem;
    the code calls Sandbox with run_code()/.commands/.files. Nothing failed
    in CI because CI never touches this backend.
    """
    e2b = pytest.importorskip(
        "e2b_code_interpreter", reason="E2B extra not installed in this environment"
    )

    assert hasattr(e2b, "Sandbox"), (
        "e2b_code_interpreter has no Sandbox — the pin is on a version whose "
        "API does not match sandbox/executor.py"
    )
    for attr in ("run_code", "commands", "files", "kill"):
        assert hasattr(e2b.Sandbox, attr), f"Sandbox.{attr} missing from this version"


class _FakeExecutionError:
    """Mirrors e2b's ExecutionError: a single object, never a list."""

    name = "ValueError"
    value = "could not convert string to float: 'abc'"
    traceback = 'Traceback (most recent call last):\n  File "<cell>", line 3'


def test_execution_error_is_rendered_not_iterated():
    """
    The old code did `"\\n".join(str(e) for e in (execution.error or []))`.
    With no error that is harmlessly empty; with an error it raises TypeError —
    so the sandbox blew up on exactly the failures the self-correction loop was
    built to recover from.
    """
    text = _format_execution_error(_FakeExecutionError())
    assert "ValueError" in text
    assert "could not convert" in text
    assert "Traceback" in text


def test_no_execution_error_renders_empty():
    assert _format_execution_error(None) == ""


# ── Subprocess backend ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "declaration",
    [
        "INPUT_CSV_PATH = '/data/input.csv'",
        'INPUT_CSV_PATH = "/data/input.csv"',
        "INPUT_CSV_PATH='/data/input.csv'",
        "INPUT_CSV_PATH  =  '/data/input.csv'",
    ],
)
def test_csv_path_is_injected_whatever_quotes_the_model_used(declaration):
    """
    Injection used to be an exact-string replace against single quotes. Any
    other spelling silently did nothing, the script went looking for
    /data/input.csv on the host, and the self-correction loop then spent three
    LLM calls trying to fix a bug that belonged to the harness.
    """
    code = f"{declaration}\nimport pandas as pd\n"
    out = _rewrite_path_constant(code, "INPUT_CSV_PATH", "/tmp/real.csv")

    assert "/data/input.csv" not in out
    assert "/tmp/real.csv" in out


def test_missing_declaration_is_added_rather_than_ignored():
    out = _rewrite_path_constant("import pandas as pd\n", "INPUT_CSV_PATH", "/tmp/a.csv")
    assert out.startswith("INPUT_CSV_PATH = '/tmp/a.csv'")


def test_only_the_first_declaration_is_rewritten():
    code = "INPUT_CSV_PATH = '/data/input.csv'\nprint(INPUT_CSV_PATH)\n"
    out = _rewrite_path_constant(code, "INPUT_CSV_PATH", "/tmp/a.csv")
    assert out.count("INPUT_CSV_PATH") == 2  # the assignment and the print


def test_generated_code_runs_under_this_interpreter():
    """
    The subprocess backend invoked bare "python". Inside a venv that is the
    wrong interpreter, and on a slim container or most Linux hosts there is no
    "python" on PATH at all — the sandbox simply could not start.
    """
    result = _execute_subprocess("import sys; print(sys.executable)", "", 60)
    assert result["success"], result["stderr"]
    assert result["stdout"].strip() == sys.executable


@pytest.mark.parametrize(
    "csv_path,expected_tail",
    [
        ("/tmp/data.csv", "data_processed.csv"),
        # `csv_path.replace(".csv", "_processed.csv")` rewrote EVERY occurrence,
        # so a ".csv" in a parent directory got mangled too.
        ("/a.csv.d/b.csv", "b_processed.csv"),
        ("", ""),  # no CSV supplied — must not raise
    ],
)
def test_output_csv_path_is_derived_safely(csv_path, expected_tail):
    out = _output_csv_for(csv_path)
    if expected_tail:
        assert out.endswith(expected_tail)
        assert "a.csv.d" not in out or "/a.csv.d/" in out.replace("\\", "/")
    else:
        assert out == ""


# ── Run store ─────────────────────────────────────────────────────────────────


def test_update_on_an_unknown_id_persists_instead_of_vanishing(tmp_path):
    """
    update() built a record when the row was missing and then ran an UPDATE
    that matched nothing, so the write was dropped silently. That is the path a
    crashing run takes to report itself failed after a TTL sweep.
    """
    store = RunStore(db_path=tmp_path / "runs.db")
    store.update("deadbeef" * 4, 1000.0, status="failed", error="worker died")

    record = store.get("deadbeef" * 4)
    assert record is not None, "the failure write disappeared"
    assert record["status"] == "failed"
    assert record["error"] == "worker died"


def test_update_preserves_created_at_so_the_ttl_still_expires_it(tmp_path):
    store = RunStore(db_path=tmp_path / "runs.db", ttl_seconds=100)
    store.create("a" * 32, 1000.0)
    store.update("a" * 32, 1050.0, status="running")

    # A later create triggers the sweep; the old run is past its TTL by then.
    store.create("b" * 32, 2000.0)
    assert store.get("a" * 32) is None, "updating a run reset its age"
