"""
tests.test_regressions — one test per bug that shipped.

The E2B tests deserve a word. The E2B backend is the only execution backend
production allows, and it had never been exercised: CI runs the subprocess
backend, so 45 green tests said nothing about the code path the app refuses to
boot without. These tests cover it without a network call or an E2B account, by
checking the two things that were actually wrong — the package's call surface,
and how a failed execution is turned into text.
"""

import json
import pathlib
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


# ── Shared LLM output parsing ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        'Here is the JSON: {"a": 1}',
        # Greedy `\{.*\}` under DOTALL spanned from the first brace to the last,
        # so a closing sentence with a brace in it broke the parse. The
        # orchestrator raises when its plan will not parse — one stray brace
        # failed the entire run at step 1.
        '{"a": 1}\nNote: set {b} later.',
        '{"a": 1}\n{"b": 2}',
        '```json\n{"a": 1}\n```',
    ],
)
def test_json_survives_prose_on_either_side(raw):
    from core.llm_utils import extract_json, strip_fences

    assert json.loads(extract_json(strip_fences(raw))) == {"a": 1}


def test_nested_json_is_extracted_whole():
    from core.llm_utils import extract_json

    got = json.loads(extract_json('text {"a": {"b": [1, 2]}} tail {x}'))
    assert got == {"a": {"b": [1, 2]}}


def test_text_with_no_json_is_returned_for_a_clear_error():
    from core.llm_utils import extract_json

    assert extract_json("sorry, I cannot help") == "sorry, I cannot help"


# ── Generated deployment artifacts ────────────────────────────────────────────


def test_generated_api_types_categoricals_as_strings_not_floats():
    """
    The deployment agent typed every feature as float, categoricals included,
    in both the LLM prompt and the OpenAPI spec — while model_trainer had
    already recorded the real dtypes in feature_schema. The served pipeline
    expects raw strings for those columns, so the generated request model
    contradicted the model it was serving.
    """
    from agents.deployment_agent import _feature_fields, _generate_openapi_spec

    state = {
        "feature_schema": [
            {"name": "age", "dtype": "int64"},
            {"name": "income", "dtype": "float64"},
            {"name": "city", "dtype": "object"},
            {"name": "is_member", "dtype": "bool"},
        ]
    }
    spec = _generate_openapi_spec(
        "lightgbm", "classification", _feature_fields(state), "churn"
    )
    props = spec["paths"]["/predict"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]

    assert props["age"]["type"] == "integer"
    assert props["income"]["type"] == "number"
    assert props["city"]["type"] == "string"
    assert props["is_member"]["type"] == "boolean"


def test_every_feature_reaches_the_spec():
    """`feature_names[:20]` silently produced a spec missing later columns."""
    from agents.deployment_agent import _feature_fields, _generate_openapi_spec

    state = {
        "feature_schema": [
            {"name": f"f{i}", "dtype": "float64"} for i in range(30)
        ]
    }
    spec = _generate_openapi_spec("xgboost", "regression", _feature_fields(state), "y")
    props = spec["paths"]["/predict"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]

    assert len(props) == 30


def test_generated_dockerfile_runs_as_non_root():
    """The repo's own image is non-root; the one it hands users should be too."""
    from agents.deployment_agent import DOCKERFILE_TEMPLATE

    assert "USER appuser" in DOCKERFILE_TEMPLATE


# ── RAG context budget ────────────────────────────────────────────────────────


def test_one_oversized_doc_does_not_wipe_out_the_context():
    """
    The budget loop used `break`, so a single doc grown past max_chars returned
    no grounding at all for any query that ranked it first.
    """
    from core.rag import retriever

    class _KB:
        def retrieve(self, query, k=3):
            return [
                {"title": "big", "source": "big.md", "text": "x" * 5000},
                {"title": "small", "source": "small.md", "text": "useful guidance"},
            ]

    original = retriever.get_knowledge_base
    retriever.get_knowledge_base = lambda: _KB()
    try:
        ctx = retriever.retrieve_context("anything", max_chars=1800)
    finally:
        retriever.get_knowledge_base = original

    assert "useful guidance" in ctx


# ── Generated code artifacts must be code, not JSON ───────────────────────────


CODE_SAMPLE = "import os\nprint(1)\n"


@pytest.mark.parametrize(
    "wrapper",
    [
        lambda c: c,
        lambda c: json.dumps({"code": c}),
        lambda c: json.dumps({"pipeline.py": c}),
        lambda c: "```python\n" + c + "```",
        lambda c: "```json\n" + json.dumps({"code": c}) + "\n```",
    ],
)
def test_source_is_recovered_from_a_json_wrapped_reply(wrapper):
    """
    build_system_prompt appended "Respond ONLY with valid JSON" to every agent
    prompt, including the two that ask for raw Python. The model obeyed the
    JSON rule, so the pipeline.py and fastapi_endpoint.py in the downloadable
    "runnable bundle" were JSON documents with a .py name.
    """
    from core.llm_utils import extract_code

    assert extract_code(wrapper(CODE_SAMPLE)).strip() == CODE_SAMPLE.strip()


def test_unwrapping_never_mangles_code_that_merely_contains_a_dict():
    """Being wrong in this direction would corrupt working source."""
    from core.llm_utils import extract_code

    src = 'CONFIG = {"a": 1}\nprint(CONFIG)\n'
    assert extract_code(src).strip() == src.strip()


def test_code_prompts_do_not_demand_json():
    from agents.code_generator import PIPELINE_CODE_PROMPT
    from agents.deployment_agent import FASTAPI_PROMPT

    for prompt in (PIPELINE_CODE_PROMPT, FASTAPI_PROMPT):
        assert "valid JSON" not in prompt
        assert "raw file contents" in prompt


def test_schema_prompts_still_demand_json():
    from agents.data_analyst import SYSTEM_PROMPT

    assert "valid JSON" in SYSTEM_PROMPT


def test_trailing_fenced_block_after_the_module_is_cut():
    """
    Observed on a real run: the model wrote pipeline.py, then appended a fenced
    list of pinned requirements. strip_fences only helps when a fence wraps the
    whole reply, so the .py artifact carried a requirements block from line 499
    and would not parse.
    """
    import ast

    from core.llm_utils import extract_code

    reply = (
        "import os\n\n\ndef main():\n    return 1\n\n"
        "```\nscikit-learn==1.3.0\nlightgbm==4.0.0\n```\n"
    )
    out = extract_code(reply)
    ast.parse(out)  # raises if the fence survived
    assert "scikit-learn==1.3.0" not in out
    assert "def main" in out


def test_a_module_that_already_parses_is_left_alone():
    """Never cut working code — including a fence inside a docstring."""
    from core.llm_utils import extract_code

    src = 'def f():\n    """Usage:\n\n    ```\n    f()\n    ```\n    """\n    return 1\n'
    assert extract_code(src).strip() == src.strip()


# ── SHAP explainability ───────────────────────────────────────────────────────


def test_multiclass_shap_values_are_reduced_to_one_matrix():
    """
    Older SHAP returned a list per class and the code sliced [1]. Newer SHAP
    returns a single (samples, features, classes) ndarray, which is not a list,
    so the slice never ran and a 3-D array reached summary_plot — which reads
    3-D as *interaction* values and drew a one-feature interaction chart instead
    of the ranked feature-importance summary. It still looked like a plot, so
    nothing caught it until someone looked at the picture.
    """
    import numpy as np

    from agents.evaluator import _positive_class_values

    # New shap: 3-D ndarray.
    arr = np.zeros((10, 4, 2))
    arr[:, :, 1] = 7.0
    out = _positive_class_values(arr)
    assert out.ndim == 2 and out.shape == (10, 4)
    assert (out == 7.0).all(), "took the wrong class slice"

    # Old shap: list of per-class arrays.
    out = _positive_class_values([np.zeros((10, 4)), np.full((10, 4), 7.0)])
    assert out.shape == (10, 4) and (out == 7.0).all()

    # Regression / single output: already 2-D, left alone.
    out = _positive_class_values(np.full((10, 4), 3.0))
    assert out.shape == (10, 4) and (out == 3.0).all()


def test_shap_plot_palette_matches_the_console():
    """The plot is embedded in a dark page; a white figure looks pasted on."""
    from agents.evaluator import _PLOT_ACCENT, _PLOT_BG

    css = pathlib.Path(__file__).resolve().parents[1] / "web" / "app" / "globals.css"
    text = css.read_text(encoding="utf-8")
    assert _PLOT_BG in text, "plot background is not one of the CSS tokens"
    assert _PLOT_ACCENT in text, "plot accent is not one of the CSS tokens"
