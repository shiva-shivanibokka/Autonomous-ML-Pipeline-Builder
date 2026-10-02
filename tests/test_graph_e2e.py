"""
tests.test_graph_e2e — the plan has to survive the real LangGraph graph.

The older tests inject the orchestrator's plan straight into the state dict
they hand to run_model_trainer(), which bypasses LangGraph's state handling.
That is exactly where the plan used to be lost: the orchestrator returned it
under "_orchestrator_plan", a key not declared in AgentState, and LangGraph
drops undeclared keys between nodes. Every unit test passed while the LLM's
model and metric choice never reached the trainer.

These tests run the compiled graph end to end with a scripted fake LLM and
assert on what came out of the trainer and evaluator.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

FE_SCRIPT = """
import pandas as pd
INPUT_CSV_PATH = '/data/input.csv'
OUTPUT_CSV_PATH = '/data/processed.csv'
df = pd.read_csv(INPUT_CSV_PATH)
df['x1_minus_x2'] = df['x1'] - df['x2']
df.to_csv(OUTPUT_CSV_PATH, index=False)
print("CREATED_FEATURES: ['x1_minus_x2']")
print("DROPPED_FEATURES: []")
"""

# Deliberately NOT the trainer's defaults (lightgbm, xgboost, random_forest)
# and NOT the evaluator's default metric (auc), so a dropped plan is visible.
PLAN = {
    "task_type": "classification",
    "primary_metric": "f1",
    "target_column": "label",
    "suggested_models": ["logistic_regression", "mlp"],
    "reasoning": "test plan",
}


class _Resp:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Answers each agent by recognising its system prompt."""

    def __init__(self):
        self.prompts: list[str] = []

    def invoke(self, messages):
        system = messages[0].content
        user = messages[-1].content
        self.prompts.append(user)
        if "ML systems architect" in system:
            return _Resp(json.dumps(PLAN))
        if "data quality" in system:
            return _Resp(json.dumps({"recommendations": ["none"], "notes": ""}))
        if "feature engineering code" in system:
            return _Resp(FE_SCRIPT)
        return _Resp(json.dumps({"justification": "fine"}))


@pytest.fixture
def csv_path(tmp_path):
    rng = np.random.RandomState(0)
    n = 300
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "cat": rng.choice(["a", "b"], size=n),
            "label": (x1 - x2 + rng.normal(scale=0.5, size=n) > 0).astype(int),
        }
    )
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return str(p)


def _run_graph(csv_path, tmp_path, fake):
    import pipeline.graph as graph_mod

    with patch("agents.orchestrator.get_llm", return_value=fake), patch(
        "agents.data_analyst.get_llm", return_value=fake
    ), patch("agents.feature_engineer.get_codegen_llm", return_value=fake), patch(
        "agents.evaluator.get_llm", return_value=fake
    ), patch("agents.model_trainer.log_training_run", return_value=""), patch(
        "agents.model_trainer.log_comparison_table"
    ), patch("sandbox.executor.settings") as s, patch.object(
        graph_mod, "run_code_generator", lambda state: {}
    ), patch.object(graph_mod, "run_deployment_agent", lambda state: {}):
        s.execution_backend = "subprocess"
        s.e2b_api_key = ""
        s.allow_local_exec = True
        s.sandbox_timeout_seconds = 60
        graph = graph_mod.build_graph()
        return graph.invoke(
            {
                "csv_path": csv_path,
                "business_problem": "Predict label",
                "provider": "groq",
                "api_key": "x",
                "model_name": "m",
                "output_dir": str(tmp_path / "out"),
                "random_seed": 7,
                "model_results": {},
                "logs": [],
            }
        )


def test_plan_reaches_trainer_and_evaluator_through_the_graph(csv_path, tmp_path):
    fake = FakeLLM()
    final = _run_graph(csv_path, tmp_path, fake)

    assert not final.get("error"), final.get("error")
    assert final["orchestrator_plan"]["suggested_models"] == ["logistic_regression", "mlp"]
    # The trainer trained the PLANNED models, not its defaults.
    assert set(final["model_results"]) == {"logistic_regression", "mlp"}
    # CV scored the planned metric (f1 -> f1_weighted), not a hard-coded one.
    for r in final["model_results"].values():
        assert r["cv_metric"] == "f1_weighted"
    # The evaluator ranked on the planned metric, on CV.
    ev = final["evaluation_result"]
    assert ev["primary_metric"] == "f1"
    assert ev["selection_basis"] == "cv"
    # The feature engineer's output, not the raw CSV, was what got trained on.
    fr = final["feature_result"]
    assert fr["transformed_csv_path"] and fr["transformed_csv_path"] != csv_path
    assert "x1_minus_x2" in {c["name"] for c in final["feature_schema"]}


def test_orchestrator_sees_the_dataset_columns(csv_path, tmp_path):
    fake = FakeLLM()
    _run_graph(csv_path, tmp_path, fake)
    orchestrator_prompt = fake.prompts[0]
    assert "'label'" in orchestrator_prompt and "'x1'" in orchestrator_prompt
    assert "Rows: 300" in orchestrator_prompt


def test_orchestrator_rejects_a_target_that_is_not_a_column(csv_path, tmp_path):
    from agents.orchestrator import run_orchestrator

    bad = FakeLLM()
    with patch("agents.orchestrator.get_llm", return_value=bad), patch.dict(
        PLAN, {"target_column": "no_such_column"}
    ):
        out = run_orchestrator(
            {
                "csv_path": csv_path,
                "business_problem": "Predict label",
                "provider": "groq",
                "api_key": "x",
                "model_name": "m",
                "logs": [],
            }
        )
    assert out.get("error") and "no_such_column" in out["error"]


def test_feature_engineering_without_output_fails_loudly(tmp_path):
    """A script that runs cleanly but writes nothing must not fall back to raw data."""
    from unittest.mock import MagicMock

    from sandbox.executor import execute_with_retry

    csv = tmp_path / "d.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(csv, index=False)
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="print('still nothing')")
    with patch("sandbox.executor.settings") as s:
        s.execution_backend = "subprocess"
        s.e2b_api_key = ""
        s.allow_local_exec = True
        s.sandbox_timeout_seconds = 30
        res = execute_with_retry(
            "print('no output written')", str(csv), llm, max_retries=2, require_output=True
        )
    assert res["success"] is False
    assert res["output_csv_path"] == ""
    assert "did not write its output" in res["last_error"]


def test_generated_code_does_not_see_api_keys(tmp_path):
    from sandbox.executor import _execute_subprocess

    code = (
        "import os\n"
        "leaked = sorted(k for k in os.environ if 'KEY' in k or 'TOKEN' in k)\n"
        "print('LEAKED=' + ','.join(leaked))\n"
        "print('CWD=' + os.getcwd())\n"
    )
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-secret", "GROQ_API_KEY": "g"}):
        res = _execute_subprocess(code, "", 30)
    assert res["success"], res["stderr"]
    assert "LEAKED=\n" in res["stdout"] or res["stdout"].startswith("LEAKED=\n")
    assert str(tmp_path) not in res["stdout"]


def test_selection_uses_cv_not_holdout():
    from agents.evaluator import _select_winner

    results = {
        # Best on the held-out test set, worse on CV: must NOT win.
        "a": {"metrics": {"auc": 0.99}, "cv_mean": 0.70, "cv_metric": "roc_auc",
              "model_object": object(), "error": None},
        "b": {"metrics": {"auc": 0.80}, "cv_mean": 0.85, "cv_metric": "roc_auc",
              "model_object": object(), "error": None},
    }
    winner, ranking = _select_winner(results, "auc")
    assert winner == "b" and ranking == ["b", "a"]
