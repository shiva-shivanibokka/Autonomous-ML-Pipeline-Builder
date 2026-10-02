"""
Reproduce each reported plumbing bug against a given checkout.

    python eval_sop/repro/repro_bugs.py <path-to-code-root>

Run it once against the original commit (8987d6f) and once against the
sop-eval branch. Each check prints BUG (the reported defect is present) or OK
(it is not). The checks only use functions that exist in both versions, so the
same script exercises both. No network, no real LLM: a scripted fake answers.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PLAN = {
    "task_type": "classification",
    "primary_metric": "f1",
    "target_column": "label",
    "suggested_models": ["logistic_regression", "mlp"],
    "reasoning": "repro",
}
FE_OK = (
    "import pandas as pd\nINPUT_CSV_PATH = '/data/input.csv'\n"
    "OUTPUT_CSV_PATH = '/data/processed.csv'\n"
    "df = pd.read_csv(INPUT_CSV_PATH)\ndf['x1_minus_x2'] = df['x1'] - df['x2']\n"
    "df.to_csv(OUTPUT_CSV_PATH, index=False)\n"
)
FE_NO_OUTPUT = "print('I forgot to write OUTPUT_CSV_PATH')\n"


class R:
    def __init__(self, c):
        self.content = c


class Fake:
    def __init__(self, fe_code=FE_OK):
        self.fe_code = fe_code
        self.prompts = []

    def invoke(self, messages):
        system, user = messages[0].content, messages[-1].content
        self.prompts.append(user)
        if "ML systems architect" in system:
            return R(json.dumps(PLAN))
        if "data quality" in system:
            return R(json.dumps({"recommendations": ["none"], "notes": ""}))
        if "feature engineering code" in system or "debugger" in system:
            return R(self.fe_code)
        return R(json.dumps({"justification": "ok"}))


def make_csv(d: Path) -> str:
    rng = np.random.RandomState(0)
    n = 300
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame(
        {"x1": x1, "x2": x2, "label": (x1 - x2 + rng.normal(scale=0.5, size=n) > 0).astype(int)}
    )
    p = d / "data.csv"
    df.to_csv(p, index=False)
    return str(p)


def sandbox_settings(s):
    s.execution_backend = "subprocess"
    s.e2b_api_key = ""
    s.allow_local_exec = True
    s.sandbox_timeout_seconds = 60


def run_graph(csv, out, fake):
    import pipeline.graph as g

    with patch("agents.orchestrator.get_llm", return_value=fake), patch(
        "agents.data_analyst.get_llm", return_value=fake
    ), patch("agents.feature_engineer.get_codegen_llm", return_value=fake), patch(
        "agents.evaluator.get_llm", return_value=fake
    ), patch("agents.model_trainer.log_training_run", return_value=""), patch(
        "agents.model_trainer.log_comparison_table"
    ), patch("sandbox.executor.settings") as s, patch.object(
        g, "run_code_generator", lambda st: {}
    ), patch.object(g, "run_deployment_agent", lambda st: {}):
        sandbox_settings(s)
        return g.build_graph().invoke(
            {
                "csv_path": csv,
                "business_problem": "Predict label",
                "provider": "groq",
                "api_key": "x",
                "model_name": "m",
                "output_dir": out,
                "model_results": {},
                "logs": [],
            }
        )


def report(name, bug, detail):
    print(f"{'BUG' if bug else 'OK '}  {name}: {detail}", flush=True)


def main():
    print(f"code root: {ROOT}")
    tmp = Path(tempfile.mkdtemp(prefix="repro_"))
    csv = make_csv(tmp)

    # 1+2. Plan propagation and orchestrator input, through the real graph.
    fake = Fake()
    final = run_graph(csv, str(tmp / "out"), fake)
    trained = sorted((final.get("model_results") or {}).keys())
    report(
        "B1 plan dropped between nodes",
        trained != sorted(PLAN["suggested_models"]),
        f"LLM planned {PLAN['suggested_models']}, trainer trained {trained}",
    )
    ev = final.get("evaluation_result") or {}
    report(
        "B1b evaluator ignores planned metric",
        ev.get("primary_metric") != PLAN["primary_metric"],
        f"planned f1, evaluator ranked on {ev.get('primary_metric')}",
    )
    p0 = fake.prompts[0]
    report(
        "B2 orchestrator sees no dataset profile",
        "'x1'" not in p0,
        "orchestrator prompt: " + p0.split("Dataset summary:")[1].split("Produce")[0].strip().replace("\n", " | "),
    )

    # 3. Feature engineering that writes nothing.
    from agents.feature_engineer import run_feature_engineer

    def fe_with(code, csv_):
        with patch("agents.feature_engineer.get_codegen_llm", return_value=Fake(fe_code=code)), patch(
            "sandbox.executor.settings"
        ) as s:
            sandbox_settings(s)
            return run_feature_engineer(
                {
                    "csv_path": csv_,
                    "provider": "groq",
                    "api_key": "x",
                    "dataset_profile": {"target_column": "label", "task_type": "classification"},
                    "logs": [],
                }
            )

    # 3a. Fresh directory: nothing was ever written next to the input.
    d3 = tmp / "b3a"
    d3.mkdir()
    csv3 = make_csv(d3)
    out = fe_with(FE_NO_OUTPUT, csv3)
    fr = out.get("feature_result") or {}
    report(
        "B3a FE without output silently falls back to raw CSV",
        not out.get("error") and fr.get("transformed_csv_path") == csv3,
        f"error={str(out.get('error'))[:70]!r} transformed_csv_path==raw input: {fr.get('transformed_csv_path') == csv3}",
    )

    # 3b. Same directory as an earlier successful run: is a STALE output reused?
    d3b = tmp / "b3b"
    d3b.mkdir()
    csv3b = make_csv(d3b)
    fe_with(FE_OK, csv3b)  # leaves data_processed.csv behind
    out = fe_with(FE_NO_OUTPUT, csv3b)
    fr = out.get("feature_result") or {}
    stale = not out.get("error") and fr.get("transformed_csv_path") not in (None, "", csv3b)
    report(
        "B3b FE without output silently reuses a stale output from an earlier run",
        stale,
        f"error={str(out.get('error'))[:70]!r} transformed_csv_path={Path(fr.get('transformed_csv_path') or '-').name}",
    )

    # 4. Winner selection on holdout vs CV.
    from agents.evaluator import _select_winner

    res = {
        "test_best": {"metrics": {"auc": 0.99}, "cv_mean": 0.70, "cv_metric": "roc_auc",
                      "model_object": object(), "error": None},
        "cv_best": {"metrics": {"auc": 0.80}, "cv_mean": 0.85, "cv_metric": "roc_auc",
                    "model_object": object(), "error": None},
    }
    w, _ = _select_winner(res, "auc")
    report("B4 winner chosen on holdout test metric", w == "test_best", f"winner={w}")

    # 4b. CV scorer vs planned metric.
    from agents.model_trainer import run_model_trainer

    plan_auc = {**PLAN, "primary_metric": "auc", "suggested_models": ["logistic_regression"]}
    with patch("agents.model_trainer.log_training_run", return_value=""), patch(
        "agents.model_trainer.log_comparison_table"
    ):
        tr = run_model_trainer(
            {
                "csv_path": csv,
                "provider": "groq",
                "dataset_profile": {"task_type": "classification", "target_column": "label"},
                # Both spellings so the check isolates the scorer, not the key bug.
                "_orchestrator_plan": plan_auc,
                "orchestrator_plan": plan_auc,
                "feature_result": {},
                "logs": [],
            }
        )
    cvm = tr["model_results"]["logistic_regression"]["cv_metric"]
    report("B4b CV scores a different metric than selection", cvm != "roc_auc", f"plan=auc, cv_metric={cvm}")

    # 5. Generated code inherits API keys.
    from sandbox.executor import _execute_subprocess

    code = "import os\nprint('LEAK=' + ','.join(sorted(k for k in os.environ if k.endswith('_API_KEY'))))\n"
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-repro-not-a-real-key"}):
        r = _execute_subprocess(code, "", 60)
    line = [ln for ln in r["stdout"].splitlines() if ln.startswith("LEAK=")]
    leaked = line[0][5:] if line else "?"
    report("B5 subprocess inherits API keys", "ANTHROPIC_API_KEY" in leaked, f"visible to generated code: [{leaked}]")

    # 6. leakage_warnings: is anything ever computed?
    src = (ROOT / "agents" / "feature_engineer.py").read_text(encoding="utf-8")
    report(
        "B6 leakage_warnings hard-coded to []",
        '"leakage_warnings": [],' in src,
        "literal present in agents/feature_engineer.py" if '"leakage_warnings": [],' in src else "computed",
    )


if __name__ == "__main__":
    main()
