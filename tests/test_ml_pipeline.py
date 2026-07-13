"""
tests.test_ml_pipeline — Phase B correctness: leakage-safe training, CV, and a
persisted model that actually runs on raw input.

No LLM is needed: model_trainer is pure sklearn, and the evaluator's only LLM call
(the justification narrative) is wrapped and falls back to a deterministic string.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _make_state(tmp_path: Path, n=400) -> dict:
    rng = np.random.RandomState(0)
    # Signal in x1/x2, a categorical, some missing values, plus an ID column.
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + x2 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "x1": x1,
            "x2": x2,
            "cat": rng.choice(["a", "b", "c"], size=n),
            "target": y,
        }
    )
    df.loc[rng.choice(n, 20, replace=False), "x1"] = np.nan  # missing values
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    return {
        "csv_path": str(csv),
        "provider": "anthropic",
        "api_key": "dummy",
        "model_name": "",
        "output_dir": str(out_dir),
        "dataset_profile": {
            "task_type": "classification",
            "target_column": "target",
            "is_imbalanced": False,
        },
        "_orchestrator_plan": {
            "task_type": "classification",
            "primary_metric": "auc",
            "target_column": "target",
            "suggested_models": ["lightgbm", "logistic_regression"],
        },
        "feature_result": {},
        "logs": [],
    }


def test_trainer_produces_fitted_pipelines_with_cv(tmp_path):
    from sklearn.pipeline import Pipeline

    from agents.model_trainer import run_model_trainer

    state = _make_state(tmp_path)
    out = run_model_trainer(state)

    results = out["model_results"]
    assert set(results) == {"lightgbm", "logistic_regression"}
    for name, r in results.items():
        assert r["error"] is None, f"{name} failed: {r['error']}"
        # Model object is a full prep+model Pipeline (not a bare estimator).
        assert isinstance(r["model_object"], Pipeline)
        assert "prep" in r["model_object"].named_steps
        # Cross-validation actually ran (leakage-free).
        assert r["cv_metric"] == "f1_weighted"
        assert 0.0 <= r["cv_mean"] <= 1.0
    # Raw input schema captured for inference, and a held-out SHAP sample exists.
    assert {c["name"] for c in out["feature_schema"]} == {"id", "x1", "x2", "cat"}
    assert len(out["shap_sample"]) > 0


def test_evaluator_selects_deterministically_and_persists_runnable_model(tmp_path):
    import joblib

    from agents.evaluator import _select_winner, run_evaluator
    from agents.model_trainer import run_model_trainer

    state = _make_state(tmp_path)
    state.update(run_model_trainer(state))

    # Deterministic selection is a pure function of the metrics — no LLM.
    winner, ranking = _select_winner(state["model_results"], "auc")
    assert winner in ("lightgbm", "logistic_regression")
    assert len(ranking) == 2

    out = run_evaluator(state)
    ev = out["evaluation_result"]
    assert ev["winner_model"] == winner  # evaluator agrees with the pure selector
    assert ev["justification"]  # some narrative present (LLM or fallback)

    # The persisted pipeline loads and predicts on RAW, unseen input (incl. NaN + new category).
    model_path = Path(state["output_dir"]) / "model.pkl"
    schema_path = Path(state["output_dir"]) / "feature_schema.json"
    assert model_path.exists() and schema_path.exists()

    pipe = joblib.load(model_path)
    raw = pd.DataFrame(
        [{"id": 1, "x1": np.nan, "x2": 0.5, "cat": "zzz"}]  # unknown category + missing
    )
    pred = pipe.predict(raw)
    assert pred.shape == (1,)  # runs end to end without a manual preprocessing step

    schema = json.loads(schema_path.read_text())
    assert {c["name"] for c in schema} == {"id", "x1", "x2", "cat"}
