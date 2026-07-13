"""
tests.test_agents — Unit tests for individual agent nodes.

Tests use synthetic data and mock LLM calls to run fast without API keys.
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def credit_fraud_csv(tmp_path):
    """Create a small synthetic credit fraud CSV for testing."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame(
        {
            "V1": np.random.randn(n),
            "V2": np.random.randn(n),
            "V3": np.random.randn(n),
            "Amount": np.random.exponential(scale=100, size=n),
            "Time": np.arange(n) * 100,
            "Class": np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
        }
    )
    csv_path = tmp_path / "test_fraud.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def base_state(credit_fraud_csv):
    """Minimal AgentState for testing."""
    return {
        "csv_path": credit_fraud_csv,
        "business_problem": "Predict fraudulent transactions. Target is Class (0=normal, 1=fraud).",
        "provider": "anthropic",
        "api_key": "test-key",
        "model_name": "claude-3-5-haiku-20241022",
        "pipeline_id": "test-pipeline-001",
        "status": "running",
        "current_step": "orchestrator",
        "error": None,
        "dataset_profile": None,
        "feature_result": None,
        "model_results": {},
        "evaluation_result": None,
        "deployment_artifacts": None,
        "logs": [],
    }


# ── Data Analyst tests ────────────────────────────────────────────────────────


class TestDataAnalyst:
    def test_profile_dataframe_numeric_detection(self, credit_fraud_csv):
        """Data Analyst correctly detects numeric columns."""
        from agents.data_analyst import _profile_dataframe

        df = pd.read_csv(credit_fraud_csv)
        profile = _profile_dataframe(df)

        assert "V1" in profile["numeric_cols"]
        assert "Amount" in profile["numeric_cols"]
        assert profile["n_rows"] == 200
        assert profile["n_cols"] == 6

    def test_profile_dataframe_missing_detection(self, tmp_path):
        """Data Analyst detects missing values."""
        from agents.data_analyst import _profile_dataframe

        df = pd.DataFrame(
            {
                "A": [1, None, 3, 4, 5],
                "B": ["x", "y", None, "y", "x"],
                "C": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        profile = _profile_dataframe(df)
        assert "A" in profile["missing_pct"]
        assert "B" in profile["missing_pct"]
        assert "C" not in profile["missing_pct"]

    def test_detect_imbalance_imbalanced(self, credit_fraud_csv):
        """Data Analyst correctly identifies imbalanced target."""
        from agents.data_analyst import _detect_imbalance

        df = pd.DataFrame({"target": [0] * 95 + [1] * 5})
        is_imbalanced, ratio = _detect_imbalance(df, "target")
        assert is_imbalanced is True
        assert ratio < 0.2

    def test_detect_imbalance_balanced(self):
        """Data Analyst correctly identifies balanced target."""
        from agents.data_analyst import _detect_imbalance

        df = pd.DataFrame({"target": [0] * 50 + [1] * 50})
        is_imbalanced, ratio = _detect_imbalance(df, "target")
        assert is_imbalanced is False

    @patch("agents.data_analyst.get_llm")
    def test_run_data_analyst_returns_profile(self, mock_get_llm, base_state):
        """Data Analyst node returns a populated dataset_profile."""
        # Mock LLM response
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"recommendations": ["Impute with median", "Scale numeric"], "notes": "Small dataset"}'
        )
        mock_get_llm.return_value = mock_llm

        # Inject pre-existing profile with target_column
        base_state["dataset_profile"] = {
            "target_column": "Class",
            "task_type": "classification",
        }

        from agents.data_analyst import run_data_analyst

        result = run_data_analyst(base_state)

        assert result.get("error") is None
        assert result["dataset_profile"]["n_rows"] == 200
        assert result["dataset_profile"]["target_column"] == "Class"
        assert len(result["logs"]) > 0


# ── Model Trainer tests ───────────────────────────────────────────────────────


class TestModelTrainer:
    def test_make_model_lightgbm_classification(self):
        """Model factory creates correct model type."""
        from agents.model_trainer import _make_model

        model = _make_model("lightgbm", "classification", scale_pos=1.0)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")

    def test_make_model_xgboost_regression(self):
        """Model factory creates regression model."""
        from agents.model_trainer import _make_model

        model = _make_model("xgboost", "regression", scale_pos=1.0)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_compute_metrics_classification(self):
        """Metric computation returns expected keys."""
        from agents.model_trainer import _make_model, _compute_metrics
        import numpy as np

        model = _make_model(
            "logistic_regression", "classification", scale_pos=1.0
        )
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1], 100)
        model.fit(X, y)
        metrics = _compute_metrics(model, X, y, "classification")

        assert "auc" in metrics
        assert "f1" in metrics
        assert "accuracy" in metrics
        assert all(0 <= v <= 1 for v in metrics.values())

    def test_compute_metrics_regression(self):
        """Regression metric computation returns expected keys."""
        from agents.model_trainer import _make_model, _compute_metrics
        import numpy as np

        model = _make_model("linear_regression", "regression", scale_pos=1.0)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model.fit(X, y)
        metrics = _compute_metrics(model, X, y, "regression")

        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["rmse"] >= 0


# ── core.llm_utils tests ──────────────────────────────────────────────────────


class TestLLMUtils:
    def test_strip_fences_json_block(self):
        """strip_fences removes ```json fences."""
        from core.llm_utils import strip_fences

        raw = '```json\n{"key": "value"}\n```'
        result = strip_fences(raw)
        assert result == '{"key": "value"}'

    def test_strip_fences_python_block(self):
        """strip_fences removes ```python fences."""
        from core.llm_utils import strip_fences

        raw = '```python\nprint("hello")\n```'
        result = strip_fences(raw)
        assert result == 'print("hello")'

    def test_strip_fences_no_fences(self):
        """strip_fences passes through content without fences."""
        from core.llm_utils import strip_fences

        raw = '{"key": "value"}'
        assert strip_fences(raw) == raw

    def test_extract_json_from_prose(self):
        """extract_json finds JSON embedded in prose."""
        from core.llm_utils import extract_json

        text = 'Sure, here is the result: {"winner": "lightgbm", "score": 0.94} Hope that helps!'
        result = extract_json(text)
        assert '"winner"' in result
        assert '"lightgbm"' in result

    def test_parse_structured_output_valid(self):
        """parse_structured_output returns typed Pydantic model."""
        from core.llm_utils import parse_structured_output
        from pydantic import BaseModel

        class MySchema(BaseModel):
            name: str
            score: float

        raw = '{"name": "lightgbm", "score": 0.94}'
        result = parse_structured_output(raw, MySchema)
        assert result.name == "lightgbm"
        assert result.score == pytest.approx(0.94)

    def test_parse_structured_output_with_fences(self):
        """parse_structured_output handles fenced JSON."""
        from core.llm_utils import parse_structured_output
        from pydantic import BaseModel

        class MySchema(BaseModel):
            value: int

        raw = '```json\n{"value": 42}\n```'
        result = parse_structured_output(raw, MySchema)
        assert result.value == 42


# ── core.providers tests ──────────────────────────────────────────────────────


class TestProviders:
    def test_provider_models_complete(self):
        """All expected providers have model lists."""
        from core.providers import PROVIDER_MODELS

        assert "anthropic" in PROVIDER_MODELS
        assert "openai" in PROVIDER_MODELS
        assert "groq" in PROVIDER_MODELS
        assert all(len(models) > 0 for models in PROVIDER_MODELS.values())

    def test_get_llm_unknown_provider_raises(self):
        """get_llm raises ValueError for unknown provider."""
        from core.providers import get_llm

        with pytest.raises(ValueError, match="Unknown provider"):
            get_llm("unknown_provider", api_key="test")

    def test_codegen_models_are_cheaper(self):
        """CODEGEN_MODELS are different from (cheaper than) defaults."""
        from core.providers import CODEGEN_MODELS, PROVIDER_DEFAULTS

        # Haiku is cheaper than Sonnet
        assert CODEGEN_MODELS["anthropic"] != PROVIDER_DEFAULTS["anthropic"]
        assert "haiku" in CODEGEN_MODELS["anthropic"].lower()
