"""
agents.state — the single source of truth for pipeline state.

Every LangGraph node receives the full AgentState and returns a partial
dict of updated fields. Only changed fields need to be returned.

Design decisions:
  - TypedDict (not Pydantic BaseModel) — required by LangGraph for state schema
  - All fields are Optional so nodes can safely update subsets
  - `logs` is a list[str] — each agent appends its own log lines
  - `model_results` is a dict keyed by model name — parallel sub-agents
    each write their own key without interfering with each other
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class ModelResult(TypedDict, total=False):
    """Result from a single model training sub-agent."""

    model_name: str
    model_object: Any  # The fitted model (not serialised in state)
    params: dict[str, Any]
    metrics: dict[str, float]  # auc, f1, precision, recall, rmse, r2, etc.
    feature_importance: dict[str, float]  # feature_name → importance score
    train_time_seconds: float
    memory_mb: float
    mlflow_run_id: str
    error: Optional[str]  # Set if training failed


class DatasetProfile(TypedDict, total=False):
    """Structured output from the Data Analyst Agent."""

    n_rows: int
    n_cols: int
    target_column: str
    task_type: str  # "classification" | "regression" | "time_series"
    is_imbalanced: bool
    class_ratio: Optional[float]  # minority / majority for classification
    missing_pct: dict[str, float]  # column → % missing
    numeric_cols: list[str]
    categorical_cols: list[str]
    datetime_cols: list[str]
    high_cardinality_cols: list[str]
    outlier_cols: list[str]
    recommended_preprocessing: list[str]  # Human-readable recommendations
    notes: str


class FeatureEngineeringResult(TypedDict, total=False):
    """Structured output from the Feature Engineer Agent."""

    preprocessing_code: str  # Python code string (executed in E2B)
    new_features_created: list[str]
    dropped_columns: list[str]
    leakage_warnings: list[str]  # Any data leakage risks flagged
    transformed_csv_path: str  # Path inside E2B sandbox


class EvaluationResult(TypedDict, total=False):
    """Structured output from the Evaluator Agent."""

    winner_model: str
    ranking: list[str]  # model names ordered best → worst
    justification: str
    primary_metric: str  # The metric used to pick the winner
    shap_plot_path: str
    bias_warnings: list[str]
    comparison_table: list[dict[str, Any]]


class DeploymentArtifacts(TypedDict, total=False):
    """Generated code artifacts from the Code Generator + Deployment Agents."""

    pipeline_code: str  # Full production-ready pipeline.py
    requirements_txt: str
    fastapi_code: str  # FastAPI inference endpoint
    dockerfile: str
    openapi_spec: dict[str, Any]
    mlflow_model_uri: str


class AgentState(TypedDict, total=False):
    """
    The complete pipeline state passed between all LangGraph nodes.

    Append-only fields (logs, model_results) are updated by each agent.
    All other fields are set once by the responsible agent and then read-only.
    """

    # ── User inputs ────────────────────────────────────────────────────────────
    csv_path: str  # Local path to the uploaded CSV
    business_problem: str  # Plain English problem description
    provider: str  # LLM provider ("anthropic", "openai", "groq")
    api_key: str  # User-supplied API key
    model_name: str  # LLM model name

    # ── Pipeline control ───────────────────────────────────────────────────────
    pipeline_id: str  # UUID for this run (for FastAPI tracking)
    status: str  # "running" | "completed" | "failed"
    current_step: str  # Human-readable current agent name
    error: Optional[str]  # Set if any agent fails fatally

    # ── Agent outputs ──────────────────────────────────────────────────────────
    dataset_profile: Optional[DatasetProfile]
    feature_result: Optional[FeatureEngineeringResult]
    model_results: dict[str, ModelResult]  # model_name → result
    evaluation_result: Optional[EvaluationResult]
    deployment_artifacts: Optional[DeploymentArtifacts]

    # ── Streaming log ─────────────────────────────────────────────────────────
    logs: list[str]  # Append-only: every agent adds lines here
