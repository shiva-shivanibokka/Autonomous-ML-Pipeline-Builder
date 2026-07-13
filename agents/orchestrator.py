"""
agents.orchestrator — Orchestrator Agent node.

Reads the business problem + dataset metadata and produces a pipeline plan:
- Identifies the ML task type (classification / regression / time_series)
- Decides which preprocessing steps are needed
- Sets the primary evaluation metric
- Populates state so all downstream agents have clear direction
"""

from __future__ import annotations

import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.state import AgentState
from core.llm_utils import build_system_prompt, extract_content, safe_parse
from core.providers import get_llm

logger = logging.getLogger(__name__)


class OrchestratorPlan(BaseModel):
    task_type: str = Field(description="classification | regression | time_series")
    primary_metric: str = Field(description="auc | f1 | rmse | r2 | mape")
    target_column: str = Field(description="The column to predict")
    suggested_models: list[str] = Field(
        description="Models to train, from: lightgbm, xgboost, random_forest, mlp, logistic_regression, linear_regression"
    )
    reasoning: str = Field(description="1-3 sentence explanation of the plan")


SYSTEM_PROMPT = build_system_prompt(
    role="an expert ML systems architect who plans machine learning pipelines",
    context=(
        "Given a business problem description and a summary of the dataset, "
        "you will produce a structured pipeline plan. "
        "Detect whether the task is classification, regression, or time series. "
        "Choose 3-5 models appropriate for the task and dataset size. "
        "For classification: prefer lightgbm, xgboost, random_forest, mlp, logistic_regression. "
        "For regression: prefer lightgbm, xgboost, random_forest, mlp, linear_regression. "
        "For time_series: prefer lightgbm with lag features, linear_regression. "
        "Always include lightgbm and xgboost — they are fast and strong baselines. "
        "Choose the primary evaluation metric: auc or f1 for classification, rmse or r2 for regression, mape for time_series."
    ),
)


def run_orchestrator(state: AgentState) -> dict:
    """
    LangGraph node: plan the full pipeline from the business problem.

    Reads:  state["business_problem"], state["csv_path"]
    Writes: state["current_step"], state["logs"]
    Also enriches state["dataset_profile"] with task_type and target_column.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    logs = list(state.get("logs", []))
    logs.append(f"[{timestamp}] ORCHESTRATOR — Planning pipeline...")

    try:
        llm = get_llm(
            provider=state["provider"],
            api_key=state["api_key"],
            model=state["model_name"],
        )

        # Build the prompt
        dataset_profile = state.get("dataset_profile") or {}
        user_prompt = (
            f"Business problem: {state['business_problem']}\n\n"
            f"Dataset summary:\n"
            f"  - Rows: {dataset_profile.get('n_rows', 'unknown')}\n"
            f"  - Columns: {dataset_profile.get('n_cols', 'unknown')}\n"
            f"  - Numeric columns: {dataset_profile.get('numeric_cols', [])}\n"
            f"  - Categorical columns: {dataset_profile.get('categorical_cols', [])}\n"
            f"  - Datetime columns: {dataset_profile.get('datetime_cols', [])}\n"
            f"  - Missing values: {dataset_profile.get('missing_pct', {})}\n\n"
            "Produce a JSON pipeline plan with keys: "
            "task_type, primary_metric, target_column, suggested_models, reasoning."
        )

        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        raw = extract_content(response)
        plan = safe_parse(raw, OrchestratorPlan)

        if plan is None:
            raise ValueError(f"Could not parse orchestrator plan from: {raw[:300]}")

        logs.append(f"[{timestamp}] ORCHESTRATOR — Task type: {plan.task_type}")
        logs.append(
            f"[{timestamp}] ORCHESTRATOR — Models to train: {', '.join(plan.suggested_models)}"
        )
        logs.append(
            f"[{timestamp}] ORCHESTRATOR — Primary metric: {plan.primary_metric}"
        )
        logs.append(f"[{timestamp}] ORCHESTRATOR — Reasoning: {plan.reasoning}")
        logs.append(f"[{timestamp}] ORCHESTRATOR — Done.")

        # Merge into dataset_profile
        updated_profile = dict(dataset_profile)
        updated_profile["task_type"] = plan.task_type
        updated_profile["target_column"] = plan.target_column

        return {
            "current_step": "data_analyst",
            "dataset_profile": updated_profile,
            "logs": logs,
            # Store plan details as tags so model_trainer can read them
            "_orchestrator_plan": {
                "task_type": plan.task_type,
                "primary_metric": plan.primary_metric,
                "target_column": plan.target_column,
                "suggested_models": plan.suggested_models,
            },
        }

    except Exception as exc:
        logger.error("Orchestrator failed: %s", exc, exc_info=True)
        logs.append(f"[{timestamp}] ORCHESTRATOR — ERROR: {exc}")
        return {
            "error": f"Orchestrator failed: {exc}",
            "status": "failed",
            "logs": logs,
        }
