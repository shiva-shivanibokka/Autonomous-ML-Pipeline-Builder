"""
agents.evaluator — Evaluator Agent node.

Compares all model training results, selects the best model with written
justification, runs SHAP explainability on the winner, and checks for
demographic bias on any sensitive features present.
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState, EvaluationResult
from core.llm_utils import build_system_prompt, extract_content, safe_parse
from core.providers import get_llm
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


SENSITIVE_FEATURES = {
    "gender",
    "sex",
    "race",
    "ethnicity",
    "age",
    "nationality",
    "religion",
    "disability",
    "marital_status",
}


class EvaluatorDecision(BaseModel):
    winner_model: str = Field(description="Name of the best model")
    ranking: list[str] = Field(description="All model names ordered best to worst")
    justification: str = Field(
        description="2-3 sentence explanation of why this model wins"
    )
    primary_metric: str = Field(description="The metric that drove the decision")
    bias_warnings: list[str] = Field(
        default=[], description="Any bias concerns flagged"
    )


SYSTEM_PROMPT = build_system_prompt(
    role="a senior ML engineer who evaluates and compares model training results",
    context=(
        "Given a comparison table of models and their evaluation metrics, "
        "select the best model for production deployment. "
        "Consider: metric performance, training time, memory usage, and bias risk. "
        "For classification: prefer AUC, then F1. "
        "For regression: prefer RMSE (lower is better), then R². "
        "Penalise models that failed (error field set). "
        "Consider whether simpler models (Logistic Regression, Linear Regression) "
        "perform within 2% of complex models — if so, recommend the simpler model "
        "for production (interpretability and latency matter). "
    ),
)


def _run_shap(
    model: Any,
    X_test: np.ndarray,
    feature_names: list[str],
    task_type: str,
    output_dir: str,
) -> str | None:
    """Run SHAP explainability on the winning model and save a summary plot."""
    try:
        import shap
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Choose explainer type
        if hasattr(model, "predict_proba") and hasattr(model, "feature_importances_"):
            # Tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(
                X_test[:200]
            )  # Limit to 200 rows for speed
        else:
            # Linear or neural models — use KernelExplainer with sample background
            background = shap.sample(X_test, 50)
            predict_fn = (
                model.predict_proba
                if hasattr(model, "predict_proba")
                else model.predict
            )
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_test[:50])

        # Handle multi-class shap_values (list of arrays)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_test[:200],
            feature_names=feature_names,
            plot_type="bar",
            show=False,
        )
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "shap_summary.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        return plot_path

    except Exception as exc:
        logger.warning("SHAP explainability failed (non-fatal): %s", exc)
        return None


def _check_bias(
    df: pd.DataFrame,
    target_col: str,
    winner_name: str,
    model: Any,
    feature_names: list[str],
) -> list[str]:
    """Flag potential bias if sensitive demographic features are in the dataset."""
    warnings = []
    present_sensitive = [col for col in df.columns if col.lower() in SENSITIVE_FEATURES]
    if present_sensitive:
        warnings.append(
            f"Sensitive features detected: {present_sensitive}. "
            f"Evaluate {winner_name} for disparate impact before deployment. "
            f"Consider fairness metrics: demographic parity, equalized odds."
        )
    return warnings


def run_evaluator(state: AgentState) -> dict:
    """
    LangGraph node: pick the best model, run SHAP, flag bias.

    Reads:  state["model_results"], state["dataset_profile"], state["feature_result"]
    Writes: state["evaluation_result"], state["current_step"], state["logs"]
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    logs = list(state.get("logs", []))
    logs.append(
        f"[{timestamp}] EVALUATOR — Comparing {len(state.get('model_results', {}))} models..."
    )

    try:
        model_results = state.get("model_results", {})
        profile = state.get("dataset_profile") or {}
        plan = state.get("_orchestrator_plan") or {}
        feature_result = state.get("feature_result") or {}

        task_type = profile.get("task_type", "classification")
        target_col = profile.get("target_column", "")
        primary_metric = plan.get(
            "primary_metric", "auc" if task_type == "classification" else "rmse"
        )

        # Build comparison table
        comparison_table = []
        for name, result in model_results.items():
            row = {
                "model": name,
                **result.get("metrics", {}),
                "train_time_s": result.get("train_time_seconds", 0),
                "memory_mb": result.get("memory_mb", 0),
                "failed": bool(result.get("error")),
            }
            comparison_table.append(row)

        # Ask LLM to evaluate
        llm = get_llm(
            provider=state["provider"],
            api_key=state["api_key"],
            model=state["model_name"],
        )

        table_str = "\n".join(
            f"  {row['model']}: "
            + ", ".join(f"{k}={v}" for k, v in row.items() if k != "model")
            for row in comparison_table
        )

        user_prompt = (
            f"Task type: {task_type}\n"
            f"Primary metric: {primary_metric}\n\n"
            f"Model results:\n{table_str}\n\n"
            "Select the best model for production. "
            "Return JSON with: winner_model, ranking, justification, primary_metric, bias_warnings."
        )

        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        raw = extract_content(response)
        decision = safe_parse(raw, EvaluatorDecision)

        if decision is None:
            # Fallback: pick by primary metric
            valid = {n: r for n, r in model_results.items() if not r.get("error")}
            ascending = primary_metric in ("rmse", "mape")
            winner_name = min(
                valid,
                key=lambda n: valid[n]["metrics"].get(
                    primary_metric, float("inf") if ascending else -float("inf")
                ),
            )
            decision = EvaluatorDecision(
                winner_model=winner_name,
                ranking=list(valid.keys()),
                justification=f"Selected {winner_name} by {primary_metric} metric.",
                primary_metric=primary_metric,
                bias_warnings=[],
            )

        logs.append(f"[{timestamp}] EVALUATOR — Winner: {decision.winner_model}")
        logs.append(
            f"[{timestamp}] EVALUATOR — Justification: {decision.justification}"
        )

        # Run SHAP on winner
        shap_plot_path = None
        winner_result = model_results.get(decision.winner_model, {})
        winner_model_obj = winner_result.get("model_object")

        if winner_model_obj is not None:
            logs.append(
                f"[{timestamp}] EVALUATOR — Running SHAP explainability on {decision.winner_model}..."
            )
            try:
                csv_path = feature_result.get("transformed_csv_path") or state.get(
                    "csv_path", ""
                )
                df = pd.read_csv(csv_path)
                X = df.drop(columns=[target_col])
                non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
                if non_numeric:
                    X = X.drop(columns=non_numeric)
                feature_names = X.columns.tolist()
                X_arr = X.values

                with tempfile.TemporaryDirectory() as tmpdir:
                    shap_plot_path = _run_shap(
                        model=winner_model_obj,
                        X_test=X_arr,
                        feature_names=feature_names,
                        task_type=task_type,
                        output_dir=tmpdir,
                    )
                    # Move to permanent per-run location
                    if shap_plot_path:
                        import shutil

                        out_dir = state.get("output_dir", "outputs")
                        os.makedirs(out_dir, exist_ok=True)
                        perm_path = os.path.join(out_dir, "shap_summary.png")
                        shutil.copy(shap_plot_path, perm_path)
                        shap_plot_path = perm_path

                # Bias check
                bias_warnings = _check_bias(
                    df,
                    target_col,
                    decision.winner_model,
                    winner_model_obj,
                    feature_names,
                )
                if bias_warnings:
                    decision.bias_warnings.extend(bias_warnings)
                    for w in bias_warnings:
                        logs.append(f"[{timestamp}] EVALUATOR — BIAS WARNING: {w}")

            except Exception as exc:
                logger.warning("Post-evaluation steps failed: %s", exc)

        logs.append(f"[{timestamp}] EVALUATOR — Done.")

        evaluation_result: EvaluationResult = {
            "winner_model": decision.winner_model,
            "ranking": decision.ranking,
            "justification": decision.justification,
            "primary_metric": decision.primary_metric,
            "shap_plot_path": shap_plot_path or "",
            "bias_warnings": decision.bias_warnings,
            "comparison_table": comparison_table,
        }

        return {
            "evaluation_result": evaluation_result,
            "current_step": "code_generator",
            "logs": logs,
        }

    except Exception as exc:
        logger.error("Evaluator failed: %s", exc, exc_info=True)
        logs.append(f"[{timestamp}] EVALUATOR — ERROR: {exc}")
        return {
            "error": f"Evaluator failed: {exc}",
            "status": "failed",
            "logs": logs,
        }
